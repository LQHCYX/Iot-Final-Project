import cv2
import time
import json
import serial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from pathlib import Path
from collections import deque, Counter


ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "best_model.pth"
IDX2CLASS_PATH = ROOT_DIR / "idx_to_class.pth"
LABELS_JSON_PATH = ROOT_DIR / "labels.json"

SERIAL_PORT = "COM4"
BAUD_RATE = 115200

TARGET_CHAR = "B"         
MIN_CONF = 0.60           
EMPTY_NAME = "empty"       

HISTORY_LEN = 8            
HISTORY_MIN_COUNT = 6      

IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_labels():
    if IDX2CLASS_PATH.exists():
        obj = torch.load(IDX2CLASS_PATH, map_location="cpu")
        if isinstance(obj, dict):
            labels = {int(k): str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            labels = {i: str(v) for i, v in enumerate(obj)}
        else:
            raise TypeError("idx_to_class.pth格式不对,既不是dict也不是list")
        print("标签映射(idx_to_class.pth):", labels)
        return labels

    if LABELS_JSON_PATH.exists():
        with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if all(isinstance(k, str) and k.isdigit() for k in raw.keys()):
                labels = {int(k): str(v) for k, v in raw.items()}
            else:
                labels = {int(v): str(k) for k, v in raw.items()}
        elif isinstance(raw, list):
            labels = {i: str(v) for i, v in enumerate(raw)}
        else:
            raise TypeError("labels.json格式不对,只能是dict或list")
        print("标签映射(labels.json):", labels)
        return labels

    labels = {0: "A", 1: "B", 2: "C", 3: "empty"}
    print("标签映射(默认):", labels)
    return labels


def build_model(num_classes, device):
    """和train.py里一致的resnet18结构"""
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model


def init_model(device):
    labels = load_labels()
    num_classes = len(labels)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件:{MODEL_PATH}")

    model = build_model(num_classes, device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("模型已加载:", MODEL_PATH)
    return model, labels

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def find_screen_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    h_img, w_img = gray.shape
    img_area = h_img * w_img

    max_box = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * 0.02:
            continue
        if area > max_area:
            max_area = area
            max_box = (x, y, w, h)

    if max_box is None:
        return None

    x, y, w, h = max_box
    pad = 10
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, w_img - x)
    h = min(h + 2 * pad, h_img - y)
    return x, y, w, h


def preprocess_roi(roi, device):
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return tensor


def decide_state(pred_label, conf):
    """
    根据预测结果决定要发给单片机的状态码:
    '1' -> 正确药物(TARGET_CHAR)
    '0' -> 错误药物(别的字母)
    '2' -> empty或置信度太低
    """
    if conf < MIN_CONF or pred_label == EMPTY_NAME:
        return "2"
    if pred_label == TARGET_CHAR:
        return "1"
    return "0"


def vote_and_send(state_code, history, ser, last_sent, pred_label, conf):
  
    history.append(state_code)

    send_code = None
    if len(history) == history.maxlen:
        most_common, count = Counter(history).most_common(1)[0]
        if count >= HISTORY_MIN_COUNT and most_common != last_sent:
            send_code = most_common

    if send_code is not None and ser is not None and ser.is_open:
        ser.write(send_code.encode("utf-8"))
        print(f"已发送到单片机:{send_code} 当前预测:{pred_label} conf={conf:.2f}")
        last_sent = send_code

    return last_sent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model, labels = init_model(device)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("串口已打开:", SERIAL_PORT)
        ser.write(b"2")
    except Exception as e:
        print("打开串口失败:", e)
        ser = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    history = deque(maxlen=HISTORY_LEN)
    last_sent = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头失败")
                break

            roi_box = find_screen_roi(frame)
            if roi_box is not None:
                x, y, w, h = roi_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = frame[y:y + h, x:x + w]
            else:
                roi = frame

            with torch.no_grad():
                input_tensor = preprocess_roi(roi, device)
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            pred_label = labels[pred_idx]

            state_code = decide_state(pred_label, conf)
            last_sent = vote_and_send(
                state_code, history, ser, last_sent, pred_label, conf
            )

            text = f"Pred:{pred_label} {conf:.2f}"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imshow("ABC Detector + MCU", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()
        print("已退出")


if __name__ == "__main__":
    main()
