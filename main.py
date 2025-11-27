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

# 路径相关
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "best_model.pth"
IDX2CLASS_PATH = ROOT_DIR / "idx_to_class.pth"
LABELS_JSON_PATH = ROOT_DIR / "labels.json"

# 串口相关
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

# 业务逻辑相关
TARGET_CHAR = "A"       # 正确药物标签
MIN_CONF = 0.60         # 最低置信度
EMPTY_NAME = "empty"    # 空集类别名(和训练时保持一致)

# 串口状态投票
HISTORY_LEN = 8         # 保存最近8帧的状态码
HISTORY_MIN_COUNT = 6   # 至少有6帧一致才发送

# 图像预处理
IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_labels():
    """优先从idx_to_class.pth加载标签映射,否则尝试labels.json,再否则用默认"""
    if IDX2CLASS_PATH.exists():
        obj = torch.load(IDX2CLASS_PATH, map_location="cpu")
        if isinstance(obj, dict):
            labels = {int(k): str(v) for k, v in obj.items()
                      }  # 可能是{0:"A",1:"B",...}
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
            # 可能是{"0":"A","1":"B"}或{"A":0,"B":1}
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

    # 实在找不到就用默认
    labels = {0: "A", 1: "B", 2: "C", 3: "empty"}
    print("标签映射(默认):", labels)
    return labels


def build_model(num_classes, device):
    """和train.py里一致的resnet18结构"""
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    model.eval()
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


# ========== 屏幕检测与预处理(参考predict_camera逻辑) ==========

def find_phone_roi(frame):
    """
    在整帧里找最亮的区域(手机屏幕),返回裁剪后的roi和框坐标.
    ROI找不到时返回(None,None)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 简单阈值+形态学,和camera版本一致
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    # 最大轮廓认为是屏幕
    cnt = max(contours, key=cv2.contourArea)
    h, w = gray.shape
    area = cv2.contourArea(cnt)

    # 屏幕太小直接忽略
    if area < 0.01 * h * w:
        return None, None

    x, y, bw, bh = cv2.boundingRect(cnt)

    # 按比例扩一点边界,比固定10像素更鲁棒
    pad_x = int(0.1 * bw)
    pad_y = int(0.1 * bh)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + bw + pad_x)
    y2 = min(frame.shape[0], y + bh + pad_y)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def preprocess_roi(roi, device):
    """把ROI转成模型需要的tensor"""
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return tensor


# ========== 业务逻辑:标签->状态码,状态投票,串口发送 ==========

def decide_state(pred_label, conf):
    """
    根据预测结果决定要发给单片机的状态码:
    "1" -> 正确药物(TARGET_CHAR)
    "0" -> 错误药物(别的字母)
    "2" -> empty或置信度太低
    """
    if conf < MIN_CONF or pred_label == EMPTY_NAME:
        return "2"
    if pred_label == TARGET_CHAR:
        return "1"
    return "0"


def vote_and_send(state_code, history, ser, last_sent, pred_label, conf):
    """
    串口侧的多数投票:
    1.把本帧state_code塞进history
    2.当history满了以后,做一次投票
    3.如果投票结果和上次发送的不一样,并且满足计数门槛,就发
    """
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


# ========== 主循环 ==========

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model, labels = init_model(device)

    # 打开串口
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("串口已打开:", SERIAL_PORT)
        # 先发一个"2"让灯和蜂鸣器全部关闭
        ser.write(b"2")
    except Exception as e:
        print("打开串口失败:", e)
        ser = None

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 状态码投票(history)和标签投票(label_history)分开
    history = deque(maxlen=HISTORY_LEN)     # "0/1/2"状态码
    label_history = deque(maxlen=10)        # 预测标签"A/B/C/empty"

    last_sent = None
    last_label = EMPTY_NAME
    last_prob = 1.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头失败")
                break

            # 1.先用camera里那套逻辑找手机屏幕
            roi, box = find_phone_roi(frame)

            if roi is None:
                # 找不到屏幕,直接当empty,并且不走模型推理
                current_label = EMPTY_NAME
                max_prob = 1.0
            else:
                # 找到了屏幕,才用ROI送进模型
                input_tensor = preprocess_roi(roi, device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                pred_idx = int(np.argmax(probs))
                max_prob = float(probs[pred_idx])
                current_label = labels[pred_idx]

                # 画框
                if box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2.标签层面的多数投票(跟predict_camera逻辑保持一致)
            label_history.append(current_label)

            if len(label_history) >= 5:
                major, count = Counter(label_history).most_common(1)[0]
                final_label = major

                # 如果多数投票给empty但置信度又很低,用上一帧的标签稳定一下
                if final_label == EMPTY_NAME and max_prob < 0.6:
                    final_label = last_label
            else:
                final_label = current_label

            last_label = final_label
            last_prob = max_prob

            # 3.根据“平滑后的标签+置信度”决定要不要给单片机发状态码
            state_code = decide_state(final_label, last_prob)
            last_sent = vote_and_send(
                state_code, history, ser, last_sent, final_label, last_prob
            )

            # 4.在画面上显示结果
            text = f"Pred:{final_label} {last_prob:.2f}"
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
