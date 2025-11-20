import cv2
import torch
import numpy as np

from collections import deque, Counter
from torchvision import models, transforms
from PIL import Image

# 直接用train.py里的常量,确保和训练时一致
from train import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# 验证时的预处理
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model_for_infer(num_classes, device):
    model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model = model.to(device)
    model.eval()
    return model


def load_model(model_path="best_model.pth", idx_path="idx_to_class.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_to_class = torch.load(idx_path)
    num_classes = len(idx_to_class)

    model = build_model_for_infer(num_classes, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    print("类别映射:", idx_to_class)
    print("使用设备:", device)

    return model, idx_to_class, device


def find_phone_roi(frame):
    """在整帧里找最亮的区域(手机屏幕),返回裁剪后的roi和框坐标."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 阈值分割,屏幕一般比较亮
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 闭运算,把小洞补上
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # 找面积最大的亮区域
    cnt = max(contours, key=cv2.contourArea)
    h, w = gray.shape
    area = cv2.contourArea(cnt)

    # 面积太小就认为没找到屏幕
    if area < 0.01 * h * w:
        return None, None

    x, y, bw, bh = cv2.boundingRect(cnt)

    # 稍微往外扩一点,避免裁得太紧
    pad_x = int(0.1 * bw)
    pad_y = int(0.1 * bh)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + bw + pad_x)
    y2 = min(frame.shape[0], y + bh + pad_y)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def preprocess_frame(frame):
    """返回(输入张量,屏幕框坐标).如果没检测到屏幕,返回(None,None)."""
    roi, box = find_phone_roi(frame)
    if roi is None:
        return None, None

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = eval_transform(pil_img).unsqueeze(0)  # [1,C,H,W]
    return tensor, box


def main():
    model, idx_to_class, device = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 用一个滑动窗口做多帧投票,平滑预测
    history = deque(maxlen=10)
    last_label = "empty"
    last_prob = 1.0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头失败")
                break

            input_tensor, box = preprocess_frame(frame)

            if input_tensor is None:
                # 没找到亮屏幕,直接认为是empty
                current_label = "empty"
                max_prob = 1.0
            else:
                input_tensor = input_tensor.to(device)
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                current_label = idx_to_class[pred_idx]
                max_prob = float(probs[pred_idx])

                # 画出检测到的屏幕框,方便你调试看ROI
                if box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

            # 记录历史标签
            history.append(current_label)

            # 多帧投票,减少A/B/C来回跳
            if len(history) >= 5:
                major, count = Counter(history).most_common(1)[0]
                final_label = major

                # 对empty严格一点:如果大多数是empty但当前置信度不高,
                # 就保持上一次的非empty标签,减少远距离抖动
                if final_label == "empty" and max_prob < 0.6:
                    final_label = last_label
            else:
                final_label = current_label

            last_label = final_label
            last_prob = max_prob

            text = f"Pred:{final_label} {last_prob:.2f}"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imshow("ABC Detector", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC退出
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
