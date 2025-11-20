import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models


IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes, device):

    model = models.resnet18(pretrained=True)


    for p in model.parameters():
        p.requires_grad = False


    for name, p in model.named_parameters():
        if "layer4" in name or "fc" in name:
            p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double().item() / total
    return epoch_loss, epoch_acc


def main(args):
    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"数据集目录不存在:{data_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print("类别顺序:", class_names)

    val_ratio = args.val_ratio
    num_samples = len(full_dataset)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    val_dataset.dataset.transform = val_transform

  
    train_indices = train_dataset.indices 
    all_targets = np.array(full_dataset.targets)
    train_labels = all_targets[train_indices]

    class_count = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_count + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(num_classes, device)


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

  
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_corrects.double().item() / total

        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch[{epoch+1}/{args.epochs}] "
              f"train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} "
              f"val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print("训练完成!最佳验证准确率:", best_val_acc)

  
    torch.save(best_state, "best_model.pth")

  
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    torch.save(idx_to_class, "idx_to_class.pth")
    print("已保存best_model.pth和idx_to_class.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Data", help="数据集根目录")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
