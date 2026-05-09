import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pickle

from dataset import WeatherDataset
from model import BaselineCNN, ImprovedCNN

from clearml import Task

# CLEARML

task = Task.init(
    project_name="Weather AI",
    task_name="Satellite Forecast Training"
)

# DATASETS

# TRAIN
train_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\train"
)

# VALIDATION
val_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\val"
)

# DATALOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

# MODELS

models = {
    "baseline": BaselineCNN(),
    "improved": ImprovedCNN()
}

# LOSS FUNCTION

criterion = nn.CrossEntropyLoss()

# HISTORIES

loss_history = {}
accuracy_history = {}
f1_history = {}

# TRAINING

for name, model in models.items():

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    losses = []
    acc_list = []
    f1_list = []

    print(f"\n==============================")
    print(f"Обучение модели: {name}")
    print(f"==============================\n")

    # EPOCHS

    for epoch in range(5):

        model.train()

        total_loss = 0

        # TRAIN LOOP

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # LOSS

        avg_loss = total_loss / len(train_loader)

        losses.append(avg_loss)

        # VALIDATION

        model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)

                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

                all_preds.extend(preds.numpy())

                all_labels.extend(labels.numpy())

        # METRICS

        acc = correct / total

        f1 = f1_score(
            all_labels,
            all_preds,
            average="weighted"
        )

        acc_list.append(acc)

        f1_list.append(f1)

        # PRINT

        print(
            f"{name} | "
            f"Epoch {epoch+1} | "
            f"Loss={avg_loss:.4f} | "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        # CLEARML LOGGING

        task.get_logger().report_scalar(
            title=name,
            series="loss",
            value=avg_loss,
            iteration=epoch
        )

        task.get_logger().report_scalar(
            title=name,
            series="accuracy",
            value=acc,
            iteration=epoch
        )

        task.get_logger().report_scalar(
            title=name,
            series="f1_score",
            value=f1,
            iteration=epoch
        )

    # SAVE HISTORY

    loss_history[name] = losses

    accuracy_history[name] = acc_list

    f1_history[name] = f1_list

    # FINAL VALIDATION

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    print(f"\n{name} FINAL ACCURACY: {accuracy:.4f}\n")

    # SAVE MODEL

    torch.save(
        model.state_dict(),
        f"{name}.pth"
    )

# SAVE HISTORIES

with open("loss.pkl", "wb") as f:
    pickle.dump(loss_history, f)

with open("accuracy.pkl", "wb") as f:
    pickle.dump(accuracy_history, f)

with open("f1.pkl", "wb") as f:
    pickle.dump(f1_history, f)

print("\n===================================")
print("Обучение завершено")
print("===================================\n")