import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pickle

from dataset import WeatherDataset
from model import WeatherConvLSTM
from clearml import Task

# =========================
# CLEARML
# =========================

task = Task.init(
    project_name="Weather AI",
    task_name="ConvLSTM Training"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASETS
# =========================

train_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\train"
)

val_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\val"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================
# MODEL
# =========================

model = WeatherConvLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# HISTORY
# =========================

loss_history = []
acc_history = []
f1_history = []

# =========================
# TRAINING
# =========================

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    # =========================
    # VALIDATION
    # =========================

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    f1 = f1_score(all_labels, all_preds, average="weighted")

    acc_history.append(acc)
    f1_history.append(f1)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"F1: {f1:.4f}"
    )

    # CLEARML LOG
    task.get_logger().report_scalar("loss", "train", avg_loss, epoch)
    task.get_logger().report_scalar("accuracy", "val", acc, epoch)
    task.get_logger().report_scalar("f1", "val", f1, epoch)

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "convlstm_weather.pth")

# =========================
# SAVE HISTORY
# =========================

with open("loss.pkl", "wb") as f:
    pickle.dump(loss_history, f)

with open("accuracy.pkl", "wb") as f:
    pickle.dump(acc_history, f)

with open("f1.pkl", "wb") as f:
    pickle.dump(f1_history, f)

# =========================
# PLOTS
# =========================

plt.figure()
plt.plot(loss_history)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss_graph.png")
plt.close()

plt.figure()
plt.plot(acc_history)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("accuracy_graph.png")
plt.close()

plt.figure()
plt.plot(f1_history)
plt.title("F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.grid()
plt.savefig("f1_graph.png")
plt.close()

print("Training finished ✔")