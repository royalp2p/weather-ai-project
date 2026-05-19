import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from torch.utils.data import DataLoader

from model import ImprovedCNN
from dataset import WeatherDataset

from clearml import Task

# CLEARML

task = Task.init(
    project_name="Weather AI",
    task_name="Confusion Matrix"
)

# MODEL

model = ImprovedCNN()

model.load_state_dict(
    torch.load("improved.pth")
)

model.eval()

# DATASET

val_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\val"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

# PREDICTIONS

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# CONFUSION MATRIX

cm = confusion_matrix(
    all_labels,
    all_preds
)

# DISPLAY

fig, ax = plt.subplots(figsize=(7, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["clear", "cloudy", "rain"]
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

# SEND TO CLEARML

task.get_logger().report_matplotlib_figure(
    title="Evaluation",
    series="Confusion Matrix",
    figure=fig,
    iteration=0
)

print("Confusion matrix saved and uploaded to ClearML")

plt.show()