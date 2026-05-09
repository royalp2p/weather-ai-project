import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import ImprovedCNN
from dataset import WeatherDataset
from torch.utils.data import DataLoader

# загружаем модель
model = ImprovedCNN()
model.load_state_dict(torch.load("improved.pth"))
model.eval()

# тест датасет
test_dataset = WeatherDataset(
    r"D:\projectx\satellite_dataset_final\val"
)
test_loader = DataLoader(test_dataset, batch_size=16)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# считаем confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# отображаем
disp = ConfusionMatrixDisplay(cm, display_labels=["clear", "cloudy", "rain"])
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()