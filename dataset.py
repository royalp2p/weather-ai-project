import os
import cv2
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, folder, sequence_length=3):
        self.samples = []
        self.sequence_length = sequence_length

        classes = {
            "clear": 0,
            "cloudy": 1,
            "rain": 2
        }

        for class_name in classes:
            class_path = os.path.join(folder, class_name)

            files = sorted(os.listdir(class_path))

            # делаем последовательности
            for i in range(len(files) - sequence_length + 1):
                seq = files[i:i + sequence_length]

                seq_paths = [
                    os.path.join(class_path, x)
                    for x in seq
                ]

                self.samples.append(
                    (seq_paths, classes[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]

        images = []

        for path in image_paths:
            img = cv2.imread(path)

            img = cv2.resize(img, (128, 128))
            img = img / 255.0

            img = torch.tensor(
                img,
                dtype=torch.float32
            ).permute(2, 0, 1)

            images.append(img)

        # объединяем 3 картинки
        image = torch.cat(images, dim=0)

        label = torch.tensor(label)

        return image, label