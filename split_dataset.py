import os
import shutil
import random

# откуда берём
source = "data/train"

# куда копируем
train_target = "data_split/train"
test_target = "data_split/test"

split_ratio = 0.8  # 80% train, 20% test

for class_name in os.listdir(source):
    class_path = os.path.join(source, class_name)

    files = os.listdir(class_path)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]

    # создаём папки
    os.makedirs(os.path.join(train_target, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_target, class_name), exist_ok=True)

    # копируем train
    for f in train_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(train_target, class_name, f)
        )

    # копируем test
    for f in test_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(test_target, class_name, f)
        )

print("Готово! Датасет разделён")