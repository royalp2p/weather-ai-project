import pickle
import matplotlib.pyplot as plt

# загружаем данные
with open("accuracy.pkl", "rb") as f:
    data = pickle.load(f)

# строим графики
plt.plot(data["baseline"], label="Baseline")
plt.plot(data["improved"], label="Improved")

plt.legend()
plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("accuracy.png")
plt.show()