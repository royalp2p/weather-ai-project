import matplotlib.pyplot as plt

# Функция построения графиков
def plot_loss(losses1, losses2):
    plt.plot(losses1, label="Baseline")
    plt.plot(losses2, label="Improved")
    plt.legend()
    plt.title("Сравнение моделей")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("comparison.png")
    plt.show()