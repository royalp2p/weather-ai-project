import pickle
import matplotlib.pyplot as plt

# =========================================
# LOAD LOSS
# =========================================

with open("loss.pkl", "rb") as f:
    loss_data = pickle.load(f)

# =========================================
# LOAD ACCURACY
# =========================================

with open("accuracy.pkl", "rb") as f:
    accuracy_data = pickle.load(f)

# =========================================
# LOAD F1
# =========================================

with open("f1.pkl", "rb") as f:
    f1_data = pickle.load(f)

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8, 5))

plt.plot(
    loss_data["baseline"],
    label="Baseline"
)

plt.plot(
    loss_data["improved"],
    label="Improved"
)

plt.title("Loss Comparison")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("loss_comparison.png")

plt.close()

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8, 5))

plt.plot(
    accuracy_data["baseline"],
    label="Baseline"
)

plt.plot(
    accuracy_data["improved"],
    label="Improved"
)

plt.title("Accuracy Comparison")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("accuracy_comparison.png")

plt.close()

# =========================================
# F1 GRAPH
# =========================================

plt.figure(figsize=(8, 5))

plt.plot(
    f1_data["baseline"],
    label="Baseline"
)

plt.plot(
    f1_data["improved"],
    label="Improved"
)

plt.title("F1 Score Comparison")

plt.xlabel("Epoch")

plt.ylabel("F1 Score")

plt.legend()

plt.grid(True)

plt.savefig("f1_comparison.png")

plt.close()

print("\n===================================")
print("Графики успешно сохранены")
print("===================================\n")