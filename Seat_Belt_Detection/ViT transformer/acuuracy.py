from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from main import true_labels,predicted_labels
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
for i in range(conf_matrix.shape[0]):
    class_acc = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
    print(f"Class {i}: Accuracy {class_acc}")