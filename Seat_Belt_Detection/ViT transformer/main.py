import torch
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

train_dir = "/mnt/DATA/CS24S001/God-dataset/train"
valid_dir = "/mnt/DATA/CS24S001/God-dataset/validation"
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

true_labels = []
predicted_labels = []
with torch.no_grad():
    model.eval()
    for images, labels in valid_loader:
        outputs = model(images)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted_classes.numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
for i in range(conf_matrix.shape[0]):
    class_acc = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
    print(f"Class {i}: Accuracy {class_acc}")

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale if needed
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of transformers")

# Save plot as an image file
plt.savefig("confusion_matrix.png")

# Display the plot
plt.show()
