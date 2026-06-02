import torch
import os
import shutil

from torchvision import datasets, transforms, models
from torch import nn

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Validation dataset
val_data = datasets.ImageFolder(
    "Dataset/val",
    transform=transform
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,      # Important: process one image at a time
    shuffle=False
)

# Class names
class_names = val_data.classes
num_classes = len(class_names)
print("Classes:", class_names)

# Model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes # Dynamically use the length of your classes list
)

# Load your model weights
model.load_state_dict(
    torch.load(
        "newspaper_classifier_10.pt", # Note: Update this to "best_newspaper_classifier.pt" if you used the checkpointing script!
        map_location="cpu"
    )
)

model.eval()

# Create output folders
output_root = "Dataset/Predictions"

for cls in class_names:
    os.makedirs(
        os.path.join(output_root, cls),
        exist_ok=True
    )

correct = 0
total = 0

# Initialize an empty confusion matrix (N x N)
conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

with torch.no_grad():
    for idx, (image, label) in enumerate(val_loader):
        output = model(image)
        _, pred = torch.max(output, 1)

        total += 1

        if pred.item() == label.item():
            correct += 1
            
        # Update confusion matrix: row = actual label, column = predicted label
        conf_matrix[label.item(), pred.item()] += 1

        # Original image path
        image_path = val_data.samples[idx][0]
        predicted_class = class_names[pred.item()]

        destination = os.path.join(
            output_root,
            predicted_class,
            os.path.basename(image_path)
        )

        shutil.copy2(image_path, destination)

accuracy = 100 * correct / total

# --- PRINT FINAL METRICS ---

print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"Predictions saved in: {output_root}\n")

print("--- Confusion Matrix ---")
# Print a clean, formatted table for the matrix
header = f"{'Actual \\ Pred':<15} " + " ".join([f"{name:>10}" for name in class_names])
print(header)
print("-" * len(header))

for i, actual_class in enumerate(class_names):
    row_str = f"{actual_class:<15} "
    row_str += " ".join([f"{val.item():>10}" for val in conf_matrix[i]])
    print(row_str)

print("\n--- Per-Class Accuracy ---")
# The diagonal of the matrix contains correct predictions
class_correct = conf_matrix.diag()
# The sum of each row represents the total actual instances of that class
class_totals = conf_matrix.sum(dim=1)

for i in range(num_classes):
    if class_totals[i] > 0:
        class_acc = 100 * class_correct[i].item() / class_totals[i].item()
        print(f"{class_names[i]:<15}: {class_acc:.2f}% ({class_correct[i]}/{class_totals[i]})")
    else:
        print(f"{class_names[i]:<15}: No samples in validation set")