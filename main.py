
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler

# 1. Transforms — resize all crops to same size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],   # ImageNet mean
        [0.229, 0.224, 0.225]    # ImageNet std
    )
])

# Load your folders
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(
    "Dataset/train",
    transform=transform
)

val_data = datasets.ImageFolder(
    "Dataset/val",
    transform=val_transform
)

# 2. Implement WeightedRandomSampler for Class Imbalance
# Assuming your class counts are exactly [4402, 903, 334] based on your weight_tensor
class_counts = [4402, 903, 334] 
total_samples = sum(class_counts)
class_weights = [total_samples / c for c in class_counts]

# Assign a weight to every single image in the dataset based on its class
sample_weights = [class_weights[label] for _, label in train_data.samples]
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

# Replace shuffle=True with the sampler
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    sampler=sampler # Do not use shuffle=True when using a sampler
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=32
)

# 3. Load pretrained EfficientNet and setup layers
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    3
)  # 3 classes

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the new classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

# Unfreeze the very last block of EfficientNet to adapt to your dataset
for param in model.features[-2:].parameters():
    param.requires_grad = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

# We use standard CrossEntropyLoss because the Sampler handles imbalance now
criterion = nn.CrossEntropyLoss()

# 4. Use AdamW with Differential Learning Rates and Weight Decay
optimizer = optim.AdamW([
    {'params': model.features[-2:].parameters(), 'lr': 1e-4}, # Lower LR for feature extractor
    {'params': model.classifier.parameters(), 'lr': 1e-3}     # Standard LR for new head
], weight_decay=1e-2)

epochs_count = 3

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs_count
)

best_acc = 0

# 5. Training Loop
for epoch in range(epochs_count):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    print(f"Epoch {epoch + 1} | Val Acc: {acc:.4f}")

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            f"newspaper_classifier_{epochs_count}_best_again.pt"
        )