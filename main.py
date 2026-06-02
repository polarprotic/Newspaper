import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim

# Transforms — resize all crops to same size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
      transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

# Load your folders
train_data = datasets.ImageFolder("Dataset/train", transform=transform)
val_data   = datasets.ImageFolder("Dataset/val",   transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=32)

# Load pretrained EfficientNet — freeze everything except last layer
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)  # 3 classes

# Only train the final layer first
for param in model.features.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# Train for 3 epochs

epochs_count = 3

for epoch in range(epochs_count):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

torch.save(model.state_dict(), f"newspaper_classifier_{epochs_count}.pt")