import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.transforms as transforms

# Same model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 56 * 56, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Dummy dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FakeData(
    size=200,
    image_size=(3, 224, 224),
    num_classes=2,
    transform=transform
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for 2 epochs (FAST)
for epoch in range(2):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training done")

# Save model
torch.save(model.state_dict(), "models/deepfake_cnn.pth")
print("✅ Model saved to models/deepfake_cnn.pth")
