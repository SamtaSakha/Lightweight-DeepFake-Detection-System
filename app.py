import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms

# ---------------- MODEL ---------------- #
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
        self.fc = nn.Linear(32 * 56 * 56, 2)  # 224x224 input → 56x56 feature map after conv+pool

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------- LOAD MODEL ---------------- #
model = SimpleCNN()
MODEL_PATH = "models/deepfake_cnn.pth"
model_loaded = False

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model_loaded = True
    print("✅ Model loaded successfully")
else:
    print("❌ Model file not found")

# ---------------- TRANSFORMS ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5]*3,
        std=[0.5]*3
    )
])

# ---------------- PREDICTION ---------------- #
def predict(image):
    """
    Predict if an image frame is REAL or FAKE.
    """
    if image is None:
        return "No image provided ❌"

    # Convert to RGB and apply transforms
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0)

    if not model_loaded:
        return "Model not loaded ❌"

    with torch.no_grad():
        out = model(image)
        prob = F.softmax(out, dim=1)[0]

        real_prob = prob[0].item()
        fake_prob = prob[1].item()

        # Debug output
        print(f"DEBUG → REAL: {real_prob:.3f}, FAKE: {fake_prob:.3f}")

        # Temporary threshold for testing
        if fake_prob > 0.5:
            return f"FAKE ❌ ({fake_prob:.2f})"
        else:
            return f"REAL ✅ ({1-fake_prob:.2f})"



# ---------------- UI ---------------- #
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Lightweight DeepFake Detection System",
    description="Upload an image frame to detect whether it is REAL or FAKE."
)

# ---------------- LAUNCH ---------------- #
if __name__ == "__main__":
    demo.launch()
