import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
