import os
import numpy as np
import cv2

def make_images(path, label):
    os.makedirs(path, exist_ok=True)
    for i in range(40):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.putText(img, label, (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(path, f"{label}_{i}.jpg"), img)

make_images("data/processed/real", "REAL")
make_images("data/processed/fake", "FAKE")

print("✅ Dummy dataset created successfully")
