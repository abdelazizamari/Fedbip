from datasets import load_dataset
import os
from PIL import Image
import numpy as np

output_dir = r"C:\Users\ESIP IT\Desktop\Mèmoir M2\code\FedBiP\data\UC_Merced"
os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("blanchon/UC_Merced", split="train")

# 🔥 Noms des classes automatiquement
label_names = dataset.features["label"].names
print("Labels:", label_names)

print("Total images:", len(dataset))

for i, item in enumerate(dataset):
    img_array = np.array(item["image"])
    label = label_names[item["label"]]

    save_dir = os.path.join(output_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    img = Image.fromarray(img_array)
    img.save(os.path.join(save_dir, f"{i}.jpg"))

    if i % 100 == 0:
        print(f"Saved {i}/{len(dataset)} images...")

print("✅ UC Merced téléchargé et organisé correctement !")
