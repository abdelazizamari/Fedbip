from datasets import load_dataset
import os
from PIL import Image
import numpy as np

output_dir = r"C:\Users\ESIP IT\Desktop\Mèmoir M2\code\FedBiP\data\PACS"
os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("flwrlabs/pacs", split="train")

# 🔥 Récupération automatique des noms des classes
label_names = dataset.features["label"].names
print("Labels:", label_names)

print("Total images:", len(dataset))

for i, item in enumerate(dataset):
    img_array = np.array(item["image"])
    domain = item["domain"]
    label = label_names[item["label"]]

    save_dir = os.path.join(output_dir, domain, label)
    os.makedirs(save_dir, exist_ok=True)

    img = Image.fromarray(img_array)
    img.save(os.path.join(save_dir, f"{i}.jpg"))

    if i % 500 == 0:
        print(f"Saved {i}/{len(dataset)} images...")

print("✅ PACS téléchargé et organisé sans erreur !")
