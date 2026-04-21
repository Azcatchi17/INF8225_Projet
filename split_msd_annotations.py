import json
import os

base_dir = "data/MSD_pancreas"

# On charge le nouveau fichier maître de la V4
with open(os.path.join(base_dir, "annotations.json"), 'r') as f:
    data = json.load(f)

# MODIFICATION 1 : On conserve l'intégralité des catégories (y compris le gaz sans boîte)
categories = data["categories"]

for split_name in ["train", "val", "test"]:
    # 1. Filtrer les images par split
    split_images = [img for img in data["images"] if img.get("split") == split_name]
    split_img_ids = {img["id"] for img in split_images}
    
    # 2. Filtrer et enrichir les annotations
    split_annotations = []
    for ann in data["annotations"]:
        # MODIFICATION 2 : On accepte le pancréas (1) ET la tumeur (2)
        if ann["image_id"] in split_img_ids and ann["category_id"] in [1, 2]:
            new_ann = ann.copy()
            new_ann["iscrowd"] = ann.get("iscrowd", 0)
            split_annotations.append(new_ann)
    
    # 3. Créer le dictionnaire final
    split_data = {
        "categories": categories,
        "images": split_images,
        "annotations": split_annotations
    }
    
    # 4. Sauvegarder avec le suffixe V4
    output_path = os.path.join(base_dir, f"{split_name}.json")
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Fichier créé : {output_path} ({len(split_images)} images, {len(split_annotations)} boîtes)")