import json
import os

base_dir = "data/MSD_pancreas"

# Chargement du fichier maître généré juste après l'extraction
master_json_path = os.path.join(base_dir, "annotations.json")

if not os.path.exists(master_json_path):
    print(f"Erreur : Le fichier {master_json_path} est introuvable.")
else:
    with open(master_json_path, 'r') as f:
        data = json.load(f)

    # On récupère les catégories (ID 1: pancreas, ID 2: tumor)
    categories = data["categories"]

    for split_name in ["train", "val", "test"]:
        # 1. Filtrer les images appartenant au split
        split_images = [img for img in data["images"] if img.get("split") == split_name]
        split_img_ids = {img["id"] for img in split_images}
        
        # 2. Filtrer les annotations pour garder les DEUX classes
        split_annotations = []
        for ann in data["annotations"]:
            # CHANGEMENT ICI : On accepte les ID 1 et 2
            if ann["image_id"] in split_img_ids and ann["category_id"] in [1, 2]:
                new_ann = ann.copy()
                new_ann["iscrowd"] = ann.get("iscrowd", 0)
                split_annotations.append(new_ann)
        
        # 3. Création du dictionnaire pour le split
        split_data = {
            "categories": categories,
            "images": split_images,
            "annotations": split_annotations
        }
        
        # 4. Sauvegarde
        output_path = os.path.join(base_dir, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=4)
            
        print(f"Fichier créé : {output_path}")
        print(f"   -> {len(split_images)} images")
        print(f"   -> {len(split_annotations)} annotations totales (pancréas + tumeurs)")