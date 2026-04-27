import json
import os
from sklearn.model_selection import train_test_split

def convert_kvasir_to_coco_split(input_json_path, output_dir):
    if not os.path.exists(input_json_path):
        print(f"Erreur : {input_json_path} introuvable.")
        return

    with open(input_json_path, 'r') as f:
        data = json.load(f)

    image_keys = list(data.keys())

    # 1. On sépare le Train (800) du reste (200)
    # 800 sur 1000 = 0.8
    train_keys, rest_keys = train_test_split(
        image_keys, train_size=0.8, random_state=42, shuffle=True
    )

    # 2. On sépare le reste en deux parts égales (100 Val / 100 Test)
    val_keys, test_keys = train_test_split(
        rest_keys, train_size=0.5, random_state=42, shuffle=True
    )

    def build_coco_dict(keys_subset):
        # ... (le reste de ta fonction build_coco_dict est identique) ...
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "polyp"}]
        }
        ann_id = 1
        for i, img_name in enumerate(keys_subset, start=1):
            img_info = data[img_name]
            coco_dict["images"].append({
                "id": i,
                "file_name": f"{img_name}.jpg",
                "width": img_info["width"],
                "height": img_info["height"]
            })
            for bbox in img_info["bbox"]:
                w = bbox["xmax"] - bbox["xmin"]
                h = bbox["ymax"] - bbox["ymin"]
                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [float(bbox["xmin"]), float(bbox["ymin"]), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0
                })
                ann_id += 1
        return coco_dict

    # 3. Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    for filename, keys in [("train.json", train_keys), ("val.json", val_keys), ("test.json", test_keys)]:
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(build_coco_dict(keys), f, indent=4)
        print(f"Créé : {filename} ({len(keys)} images)")

if __name__ == "__main__":
    # Trouve le dossier parent (la racine) par rapport à l'emplacement de ce script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    input_path = os.path.join(PROJECT_ROOT, 'data/Kvasir-SEG/kavsir_bboxes.json')
    output_path = os.path.join(PROJECT_ROOT, 'data/Kvasir-SEG')

    convert_kvasir_to_coco_split(input_path, output_path)