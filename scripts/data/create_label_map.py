import json
import os

os.makedirs('data/Kvasir-SEG/', exist_ok=True)
with open('data/Kvasir-SEG/polyp_label_map.json', 'w') as f:
    json.dump({'0': 'polyp'}, f)
    
print("Fichier label_map créé avec succès !")