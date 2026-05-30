import numpy as np
from PIL import Image
from pathlib import Path
import io

BASE_DIR = Path(__file__).resolve().parent.parent
NPZ_INPUT_DIR = BASE_DIR / "output_parsed"      # Où sont les .npz
BIN_OUTPUT_DIR = BASE_DIR / "output_bin"        # Où sauvegarder les .bin
 
BIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def npz_to_bin(npz_path: str, output_path: str):
    try: 
        # Charger les données du fichier .npz
        data = np.load(npz_path)
        image_array = data['image_array']  

        # Convertir array en image
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')

        # Convertir en binaire PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_binary = buffer.getvalue()

        # Sauvegarder le binaire dans un fichier .bin
        with open(output_path, 'wb') as f:
            f.write(image_binary)

        return True, len(image_binary)  # Retourner la taille du binaire pour vérification
    
    except Exception as e:
        return False, str(e)  # Retourner l'erreur en cas d'échec
    

# =======
# MAIN
# =======

if __name__ == "__main__":
    print("="*60)
    print("CONVERSION NPZ -> BIN")
    print("="*60 + "\n")

    # Trouver les fichiers npz
    npz_files = list(NPZ_INPUT_DIR.glob("**/*.npz"))

    if not npz_files:
        print("Aucun fichier .npz trouvé dans le répertoire d'entrée.")
        exit(1)

    print(f"{len(npz_files)} fichier(s) .npz trouvé(s).")

    # Convertir chaque fichier .npz en .bin
    success_count = 0
    error_count = 0
    total_size = 0

    for npz_file in sorted(npz_files):
        # bCréer le chemin de sortie .bin (struct Train/Test/label)
        relative_path = npz_file.relative_to(NPZ_INPUT_DIR)
        output_file = BIN_OUTPUT_DIR / relative_path.with_suffix('.bin')

        # Créer le répertoire si il n'existe pas
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir et sauvegarder
        success, result = npz_to_bin(str(npz_file), str(output_file))

        if success:
            size_bytes = result
            total_size += size_bytes
            success_count += 1
            print(f"{relative_path.parent.name}/{npz_file.stem}.bin ({size_bytes} bytes)")

        else: 
            error_count += 1
            print(f"Erreur pour {relative_path}: {result}")

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"✓ Conversions réussies: {success_count}")
    print(f"✗ Erreurs: {error_count}")
    print(f"📂 Dossier source: {NPZ_INPUT_DIR}")
    print(f"📂 Dossier destination: {BIN_OUTPUT_DIR}")
    print(f"💾 Taille totale: {total_size / (1024*1024):.2f} MB")
    print("="*60 + "\n")

    if success_count > 0:
        print(f"✅ Conversion terminée!")
        print(f"   {success_count} fichiers .bin créés dans {BIN_OUTPUT_DIR}")
    else:
        print("❌ Aucune conversion réussie")