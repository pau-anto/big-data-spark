import numpy as np
from PIL import Image
from pathlib import Path
from pyspark.sql import SparkSession
import io
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# =======
# CONFIG
# =======

BASE_DIR = Path(__file__).resolve().parent.parent
NPZ_INPUT_DIR = str(BASE_DIR / "output_parsed" )     # Où sont les .npz
BIN_OUTPUT_DIR = str(BASE_DIR / "output_bin")        # Où sauvegarder les .bin
 
Path(BIN_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================
# FONCTION DE CONVERSION
# ============================

def npz_to_bin(npz_info: tuple):

    npz_path, relative_path_str = npz_info

    try: 
        # Redéfinir les chemins (nécessaire pour les workers Spark)
        BIN_OUTPUT_DIR_LOCAL = Path(__file__).resolve().parent.parent / "output_bin"

        # Charger les données du fichier .npz
        data = np.load(npz_path)
        image_array = data['image_array']  

        # Convertir array en image
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')

        # Convertir en binaire PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_binary = buffer.getvalue()

        # Créer le chemin de sortie
        relative_path = Path(relative_path_str)
        output_file = BIN_OUTPUT_DIR_LOCAL / relative_path.with_suffix('.bin')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le binaire dans un fichier .bin
        with open(str(output_file), 'wb') as f:
            f.write(image_binary)

        return (True, len(image_binary), str(relative_path))  # Retourner la taille du binaire pour vérification
    
    except Exception as e:
        return (False, str(e), str(relative_path_str))  # Retourner l'erreur en cas d'échec
    

# =======
# MAIN
# =======

if __name__ == "__main__":
    print("="*70)
    print("CONVERSION NPZ -> BIN")
    print("="*70 + "\n")

    spark = SparkSession.builder \
        .appName("NPZ to BIN Conversion") \
        .master("local[*]") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .getOrCreate()
    
    print("Spark session créée avec succès.\n")

    # Trouver les fichiers npz
    npz_files = list(Path(NPZ_INPUT_DIR).glob("**/*.npz"))

    if not npz_files:
        print("Aucun fichier .npz trouvé dans le répertoire d'entrée.")
        spark.stop()
        exit(1)

    print(f"{len(npz_files)} fichier(s) .npz trouvé(s).")

    # Créer les tuples (npz_path, relative_path)
    npz_infos = [
        (str(npz_file), npz_file.relative_to(NPZ_INPUT_DIR))
        for npz_file in sorted(npz_files)
    ]
 
    # Créer RDD et paralléliser
    rdd = spark.sparkContext.parallelize(npz_infos, 4)
    
    # Appliquer la conversion sur chaque partition
    results = rdd.map(npz_to_bin).collect()

    # Convertir chaque fichier .npz en .bin
    success_count = 0
    error_count = 0
    total_size = 0
    errors_list = []

    print("Conversions:")
    for success, result, path in results:
        if success:
            size_bytes = result
            total_size += size_bytes
            success_count += 1
            if success_count % 100 == 0:
                print(f"  ✓ {success_count} fichiers traités...")
        else:
            error_count += 1
            errors_list.append((path, result))
        
     # Afficher les premières erreurs
    if errors_list:
        print("\nPremiers erreurs:")
        for path, error in errors_list[:5]:
            print(f"  ✗ {path}: {error}")

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Conversions réussies: {success_count}")
    print(f"Erreurs: {error_count}")
    print(f"Dossier source: {NPZ_INPUT_DIR}")
    print(f"Dossier destination: {BIN_OUTPUT_DIR}")
    print(f"Taille totale: {total_size / (1024*1024):.2f} MB")
    print("="*60 + "\n")

    if success_count > 0:
        print(f"Conversion terminée!")
        print(f"   {success_count} fichiers .bin créés dans {BIN_OUTPUT_DIR}")
    else:
        print("Aucune conversion réussie")

    spark.stop()
