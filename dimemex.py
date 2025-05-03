import json
import re
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
import os



#Ordenamos los datos en el train_data.json para facilitar su relación con las etiquetas
with open("train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def extraer_numero(meme_id):
    match = re.search(r'(\d+)', meme_id)
    return int(match.group(1)) if match else 0

datos_ordenados = sorted(data, key=lambda x: extraer_numero(x["MEME-ID"]))

with open("train_data_sorted.json", "w", encoding="utf-8") as f:
    json.dump(datos_ordenados, f, indent=4, ensure_ascii=False)


with open("train_data_sorted.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df_json = pd.DataFrame(data)
df_json = df_json[["MEME-ID"]]

df_labels = pd.read_csv("train_labels_tasks_1_3.csv")

df_combined = pd.concat([df_json, df_labels], axis=1)

df_combined.to_csv("data.csv", index=False, header=False)

#Preprocesar las imágenes

def preprocess_image(img_path, size=(224, 224)):
    #convertir a rgb
    img = Image.open(img_path).convert('RGB')

    #redimensionar imagenes
    img = img.resize(size, Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0  # Normalización [0,1]
    return img

df = pd.read_csv("data.csv", header=None)

#asignar nombres a las columnas
df.columns = ["MEME-ID", "ninguno", "contenido_inapropiado", "discurso_odio"]

print(df.columns)

#mapear cada imagen a su etiqueta

#función para extraer etiqueta
def obtener_etiqueta(row):
    if row["discurso_odio"] == 1:
        return "discurso_odio"
    elif row["contenido_inapropiado"] == 1:
        return "contenido_inapropiado"
    elif row["ninguno"] == 1:
        return "ninguno"
    else:
        return "desconocido"

#aplicar función
df["etiqueta"] = df.apply(obtener_etiqueta, axis=1)



#solo nos quedamos con MEME-ID y etiqueta
df = df[["MEME-ID", "etiqueta"]]

#crear diccionario {MEME-ID: etiqueta}
id_to_label = dict(zip(df["MEME-ID"], df["etiqueta"]))


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar: {image_path}")
        return np.zeros((224, 224, 3), dtype=np.uint8)  #imagen vacía para no romper el flujo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #OpenCV carga en BGR, convertimos a RGB
    img = cv2.resize(img, (224, 224))           #redimensionar
    img = img / 255.0                            #normalizar
    return img

def load_dataset(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = preprocess_image(img_path)
            images.append(img)

            label = id_to_label.get(filename, None)
            if label is not None:
                labels.append(label)
            else:
                print(f"Advertencia: {filename} no tiene etiqueta.")

    return np.array(images), np.array(labels)

X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("validation")

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
