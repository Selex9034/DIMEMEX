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

