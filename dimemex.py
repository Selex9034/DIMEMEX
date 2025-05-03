import json
import re
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf



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


import json

with open("train_data_sorted.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

id_to_text = {}

for item in train_data:
    meme_id = item["MEME-ID"]
    text = item["text"]
    id_to_text[meme_id] = text
    
texts_train = []
for i in range(len(y_train)):
    meme_id = df.iloc[i]["MEME-ID"]
    text = id_to_text.get(meme_id, "")
    texts_train.append(text)

texts_train = np.array(texts_train)

texts_val = []
for i in range(len(y_val)):
    meme_id = df.iloc[i]["MEME-ID"]
    text = id_to_text.get(meme_id, "")
    texts_val.append(text)

texts_val = np.array(texts_val)

from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Lambda

#tokenizador BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#parámetros
MAX_LEN = 64  #máxima longitud del texto tokenizado

#imagen
image_input = Input(shape=(224, 224, 3), name="image_input")

#texto (lo vamos a tokenizar antes de alimentar a BERT)
text_input = Input(shape=(MAX_LEN,), dtype=tf.int32, name="text_input")
attention_mask_input = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask_input")

vgg = VGG16(include_top=False, weights="imagenet", input_tensor=image_input)
for layer in vgg.layers:
    layer.trainable = False  #congelamos VGG16

x_image = layers.Flatten()(vgg.output)
x_image = layers.Dense(256, activation="relu")(x_image)

bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def bert_layer(inputs):
    input_ids, attention_mask = inputs
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.pooler_output  # <- (batch_size, 768)

x_text = Lambda(bert_layer, output_shape=(768,))([text_input, attention_mask_input])
x_text = layers.Dense(256, activation="relu")(x_text)

combined = layers.concatenate([x_image, x_text])
z = layers.Dense(128, activation="relu")(combined)
z = layers.Dropout(0.3)(z)
output = layers.Dense(3, activation="softmax")(z)  #3 clases

model = Model(inputs=[image_input, text_input, attention_mask_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
import numpy as np

X_train = np.array(X_train, dtype=np.float32)
X_val   = np.array(X_val,   dtype=np.float32)

label_map = {
    "discurso_odio":            0,
    "contenido_inapropiado":    1,
    "ninguno":                  2
}

X_train_filt, texts_train_filt, y_train_filt = [], [], []
for img, txt, lbl in zip(X_train, texts_train, y_train):
    if lbl in label_map:                   # solo guardamos si lbl está en el mapa
        X_train_filt.append(img)
        texts_train_filt.append(txt)
        y_train_filt.append(lbl)

X_train = np.array(X_train_filt, dtype=np.float32)
texts_train = texts_train_filt        # seguimos pasándolo a tokenizar más adelante
y_train = np.array(y_train_filt)      # strings, por ahora

X_val_filt, texts_val_filt, y_val_filt = [], [], []
for img, txt, lbl in zip(X_val, texts_val, y_val):
    if lbl in label_map:
        X_val_filt.append(img)
        texts_val_filt.append(txt)
        y_val_filt.append(lbl)

X_val   = np.array(X_val_filt, dtype=np.float32)
texts_val = texts_val_filt
y_val   = np.array(y_val_filt)


#ahora sí todo el contenido de y_train/y_val está en label_map
y_train = np.array([ label_map[l] for l in y_train ], dtype=np.int32)
y_val   = np.array([ label_map[l] for l in y_val   ], dtype=np.int32)

# Verifica que solo tengas 0,1,2
print("Clases en train:", np.unique(y_train))
print("Clases en val:  ", np.unique(y_val))


