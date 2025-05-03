import json
import re
import pandas as pd
import json

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

