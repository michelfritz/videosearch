import json
import csv

# Charger la transcription JSON de Whisper
with open("extrait_fin.json", "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

# Créer des blocs de 60 secondes
blocs = []
bloc = {"start": None, "text": ""}

for segment in segments:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]

    if bloc["start"] is None:
        bloc["start"] = start

    if end - bloc["start"] <= 60:
        bloc["text"] += " " + text
    else:
        blocs.append(bloc)
        bloc = {"start": start, "text": text}

# Ajouter le dernier bloc s'il contient du texte
if bloc["text"].strip():
    blocs.append(bloc)

# Enregistrer en CSV
with open("blocs_alur.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["start", "text"])
    writer.writeheader()
    writer.writerows(blocs)

print(f"✅ {len(blocs)} blocs de 60 secondes exportés dans blocs_alur.csv")
