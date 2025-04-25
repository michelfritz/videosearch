import whisper
import time

# Utiliser un modèle plus rapide si nécessaire
model = whisper.load_model("base")  # ou "small", ou "medium" si t’es patient

print("✅ Modèle chargé.")

start = time.time()
print("⏳ Transcription en cours...")

# Transcription avec options
result = model.transcribe("Alur.mp4", verbose=True, language="fr")

# Sauvegarde brute
with open("transcription_alur.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"✅ Transcription terminée en {round(time.time() - start, 2)} secondes.")
