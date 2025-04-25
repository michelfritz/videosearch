import pandas as pd

df = pd.read_csv("blocs_transcription.csv")
df["video_id"] = "t21LM4CXaqE"  # identifiant de la vidéo YouTube
df["start"] = df["start"].astype(float).astype(int)  # au cas où
df.to_csv("blocs_transcription_normalise.csv", index=False)
print("✅ blocs_transcription_normalise.csv exporté.")
