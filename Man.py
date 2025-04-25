import pandas as pd

df1 = pd.read_csv("blocs_transcription_normalise.csv")
df2 = pd.read_csv("blocs_alur_normalise.csv")
fusion = pd.concat([df1, df2], ignore_index=True)
fusion.to_csv("blocs_fusionnes.csv", index=False)
print("✅ blocs_fusionnes.csv créé avec", len(fusion), "blocs.")
