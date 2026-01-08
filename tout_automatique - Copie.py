
# tout_automatique.py

import os
import subprocess

# Étape 1 : Transcription et découpage
print("🔁 Étape 1 : Transcription + Découpage en blocs")
transcrire_script = "transcrire_et_decouper.py"
ret1 = subprocess.run(["python", transcrire_script])

if ret1.returncode != 0:
    print(f"❌ Échec de {transcrire_script}")
    exit(1)
else:
    print(f"✅ {transcrire_script} terminé avec succès.")

# Étape 2 : Fusion + Vectorisation
print("\n🧠 Étape 2 : Fusion des blocs + vectorisation")
fusionner_script = "fusionner_et_vectoriser.py"
ret2 = subprocess.run(["python", fusionner_script])

if ret2.returncode != 0:
    print(f"❌ Échec de {fusionner_script}")
    exit(1)
else:
    print(f"✅ {fusionner_script} terminé avec succès.")

print("\n🏁 Pipeline complet terminé.")
