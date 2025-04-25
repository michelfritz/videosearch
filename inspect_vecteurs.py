import pickle
import numpy as np

with open("vecteurs.pkl", "rb") as f:
    vecteurs = pickle.load(f)

print("Type du contenu :", type(vecteurs))

if isinstance(vecteurs, np.ndarray):
    print("✅ C'est un numpy array.")
    print("Shape :", vecteurs.shape)
else:
    print("❌ Ce n'est pas un numpy array.")
    try:
        print("Premier élément :", vecteurs[0])
        print("Type du premier élément :", type(vecteurs[0]))
    except Exception as e:
        print("Erreur lors de l'affichage :", e)
