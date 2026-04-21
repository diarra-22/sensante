import pandas as pd
import numpy as np
import os 

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")
from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques en nombres
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")   # doit afficher (500, 8)
print(f"Cible : {y.shape}")      # doit afficher (500,)
from sklearn.model_selection import train_test_split

# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42,     # Pour avoir les mêmes résultats à chaque fois
    stratify=y           # Garder les mêmes proportions de diagnostics
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,  # 100 arbres de décision
    random_state=42    # Reproductibilité
)

# Entraîner sur les données d'entraînement
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")
# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions avec la réalité
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
# Visualiser avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=model.classes_,
yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion- SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")
import joblib

# Créer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sérialiser le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")
# Sauvegarder les encodeurs (indispensables pour les nouvelles données)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")
# Charger le modèle DEPUIS LE FICHIER (pas depuis la mémoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")
# Un nouveau patient arrive au centre de santé
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:12s} : {proba:.1%} {bar}")
    