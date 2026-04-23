# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Schémas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# --- Chargement du modèle (une seule fois au démarrage) ---
print("Chargement du modèle...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {list(model.classes_)}")

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # 1. Encoder les variables catégoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F.")
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Région inconnue : {patient.region}")

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prédire
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # 4. Niveau de confiance
    confiance = ("haute" if proba_max >= 0.7
                 else "moyenne" if proba_max >= 0.4
                 else "faible")

    # 5. Message
    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )
    