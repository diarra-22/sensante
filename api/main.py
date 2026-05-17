# ============================================
# api/main.py - VERSION COMPLETE LAB 5
# ============================================

# ============================================
# PARTIE 1 : IMPORTS (EXISTANTS + NOUVEAUX)
# ============================================

# --- IMPORTS EXISTANTS (Lab 4) ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# --- NOUVEAUX IMPORTS (Étape 4.1) ---
import os
from dotenv import load_dotenv
from groq import Groq
# On ajoute aussi Field pour les descriptions (Étape 4.2)
from pydantic import Field


# ============================================
# PARTIE 2 : CHARGEMENT .env ET CLIENT GROQ (Étape 4.1)
# ============================================

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. "
          "/explain sera desactive.")


# ============================================
# PARTIE 3 : INITIALISATION FASTAPI (EXISTANT)
# ============================================

app = FastAPI(title="SénSanté API", description="API pour le diagnostic medical au Senegal")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PARTIE 4 : CHARGEMENT MODELE (EXISTANT)
# ============================================

model_path = Path(__file__).parent.parent / "models" / "model.pkl"
encoder_sexe_path = Path(__file__).parent.parent / "models" / "encoder_sexe.pkl"
encoder_region_path = Path(__file__).parent.parent / "models" / "encoder_region.pkl"

model = joblib.load(model_path)
le_sexe = joblib.load(encoder_sexe_path)
le_region = joblib.load(encoder_region_path)
print(f"Modele charge: {model.classes_}")



# ============================================
# PARTIE 5 : SCHEMAS EXISTANTS POUR /predict (EXISTANT)
# ============================================

class SymptomesInput(BaseModel):
    age: int
    sexe: str
    temperature: float
    tension_sys: int
    toux: int
    fatigue: int
    maux_tete: int
    region: str

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str = ""
    message: str = ""

# ============================================
# PARTIE 6 : NOUVEAUX SCHEMAS POUR /explain (Étape 4.2)
# ============================================

class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en français")
    modele_llm: str = Field(default="llama-3.1-8b-instant", description="Modele LLM utilise")


# ============================================
# PARTIE 7 : MAPPINGS (EXISTANT)
# ============================================

SEXE_MAP = {"M": 0, "F": 1}
REGION_MAP = {"Dakar": 0, "Thies": 1, "Ziguinchor": 2, "Saint-Louis": 3, "Touba": 4}


# ============================================
# PARTIE 8 : ENDPOINT /health (EXISTANT)
# ============================================

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


# ============================================
# PARTIE 9 : ENDPOINT /predict (EXISTANT)
# ============================================

@app.post("/predict", response_model=DiagnosticOutput)
def predict(symptoms: SymptomesInput):
    try:
        sexe_enc = le_sexe.transform([symptoms.sexe])[0]
        region_enc = le_region.transform([symptoms.region])[0]

        data = pd.DataFrame([{
            "age": symptoms.age,
            "sexe_encoded": sexe_enc,
            "temperature": symptoms.temperature,
            "tension_sys": symptoms.tension_sys,
            "toux": int(symptoms.toux),
            "fatigue": int(symptoms.fatigue),
            "maux_tete": int(symptoms.maux_tete),
            "region_encoded": region_enc
        }])

        proba = model.predict_proba(data)[0]
        pred_class = model.predict(data)[0]
        proba_max = max(proba)

        conf = "haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.5 else "faible"
        msg = "Suspicion de " + pred_class + ". Consultez un médecin rapidement."

        return DiagnosticOutput(
            diagnostic=pred_class,
            probabilite=round(proba_max, 2),
            confiance=conf,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        


# ============================================
# PARTIE 10 : SYSTEM PROMPT ET ENDPOINT /explain (Étape 4.3)
# ============================================

SYSTEM_PROMPT = """Tu es un assistant medical senegalais. Tu recois un diagnostic et des donnees patient. Explique le resultat en francais simple, comme un medecin parlerait a son patient. Sois rassurant mais recommande toujours une consultation medicale. Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-meme. Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. "
                        "Cle API non configuree.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    
    return ExplainOutput(explication=explication, modele_llm="llama-3.1-8b-instant")