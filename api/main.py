# ============================================
# api/main.py - VERSION COMPLETE LAB 6
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import os
import httpx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key, http_client=httpx.Client())
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

app = FastAPI(title="SenSante API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path(__file__).parent.parent / "models" / "model.pkl"
encoder_sexe_path = Path(__file__).parent.parent / "models" / "encoder_sexe.pkl"
encoder_region_path = Path(__file__).parent.parent / "models" / "encoder_region.pkl"

model = joblib.load(model_path)
le_sexe = joblib.load(encoder_sexe_path)
le_region = joblib.load(encoder_region_path)
print(f"Modele charge: {model.classes_}")

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

class ExplainInput(BaseModel):
    diagnostic: str = Field(...)
    probabilite: float = Field(...)
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(...)
    modele_llm: str = Field(default="llama-3.1-8b-instant")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

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
        msg = "Suspicion de " + pred_class + ". Consultez un medecin rapidement."
        return DiagnosticOutput(
            diagnostic=pred_class,
            probabilite=round(proba_max, 2),
            confiance=conf,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

SYSTEM_PROMPT = """Tu es un assistant medical senegalais qui parle en francais.
Explique le resultat simplement, comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-meme."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
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

# ============================================
# FRONTEND STATIQUE
# ============================================

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")