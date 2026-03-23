"""
API Server - Interrogazione DuckDB in linguaggio naturale
Supporta Gemini API (primario) + Ollama (fallback)
"""

import json
import logging
import os
import re
import time
from typing import Optional

import duckdb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Configurazione ──────────────────────────────────────────────────────────

DB_PATH      = os.path.expanduser("~/rag-pipeline/documenti.duckdb")

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Ollama (fallback)
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen3:8b"

# Quale LLM usare: "gemini" o "ollama"
LLM_PROVIDER = "gemini" if GEMINI_API_KEY else "ollama"

# ── FastAPI ─────────────────────────────────────────────────────────────────

app = FastAPI(title="DocQuery API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema DB ───────────────────────────────────────────────────────────────

SCHEMA_DESCRIZIONE = """
Hai accesso a un database DuckDB con queste tabelle:

TABLE fatture (
    id              VARCHAR,
    file_path       VARCHAR,
    numero_fattura  VARCHAR,        -- es. "2025-001"
    data_fattura    DATE,           -- data emissione (YYYY-MM-DD)
    cliente         VARCHAR,
    totale          DOUBLE,         -- importo totale fattura in EUR
    voci            JSON,           -- array di {descrizione, importo, categoria}
    data_import     TIMESTAMP
)

TABLE documenti (
    id          VARCHAR,
    file_path   VARCHAR,
    file_name   VARCHAR,
    tipo        VARCHAR,            -- "fattura", "contratto", "altro"
    testo       TEXT,
    metadata    JSON,
    data_import TIMESTAMP
)

NOTE IMPORTANTI:
- Le date sono in formato DATE: YEAR(data_fattura) = 2025
- Trimestri: Q1=gen-mar, Q2=apr-giu, Q3=lug-set, Q4=ott-dic
- Quadrimestri: 1°=gen-apr, 2°=mag-ago, 3°=set-dic
- Ogni voce ha: descrizione (testo libero), importo (numero), categoria (parola breve)
- Categorie disponibili: wordpress, seo, marketing, grafica, blog, content, reporting, web, database, hosting
- Per sommare importi di un tipo di servizio usa UNNEST:
  SELECT SUM(CAST(json_extract(voce, '$.importo') AS DOUBLE))
  FROM fatture, UNNEST(CAST(voci AS JSON[])) AS t(voce)
  WHERE (json_extract_string(voce, '$.categoria') ILIKE '%wordpress%'
     OR CAST(voce AS VARCHAR) ILIKE '%wordpress%')
  AND YEAR(data_fattura) = 2025
- Per il totale fattura usa SUM(totale) senza UNNEST
- Per raggruppare per mese: strftime(data_fattura, '%Y-%m')
"""

# ── Prompt templates ────────────────────────────────────────────────────────

PROMPT_SQL = """{schema}

Rispondi SOLO con una query SQL DuckDB valida e completa, senza spiegazioni, senza markdown, senza backtick, senza puntini di sospensione.
La query deve essere eseguibile direttamente su DuckDB.

Domanda: {domanda}

SQL:"""

PROMPT_RISPOSTA = """Sei un assistente aziendale. Rispondi in italiano in modo chiaro e professionale.

Domanda originale: {domanda}

Risultato della query SQL:
{risultato}

Fornisci una risposta completa e leggibile basata sui dati.
Se il risultato è vuoto o NaN, dì che non hai trovato dati per quella categoria.
Non mostrare codice SQL nella risposta."""

PROMPT_RAG = """Sei un assistente aziendale. Rispondi in italiano in modo chiaro e professionale.
Usa solo le informazioni fornite nel contesto.

Domanda: {domanda}

Contesto (estratto dai documenti):
{contesto}

Rispondi in modo completo e utile."""

# ── Router keyword-based ────────────────────────────────────────────────────

SQL_KEYWORDS = {
    "fatturato", "fatturare", "fattura", "fatture", "totale", "totali",
    "quante", "quanti", "quanto", "somma", "importo", "importi",
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    "trimestre", "quadrimestre", "anno", "mese", "periodo",
    "cliente", "clienti", "elenco", "lista", "media", "conteggio",
    "wordpress", "seo", "hosting", "servizi", "ricavo", "ricavi",
    "pagato", "incassato", "emesso", "emessa", "numero",
    "riconciliazione", "estratto", "conto", "spesa", "spese",
    "marketing", "grafica", "newsletter", "campagna", "blog",
    "preventivo", "preventivi", "offerta", "offerte", "proposta", "contratto", "contratti"
}


def router(domanda: str) -> str:
    parole = set(domanda.lower().split())
    if parole & SQL_KEYWORDS:
        log.info("Router → sql")
        return "sql"
    log.info("Router → rag")
    return "rag"


# ── LLM: Gemini ─────────────────────────────────────────────────────────────

def chiama_gemini(prompt: str) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
        }
    }
    try:
        url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return result if result else None
    except Exception as e:
        log.error("Errore Gemini: %s", e)
        return None


# ── LLM: Ollama ─────────────────────────────────────────────────────────────

def chiama_ollama(prompt: str, max_tokens: int = 1024) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(10, 180))
        resp.raise_for_status()
        raw = resp.json()
        result = raw.get("response", "").strip()
        if not result:
            result = raw.get("thinking", "").strip()
        return result if result else None
    except Exception as e:
        log.error("Errore Ollama: %s", e)
        return None


# ── LLM unificato ───────────────────────────────────────────────────────────

def chiama_llm(prompt: str, max_tokens: int = 1024) -> Optional[str]:
    if LLM_PROVIDER == "gemini":
        result = chiama_gemini(prompt)
        if result:
            return result
        log.warning("Gemini fallito, uso Ollama come fallback")
    return chiama_ollama(prompt, max_tokens)


# ── SQL helpers ─────────────────────────────────────────────────────────────

def estrai_sql_da_testo(testo: str) -> Optional[str]:
    if not testo:
        return None
    testo = re.sub(r"```(?:sql)?", "", testo).strip().strip("`").strip()
    if re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)", testo, re.IGNORECASE):
        if ";" in testo:
            testo = testo[:testo.index(";") + 1]
        return testo.strip()
    matches = list(re.finditer(r"(SELECT\s.+?)(?:;|$)", testo, re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def genera_sql(domanda: str) -> Optional[str]:
    prompt = PROMPT_SQL.format(schema=SCHEMA_DESCRIZIONE, domanda=domanda)
    risposta = chiama_llm(prompt)
    if not risposta:
        return None
    log.info("SQL raw: %s", risposta[:300])
    return estrai_sql_da_testo(risposta)


def esegui_sql(sql: str) -> tuple[bool, str]:
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        df = conn.execute(sql).df()
        conn.close()
        if df.empty:
            return True, "Nessun risultato trovato."
        return True, df.to_string(index=False)
    except Exception as e:
        return False, str(e)


def cerca_rag(domanda: str, limit: int = 5) -> str:
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        parole = [p for p in domanda.lower().split()
                  if len(p) > 3 and p not in
                  {"cosa", "come", "quando", "dove", "quale", "quali",
                   "questo", "questa", "sono", "essere", "avere", "fare"}]
        if not parole:
            parole = domanda.lower().split()[:3]
        condizioni = " OR ".join(
            [f"LOWER(testo) LIKE '%{p}%'" for p in parole[:5]]
        )
        sql = f"""
            SELECT file_path, tipo, testo
            FROM documenti
            WHERE {condizioni}
            LIMIT {limit}
        """
        df = conn.execute(sql).df()
        conn.close()
        if df.empty:
            return "Nessun documento rilevante trovato."
        frammenti = []
        for _, row in df.iterrows():
            nome = row["file_path"].split("/")[-1]
            testo_breve = row["testo"][:500].replace("\n", " ")
            frammenti.append(f"[{nome}]: {testo_breve}...")
        return "\n\n".join(frammenti)
    except Exception as e:
        log.error("Errore RAG: %s", e)
        return "Errore nella ricerca documenti."


# ── Models ──────────────────────────────────────────────────────────────────

class DomandaRequest(BaseModel):
    domanda: str
    modalita: Optional[str] = "auto"


class RispostaResponse(BaseModel):
    risposta: str
    modalita_usata: str
    llm_provider: str
    sql_generato: Optional[str] = None
    risultato_raw: Optional[str] = None
    tempo_ms: int


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "servizio": "DocQuery API",
        "versione": "2.0",
        "llm_provider": LLM_PROVIDER,
        "gemini_configurato": bool(GEMINI_API_KEY)
    }


@app.get("/stats")
def stats():
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        n_doc  = conn.execute("SELECT COUNT(*) FROM documenti").fetchone()[0]
        n_fatt = conn.execute("SELECT COUNT(*) FROM fatture").fetchone()[0]
        totale = conn.execute(
            "SELECT SUM(totale) FROM fatture WHERE totale IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        return {
            "documenti_totali": n_doc,
            "fatture": n_fatt,
            "fatturato_totale_eur": round(totale, 2) if totale else 0,
            "llm_provider": LLM_PROVIDER
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chiedi", response_model=RispostaResponse)
def chiedi(req: DomandaRequest):
    t_start = time.time()
    log.info("Domanda: %s [provider=%s]", req.domanda, LLM_PROVIDER)

    modalita = req.modalita
    sql_generato = None
    risultato_raw = None

    if modalita == "auto":
        modalita = router(req.domanda)

    if modalita == "sql":
        sql_generato = genera_sql(req.domanda)
        if not sql_generato:
            risposta = "Non sono riuscito a generare una query per questa domanda."
        else:
            log.info("SQL generato: %s", sql_generato)
            successo, risultato_raw = esegui_sql(sql_generato)

            if not successo:
                log.warning("SQL fallito: %s | %s", sql_generato, risultato_raw)
                prompt_fix = f"""La seguente query SQL DuckDB ha prodotto un errore. Correggila.
Schema: {SCHEMA_DESCRIZIONE}
Query errata: {sql_generato}
Errore: {risultato_raw}
Rispondi SOLO con la query SQL corretta, senza markdown."""
                risposta_fix = chiama_llm(prompt_fix)
                sql_corretto = estrai_sql_da_testo(risposta_fix) if risposta_fix else None
                if sql_corretto:
                    successo2, risultato_raw = esegui_sql(sql_corretto)
                    if successo2:
                        sql_generato = sql_corretto
                    else:
                        risposta = f"Impossibile eseguire la query. Errore: {risultato_raw}"
                        return RispostaResponse(
                            risposta=risposta,
                            modalita_usata="sql",
                            llm_provider=LLM_PROVIDER,
                            sql_generato=sql_generato,
                            risultato_raw=risultato_raw,
                            tempo_ms=int((time.time() - t_start) * 1000)
                        )
                else:
                    risposta = f"Impossibile correggere la query. Errore: {risultato_raw}"
                    return RispostaResponse(
                        risposta=risposta,
                        modalita_usata="sql",
                        llm_provider=LLM_PROVIDER,
                        sql_generato=sql_generato,
                        risultato_raw=risultato_raw,
                        tempo_ms=int((time.time() - t_start) * 1000)
                    )

            prompt_risposta = PROMPT_RISPOSTA.format(
                domanda=req.domanda,
                risultato=risultato_raw
            )
            risposta = chiama_llm(prompt_risposta) or risultato_raw

    else:
        contesto = cerca_rag(req.domanda)
        prompt_rag = PROMPT_RAG.format(domanda=req.domanda, contesto=contesto)
        risposta = chiama_llm(prompt_rag) or "Non ho trovato informazioni rilevanti."

    tempo_ms = int((time.time() - t_start) * 1000)
    log.info("Risposta in %dms [%s]", tempo_ms, LLM_PROVIDER)

    return RispostaResponse(
        risposta=risposta or "Nessuna risposta generata.",
        modalita_usata=modalita,
        llm_provider=LLM_PROVIDER,
        sql_generato=sql_generato,
        risultato_raw=risultato_raw,
        tempo_ms=tempo_ms
    )


@app.post("/sql")
def esegui_sql_diretto(payload: dict):
    sql = payload.get("sql", "")
    if not sql:
        raise HTTPException(status_code=400, detail="Campo 'sql' mancante")
    successo, risultato = esegui_sql(sql)
    return {"successo": successo, "risultato": risultato}


if __name__ == "__main__":
    import uvicorn
    log.info("Avvio DocQuery API v2.0 [provider=%s]", LLM_PROVIDER)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
