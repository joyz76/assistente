"""
Retry estrazione per fatture senza totale
"""
import json
import logging
from ingestion import (
    get_conn, estrai_testo, estrai_dati_fattura_llm,
    chiama_llm, pulisci_json, query, PROMPT_FATTURA, MAX_CHARS_LLM
)
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Aumenta num_predict per questo retry
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

def chiama_llm_lungo(prompt: str):
    """Come chiama_llm ma con num_predict più alto."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 2048,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.error("Errore LLM: %s", e)
        return None


def retry_fatture_senza_totale():
    # Trova i file da riprocessare
    df = query("""
        SELECT d.id, d.file_path, d.hash_file
        FROM documenti d
        LEFT JOIN fatture f ON d.id = f.id
        WHERE f.id IS NULL AND d.tipo = 'fattura'
        ORDER BY d.file_path
    """)

    if len(df) == 0:
        print("Nessuna fattura da riprocessare!")
        return

    print(f"Riprocesso {len(df)} fatture...")
    ok = 0
    errori = 0

    for _, row in df.iterrows():
        file_path = row["file_path"]
        doc_id    = row["id"]
        hash_val  = row["hash_file"]
        file_name = file_path.split("/")[-1]

        log.info("Retry: %s", file_name)

        testo = estrai_testo(file_path)
        if not testo.strip():
            log.warning("  Testo vuoto: %s", file_name)
            errori += 1
            continue

        prompt = PROMPT_FATTURA.format(testo=testo[:MAX_CHARS_LLM])
        risposta = chiama_llm_lungo(prompt)
        dati = pulisci_json(risposta)

        if not dati or dati.get("totale") is None:
            log.warning("  Ancora JSON invalido: %s | risposta: %s",
                        file_name, str(risposta)[:200] if risposta else "None")
            errori += 1
            continue

        try:
            totale = float(str(dati["totale"]).replace(",", "."))
        except (ValueError, TypeError):
            log.warning("  Totale non numerico: %s", dati.get("totale"))
            errori += 1
            continue

        conn = get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fatture
                  (id, file_path, numero_fattura, data_fattura,
                   cliente, totale, voci, hash_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                doc_id, file_path,
                dati.get("numero_fattura"),
                dati.get("data_fattura"),
                dati.get("cliente") or dati.get("fornitore"),
                totale,
                json.dumps(dati.get("voci", [])),
                hash_val
            ])
            conn.commit()
            log.info("  ✓ %s → %.2f EUR", file_name, totale)
            ok += 1
        except Exception as e:
            log.error("  Errore DB: %s", e)
            errori += 1
        finally:
            conn.close()

    print(f"\nRetry completato: {ok} ok, {errori} errori")

    # Riepilogo finale
    conn = get_conn()
    n_fatt = conn.execute("SELECT COUNT(*) FROM fatture").fetchone()[0]
    totale_db = conn.execute(
        "SELECT SUM(totale) FROM fatture WHERE totale IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    print(f"Fatture totali in DB: {n_fatt}")
    print(f"Fatturato totale:     {totale_db:,.2f} EUR" if totale_db else "N/D")


if __name__ == "__main__":
    retry_fatture_senza_totale()
