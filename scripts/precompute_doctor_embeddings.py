"""Precompute and store embeddings for doctor profiles.

Usage:
    python scripts/precompute_doctor_embeddings.py --force
    python scripts/precompute_doctor_embeddings.py --sample "fever and cough"

This script reuses the SentenceTransformer model (same as app.py) to compute embeddings
and writes them into the `users` collection under the `embedding` field as lists.
It also exposes a small sample query runner that shows ranked doctors for a given
symptom text (useful for tuning thresholds and boosts).

Note: Run this in the same environment as the app so the sentence-transformers
model and MongoDB connection are available.
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `tools` is importable when running scripts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.specialization_utils import normalize_profile

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["caremate"]
users_col = db["users"]

# Use same model as app
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedder():
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print("Model loaded")
    return model


def build_doc_text(doc):
    profile = doc.get("doctorProfile", {})
    parts = []
    if profile.get("specialization"):
        parts.append(profile.get("specialization"))
    if profile.get("qualifications"):
        parts.append(" ".join(profile.get("qualifications")))
    if profile.get("yearsExperience"):
        parts.append(f"{profile.get('yearsExperience')} years experience")
    if profile.get("bio"):
        parts.append(profile.get("bio"))
    if profile.get("clinic"):
        parts.append(profile.get("clinic"))
    if profile.get("languages"):
        parts.append(" ".join(profile.get("languages")))
    if not parts:
        parts.append(doc.get("name", ""))
    return " ".join(parts).strip()


def precompute_all(force=False):
    embedder = get_embedder()
    doctors = list(users_col.find({"role": "doctor"}))
    print(f"Found {len(doctors)} doctors")

    count = 0
    for doc in doctors:
        if doc.get("embedding") and not force:
            continue
        # Normalize specialization in DB so embeddings include canonical specialization
        profile = doc.get('doctorProfile', {}) or {}
        normalized_profile = normalize_profile(profile)
        if normalized_profile.get('specialization') and normalized_profile.get('specialization') != profile.get('specialization'):
            users_col.update_one({"_id": doc["_id"]}, {"$set": {"doctorProfile.specialization": normalized_profile.get('specialization')}})

        text = build_doc_text(dict(doc, doctorProfile=normalized_profile))
        try:
            emb = embedder.encode(text, normalize_embeddings=True)
            users_col.update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb.tolist()}})
            count += 1
        except Exception as e:
            print(f"Failed to compute embedding for {doc.get('name')}: {e}")
    print(f"Updated embeddings for {count} doctors")


# Small query runner using cosine similarity
def rank_doctors_for_query(query, top_k=5):
    # Default scoring mirrors app.py logic (cosine + keyword boosts)
    embedder = get_embedder()
    q_emb = embedder.encode(query, normalize_embeddings=True)
    docs = list(users_col.find({"role": "doctor"}))

    # Use same SPECIALIZATION_KEYWORDS and boost logic as app.py for parity
    SPECIALIZATION_KEYWORDS = {
        "cardiologist": ["chest pain", "shortness of breath", "palpitat", "heart attack", "angina", "tachycardia"],
        "pulmonologist": ["cough", "shortness of breath", "wheeze", "wheezing", "bronchitis", "asthma"],
        "dermatologist": ["rash", "itch", "itching", "redness", "eczema", "psoriasis", "skin"],
        "ent": ["ear", "hearing", "hearing loss", "ear pain", "tinnitus", "hoarseness"],
        "neurology": ["headache", "seizure", "numbness", "weakness", "dizziness", "migraine"],
        "pediatrician": ["child", "kid", "baby", "fever in child", "pediatric"],
        "orthopedist": ["joint pain", "back pain", "fracture", "sprain", "arthritis"],
    }

    q_text_lc = (query or "").lower()
    implied_specs = set()
    for spec, keys in SPECIALIZATION_KEYWORDS.items():
        for k in keys:
            if k in q_text_lc:
                implied_specs.add(spec)
                break

    def score_doc(doc, q_emb, q_text, spec_boost=0.12, qual_boost=0.05):
        profile = doc.get("doctorProfile", {})
        specialization = profile.get("specialization", "")
        qualifications = profile.get("qualifications", [])

        emb = doc.get("embedding")
        if not emb:
            return None
        emb_np = np.array(emb, dtype=float)

        qn = np.linalg.norm(q_emb)
        dn = np.linalg.norm(emb_np)
        if qn == 0 or dn == 0:
            return None

        sim = float(np.dot(q_emb, emb_np) / (qn * dn))

        # keyword boosts mirroring app.py
        boost = 0.0
        spec_lc = (specialization or "").lower()
        for implied in implied_specs:
            if implied in spec_lc or spec_lc in implied:
                boost += 0.35
        if specialization and specialization.lower() in q_text.lower():
            boost += 0.12
        for qual in qualifications:
            if qual and qual.lower() in q_text.lower():
                boost += 0.05

        return sim + boost

    scores = []
    for doc in docs:
        s = score_doc(doc, q_emb, query)
        if s is None:
            continue
        scores.append((doc, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    for doc, score in scores[:top_k]:
        print(f"{doc.get('name', '')} - {doc.get('doctorProfile', {}).get('specialization','')} - {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force recompute embeddings for all doctors")
    parser.add_argument("--sample", type=str, help="Run a sample query and print rankings")
    parser.add_argument("--top", type=int, default=5, help="Top K to show for sample query")
    args = parser.parse_args()

    if args.force:
        precompute_all(force=True)

    if args.sample:
        rank_doctors_for_query(args.sample, top_k=args.top)

    if not args.force and not args.sample:
        print("Nothing to do. Use --force to recompute embeddings or --sample 'text' to run a query.")
