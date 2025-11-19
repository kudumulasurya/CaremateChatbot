from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import secrets
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

from flask_cors import CORS

from core.langgraph_workflow import create_workflow
from core.state import initialize_conversation_state, reset_query_state
from tools.pdf_loader import process_pdf
from tools.vector_store import get_or_create_vectorstore

from sentence_transformers import SentenceTransformer
import numpy as np
from tools.specialization_utils import normalize_profile
from tools.lang_utils import detect_language

# --------------------------------------
# Load environment variables
# --------------------------------------
load_dotenv()

# --------------------------------------
# Flask app setup
# --------------------------------------
app = Flask(__name__)


# Allow all origins (development). For production, restrict this to a safe list.
FRONTEND_URL = "https://sanketyelugotla-caremate.vercel.app"

CORS(
    app,
    supports_credentials=True,
    resources={
        r"/*": {"origins": [FRONTEND_URL, "http://localhost:3000"]}
    }
)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_URL
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    return response
# --------------------------------------

# Load local embedding model
print("Loading multilingual embedding model... (this will take a few seconds the first time)")
# multilingual model for better cross-lingual matching
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded successfully")

app.secret_key = secrets.token_hex(32)

# --------------------------------------
# MongoDB setup
# --------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["caremate"]
sessions_collection = db["sessions"]
messages_collection = db["messages"]

# --------------------------------------
# Global workflow and conversation state
# --------------------------------------
workflow_app = None
conversation_states = {}


# --------------------------------------
# Doctor recommendation functions
# --------------------------------------

def get_embedding(text):
    """Generate embedding vector for given text using local SentenceTransformer"""
    return np.array(embedder.encode(text, normalize_embeddings=True))


def find_related_doctors(user_message, limit=3):
    print("Started")
    """Find doctors semantically related to user symptoms using local embeddings"""
    doctors = list(db["users"].find({"role": "doctor"}))
    print(doctors)
    if not doctors:
        return []
    # Get user embedding and ensure it's a numpy array
    user_emb = get_embedding(user_message)
    if not isinstance(user_emb, np.ndarray):
        user_emb = np.array(user_emb)

    # Preprocess message for simple keyword matching
    user_text_lc = (user_message or "").lower()

    # Symptom -> specialization keywords mapping (tune as needed)
    SPECIALIZATION_KEYWORDS = {
        "cardiologist": ["chest pain", "shortness of breath", "palpitat", "heart attack", "angina", "tachycardia"],
        "pulmonologist": ["cough", "shortness of breath", "wheeze", "wheezing", "bronchitis", "asthma"],
        "dermatologist": ["rash", "itch", "itching", "redness", "eczema", "psoriasis", "skin"],
        "ent": ["ear", "hearing", "hearing loss", "ear pain", "tinnitus", "hoarseness"],
        "neurology": ["headache", "seizure", "numbness", "weakness", "dizziness", "migraine"],
        "pediatrician": ["child", "kid", "baby", "fever in child", "pediatric"],
        "orthopedist": ["joint pain", "back pain", "fracture", "sprain", "arthritis"],
    }

    # Canonicalization map for specialization names (common typos / variants)
    CANONICAL_SPECIALIZATIONS = {
        "pediadrist": "pediatrician",
        "pediatrist": "pediatrician",
        "pediatrician": "pediatrician",
        "cardiologist": "cardiologist",
        "cardiology": "cardiologist",
        "ent": "ent",
        "ear nose throat": "ent",
        "neurology": "neurology",
        "neuro": "neurology",
        "dermatologist": "dermatologist",
        "derm": "dermatologist",
        "orthopedist": "orthopedist",
        "orthopedic": "orthopedist",
    }

    # Detect which specializations are implied by the user's message
    implied_specs = set()
    for spec, keys in SPECIALIZATION_KEYWORDS.items():
        for k in keys:
            if k in user_text_lc:
                # canonicalize implied spec if present
                implied_specs.add(CANONICAL_SPECIALIZATIONS.get(spec, spec))
                break

    # First pass: canonicalize specializations and build available spec set
    available_specs = set()
    for doc in doctors:
        profile = doc.get("doctorProfile", {})
        raw_spec = (profile.get("specialization", "") or "").strip().lower()
        canon_spec = CANONICAL_SPECIALIZATIONS.get(raw_spec, raw_spec) if raw_spec else ''
        if canon_spec:
            available_specs.add(canon_spec)
        doc['_canonical_specialization'] = canon_spec

    # If implied specialization exists but no doctors of that specialization are available,
    # do NOT immediately return an empty list. Previously this prevented returning any
    # recommendations for short follow-up replies (e.g. "4 days") when the earlier
    # context implied a spec we don't have in the DB. Instead, continue and allow
    # semantic matching to surface relevant doctors (with optional later filtering).
    if implied_specs:
        missing = [implied for implied in implied_specs if implied and implied not in available_specs]
        if missing:
            print(f"[debug] implied specializations not available in DB: {missing} — continuing semantic matching")

    # Second pass: compute similarities and scores
    similarities = []
    score_map = {}
    for doc in doctors:
        profile = doc.get("doctorProfile", {})
        specialization = profile.get("specialization", "")
        qualifications = profile.get("qualifications", [])
        years_exp = profile.get("yearsExperience", "")

        parts = []
        if specialization:
            parts.append(specialization)
        if qualifications:
            parts.append(" ".join(qualifications))
        if years_exp:
            parts.append(f"{years_exp} years experience")
        bio = profile.get("bio", "") or doc.get("bio", "")
        if bio:
            parts.append(bio)
        clinic = profile.get("clinic", "")
        if clinic:
            parts.append(clinic)
        languages = profile.get("languages", [])
        if languages:
            parts.append(" ".join(languages))

        doc_text = " ".join(parts).strip()
        if not doc_text:
            doc_text = f"{doc.get('name','')} {specialization}"

        doc_emb = doc.get("embedding")
        if not doc_emb:
            try:
                # Normalize and persist canonical specialization (if changed)
                profile = doc.get('doctorProfile', {}) or {}
                normalized_profile = normalize_profile(profile)
                if normalized_profile.get('specialization') and normalized_profile.get('specialization') != profile.get('specialization'):
                    db["users"].update_one({"_id": doc["_id"]}, {"$set": {"doctorProfile.specialization": normalized_profile.get('specialization')}})

                # Rebuild doc_text using normalized profile
                doc_text = " ".join([p for p in [
                    normalized_profile.get('specialization',''),
                    " ".join(normalized_profile.get('qualifications', [])) if normalized_profile.get('qualifications') else '',
                    f"{normalized_profile.get('yearsExperience','')} years experience" if normalized_profile.get('yearsExperience') else '',
                    normalized_profile.get('bio','') or doc.get('bio',''),
                    normalized_profile.get('clinic',''),
                    " ".join(normalized_profile.get('languages', [])) if normalized_profile.get('languages') else ''
                ] if p]).strip()

                computed = get_embedding(doc_text)
                db["users"].update_one({"_id": doc["_id"]}, {"$set": {"embedding": computed.tolist()}})
                doc_emb = np.array(computed)
            except Exception as e:
                print(f"Error computing embedding for doctor {doc.get('name')}: {e}")
                continue
        else:
            doc_emb = np.array(doc_emb, dtype=float)

        # Ensure no-zero norm
        user_norm = np.linalg.norm(user_emb)
        doc_norm = np.linalg.norm(doc_emb)
        if user_norm == 0 or doc_norm == 0:
            continue

        similarity = float(np.dot(user_emb, doc_emb) / (user_norm * doc_norm))

        # Boosting logic
        boost = 0.0
        spec_lc = (doc.get('_canonical_specialization') or specialization or "").lower()
        for implied in implied_specs:
            if implied and (implied in spec_lc or spec_lc in implied):
                boost += 0.35
        if specialization and specialization.lower() in user_text_lc:
            boost += 0.12
        for qual in qualifications:
            if qual and qual.lower() in user_text_lc:
                boost += 0.05

        score = similarity + boost
        similarities.append((doc, score))
        score_map[doc.get('_id')] = score

    # Return top matches above a reasonable similarity threshold
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Filter out low-similarity matches (threshold can be tuned)
    SIMILARITY_THRESHOLD = 0.15
    filtered = [d for d in similarities if d[1] >= SIMILARITY_THRESHOLD]

    top_matches = [d[0] for d in filtered[:limit]]

    # Previously we returned an empty list when an implied specialization had no
    # matching doctors in the DB. That behavior caused follow-up messages with
    # little content to produce no recommendations. Keep results (possibly empty)
    # based on similarity filtering above; do not force an early empty return here.

    results = []
    for d in top_matches:
        doc_id = d.get('_id')
        score = score_map.get(doc_id)
        if score is None:
            # fallback: search in similarities list
            score = next((s for (dd, s) in similarities if dd.get('_id') == doc_id), None)
        try:
            score = round(float(score), 4) if score is not None else None
        except Exception:
            score = None

        results.append({
            "name": d.get("name", ""),
            "email": d.get("email", ""),
            "specialization": d.get("doctorProfile", {}).get("specialization", ""),
            "yearsExperience": d.get("doctorProfile", {}).get("yearsExperience", ""),
            "qualifications": d.get("doctorProfile", {}).get("qualifications", []),
            "score": score
        })

    return results


# --------------------------------------
# MongoDB helper functions
# --------------------------------------

def save_message(session_id, role, content, source=None):
    """Save user or assistant message to MongoDB"""
    messages_collection.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "source": source,
        "timestamp": datetime.now(timezone.utc)
    })
    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            "$set": {"last_active": datetime.now(timezone.utc)}
        },
        upsert=True
    )



def get_chat_history(session_id):
    """Retrieve ordered chat messages for a session"""
    messages = list(messages_collection.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1))
    return messages


def get_all_sessions():
    """Fetch all sessions with preview"""
    sessions = []
    for s in sessions_collection.find().sort("last_active", -1):
        first_user_msg = messages_collection.find_one(
            {"session_id": s["session_id"], "role": "user"},
            sort=[("timestamp", 1)]
        )
        preview = None
        if first_user_msg and first_user_msg.get("content"):
            content = first_user_msg["content"]
            preview = content[:50] + "..." if len(content) > 50 else content

        sessions.append({
            "session_id": s["session_id"],
            "created_at": s.get("created_at"),
            "last_active": s.get("last_active"),
            "preview": preview
        })
    return sessions


def delete_session(session_id):
    """Delete all messages + session document"""
    messages_collection.delete_many({"session_id": session_id})
    sessions_collection.delete_one({"session_id": session_id})


# --------------------------------------
# MediGenius Initialization
# --------------------------------------
def initialize_system():
    global workflow_app

    pdf_path = './data/medical_book.pdf'
    persist_dir = './medical_db/'

    print("Initializing MediGenius System...")

    # Initialize vector DB
    existing_db = get_or_create_vectorstore(persist_dir=persist_dir)

    if not existing_db and os.path.exists(pdf_path):
        print("Creating vector database from PDF...")
        doc_splits = process_pdf(pdf_path)
        get_or_create_vectorstore(documents=doc_splits, persist_dir=persist_dir)
    elif not existing_db:
        print("No vector database and no PDF found — RAG features will be limited")

    workflow_app = create_workflow()
    print("MediGenius Web Interface Ready!")


# --------------------------------------
# Flask Routes
# --------------------------------------

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    global workflow_app, conversation_states

    data = request.json
    message = data.get('message', '')
    session_id = session.get('session_id')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    if not workflow_app:
        return jsonify({'error': 'System not initialized'}), 500

    # Save user message
    save_message(session_id, 'user', message)

    # Detect user language and store in conversation state so downstream
    # agents/LLM can respond in the same language.
    user_lang = detect_language(message)

    # Fetch last 5 messages (for context)
    previous_messages = list(messages_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(5))
    previous_messages.reverse()  # so oldest comes first

    # Build context string
    context = ""
    for msg in previous_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    # Initialize or get conversation state
    if session_id not in conversation_states:
        conversation_states[session_id] = initialize_conversation_state()

    conversation_state = conversation_states[session_id]
    conversation_state = reset_query_state(conversation_state)

    # attach detected language
    conversation_state['language'] = user_lang

    # Add both context and question
    conversation_state["context"] = context.strip()
    conversation_state["question"] = message

    # Process query through workflow
    result = workflow_app.invoke(conversation_state)
    conversation_states[session_id].update(result)

    # Build a combined text from recent context + current message so that
    # short replies (e.g., "4 days") are matched against earlier symptom
    # mentions in the conversation. This improves doctor suggestion recall.
    combined_text = (context or '').strip()
    if combined_text and message:
        combined_query = f"{combined_text} {message}"
    else:
        combined_query = message or combined_text

    related_doctors = find_related_doctors(combined_query)
    print(f"[debug] related_doctors found: {len(related_doctors)} for query='{combined_query[:120]}'")

    # Extract response and source
    response = result.get('generation', 'Unable to generate response.')
    source = result.get('source', 'Unknown')

    # Save assistant response
    save_message(session_id, 'assistant', response, source)

    timestamp = datetime.now().strftime("%I:%M %p")

    return jsonify({
        'response': response,
        'source': source,
        'timestamp': timestamp,
        'related_doctors': related_doctors,
        'context_used': context,  # for debugging (optional)
        'success': bool(result.get('generation'))
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return chat history for the active session"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'messages': []})
    messages = get_chat_history(session_id)
    return jsonify({'messages': messages, 'success': True})


@app.route('/api/sessions', methods=['GET'])
def get_sessions_route():
    """Return all sessions"""
    sessions = get_all_sessions()
    return jsonify({'sessions': sessions, 'success': True})


@app.route('/api/session/<session_id>', methods=['GET'])
def load_session(session_id):
    """Load messages from a specific session"""
    session['session_id'] = session_id
    messages = get_chat_history(session_id)
    return jsonify({
        'messages': messages,
        'session_id': session_id,
        'success': True
    })


@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete a specific session"""
    delete_session(session_id)
    if session.get('session_id') == session_id:
        session['session_id'] = str(uuid.uuid4())
    return jsonify({'message': 'Session deleted', 'success': True})


@app.route('/api/clear', methods=['POST'])
def clear():
    """Reset conversation state (in-memory only)"""
    session_id = session.get('session_id')
    if session_id in conversation_states:
        conversation_states[session_id] = initialize_conversation_state()
    return jsonify({'message': 'Conversation cleared', 'success': True})


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Create a new session"""
    new_session_id = str(uuid.uuid4())
    session['session_id'] = new_session_id
    return jsonify({
        'message': 'New chat created',
        'session_id': new_session_id,
        'success': True
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'CareMate'})


# ----- Admin: create or update doctor (canonicalize before write) -----
@app.route('/api/admin/doctor', methods=['POST'])
def upsert_doctor():
    """Create or update a doctor record. This endpoint will canonicalize
    the provided `doctorProfile` using `normalize_profile` before persisting.

    Expected JSON body:
      {
        "name": "Dr. Alice",
        "email": "alice@example.com",
        "doctorProfile": { ... }
      }

    If a doctor with the same email exists, it will be updated (upsert).
    For safety this endpoint is intended for admin use only; add auth in
    front of it for production.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip()
    name = data.get('name') or data.get('displayName') or ''
    raw_profile = data.get('doctorProfile') or {}

    if not email:
        return jsonify({'error': 'email is required', 'success': False}), 400

    # Canonicalize/normalize the profile
    try:
        normalized = normalize_profile(raw_profile or {})
    except Exception as e:
        return jsonify({'error': f'failed to normalize profile: {e}', 'success': False}), 500

    # Build document to upsert
    doc = {
        'name': name,
        'email': email,
        'role': 'doctor',
        'doctorProfile': normalized,
    }

    # Optional top-level bio field
    if normalized.get('bio'):
        doc['bio'] = normalized.get('bio')

    try:
        result = db['users'].update_one({'email': email}, {'$set': doc}, upsert=True)
    except Exception as e:
        return jsonify({'error': f'database upsert failed: {e}', 'success': False}), 500

    # Compute and persist embedding for the doctor text (best-effort)
    try:
        # Build text used for embedding (same logic as find_related_doctors)
        parts = [
            normalized.get('specialization',''),
            ' '.join(normalized.get('qualifications', [])) if normalized.get('qualifications') else '',
            f"{normalized.get('yearsExperience','')} years experience" if normalized.get('yearsExperience') else '',
            normalized.get('bio',''),
            normalized.get('clinic',''),
            ' '.join(normalized.get('languages', [])) if normalized.get('languages') else ''
        ]
        doc_text = ' '.join([p for p in parts if p]).strip()
        if not doc_text:
            doc_text = name or email

        emb = get_embedding(doc_text)
        # store as plain list so other scripts can read
        db['users'].update_one({'email': email}, {'$set': {'embedding': emb.tolist()}})
    except Exception as e:
        # Do not fail the whole request because embedding failed; return warning
        return jsonify({
            'message': 'doctor upserted, but embedding failed',
            'email': email,
            'upserted': bool(result.upserted_id or result.matched_count),
            'warning': str(e),
            'success': True
        }), 200

    return jsonify({
        'message': 'doctor upserted',
        'email': email,
        'upserted': bool(result.upserted_id or result.matched_count),
        'success': True
    })


# --------------------------------------
# Run app
# --------------------------------------
if __name__ == '__main__':
    initialize_system()
    # Bind to all interfaces so remote port-forwarding / tunnels can reach the server.
    app.run(host='0.0.0.0', debug=True, port=8000)