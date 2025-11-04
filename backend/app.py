# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# ---------- Config ----------
DB_PATH = Path(__file__).parent / "tickets.db"
CSV_PATH = "tickets_with_solutions.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRAINED_MODEL_PATH = Path(__file__).parent / "model" / "checkpoint-5807"
EMBEDDING_FILE = "ticket_embeddings.pkl"

# ---------- Initialize Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Load Models ----------
print("🔹 Loading models...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------- Load Dataset ----------
print("🔹 Loading dataset...")
df_tickets = pd.read_csv(CSV_PATH)

# ---------- Load or Precompute Embeddings ----------
if Path(EMBEDDING_FILE).exists():
    print("🔹 Loading precomputed embeddings from file...")

    # Force loading on CPU (even if file was saved from GPU)
    with open(EMBEDDING_FILE, "rb") as f:
        try:
            ticket_embeddings = torch.load(f, map_location=torch.device('cpu'))
        except Exception:
            # Fallback for non-torch pickle data
            f.seek(0)
            ticket_embeddings = pickle.load(f)

    print("✅ Embeddings loaded successfully (CPU mode).")

else:
    print("⚙️ Encoding all ticket texts (first-time only)...")
    ticket_embeddings = embedding_model.encode(
        df_tickets["ticket_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Save embeddings in CPU format for portability
    with open(EMBEDDING_FILE, "wb") as f:
        torch.save(ticket_embeddings.cpu(), f)

    print("✅ Embeddings saved for future use!")

# ---------- Load Fine-tuned Classification Model ----------
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
classification_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)

# ---------- Database Setup ----------
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT,
            priority TEXT,
            embedding BLOB
        );
        """)
        conn.commit()

def insert_ticket_to_db(filename, content, priority, embedding):
    emb_blob = pickle.dumps(embedding)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO tickets (filename, content, priority, embedding) VALUES (?, ?, ?, ?)",
            (filename, content, priority, emb_blob)
        )
        conn.commit()

def fetch_tickets_from_db(search="", priority_filter="All"):
    query = "SELECT id, filename, content, priority FROM tickets WHERE 1=1"
    params = []
    if search:
        query += " AND content LIKE ?"
        params.append(f"%{search}%")
    if priority_filter != "All":
        query += " AND priority = ?"
        params.append(priority_filter)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    return [{"id":r[0], "filename":r[1], "content":r[2], "priority":r[3]} for r in rows]

# ---------- Priority Prediction ----------
def predict_priority(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = classification_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs).item()
    labels = ["Low", "Medium", "High"]
    return labels[label_id] if label_id < len(labels) else "Unknown"

# ---------- Routes ----------
@app.route("/upload", methods=["POST"])
def upload_ticket():
    try:
        if request.is_json:
            data = request.get_json()
            filename = data.get("filename", "unknown")
            content = data.get("content", "").strip()
        elif "file" in request.files:
            f = request.files["file"]
            filename = f.filename
            content = f.read().decode("utf-8").strip()
        else:
            return jsonify({"error": "No file or JSON provided"}), 400

        if not content:
            return jsonify({"error": "Empty content"}), 400

        priority = predict_priority(content)
        embedding = embedding_model.encode(content)

        insert_ticket_to_db(filename, content, priority, embedding)

        return jsonify({
            "message": "Ticket saved",
            "filename": filename,
            "priority": priority,
            "embedding_dim": len(embedding)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_ticket():
    data = request.get_json(force=True)
    text = data.get("content", "")
    if not text:
        return jsonify({"error": "content required"}), 400
    priority = predict_priority(text)
    embedding = embedding_model.encode(text)
    return jsonify({"priority": priority, "embedding_dim": len(embedding)}), 200


@app.route("/tickets", methods=["GET"])
def list_tickets():
    search = request.args.get("search", "")
    priority = request.args.get("priority", "All")
    return jsonify(fetch_tickets_from_db(search, priority)), 200


@app.route("/recommend", methods=["POST"])
def recommend_solution():
    """Recommend a solution using precomputed embeddings (fast)."""
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        query_emb = embedding_model.encode(ticket_text, convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, ticket_embeddings).squeeze()
        best_idx = torch.argmax(similarities).item()

        top_match = df_tickets.iloc[best_idx]
        return jsonify({
            "recommended_solution": top_match["solution"],
            "similarity_score": round(float(similarities[best_idx]), 3)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Run App ----------
print("🚀 Starting Flask app...")
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
