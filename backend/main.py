import os
import requests
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ✅ Load .env locally (no-op on Render — env vars are set via Render dashboard)
load_dotenv()

# ✅ API Keys — read from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ✅ Fail fast at startup if any key is missing
missing = [k for k, v in {
    "GROQ_API_KEY": GROQ_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "SERPER_API_KEY": SERPER_API_KEY,
}.items() if not v]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# ✅ Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# ✅ SentenceTransformer for embeddings
# NOTE: On first boot, this downloads ~90MB model from HuggingFace
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "genai-intel-chat"
index = pc.Index(index_name)

app = FastAPI(title="AI Intel Agent", version="1.0.0")

# In-memory user history (resets on restart — use a DB for persistence)
user_histories = {}

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schema ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class HistoryEntry(BaseModel):
    user_id: str
    query: str
    response: str

# ---------- History Endpoints ----------
@app.post("/save_history")
async def save_history(entry: HistoryEntry):
    history = user_histories.get(entry.user_id, [])
    history.append({"query": entry.query, "response": entry.response})
    user_histories[entry.user_id] = history
    return {"status": "saved"}

@app.get("/get_history/{user_id}")
async def get_history(user_id: str):
    return user_histories.get(user_id, [])

# ---------- Embedding ----------
def get_embedding(text: str):
    try:
        vector = embedding_model.encode(text).tolist()
        return vector
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None

# ---------- Memory ----------
def store_memory(user_id: str, topic: str, full_message: str, category: str = "general"):
    vector = get_embedding(full_message)
    if vector:
        try:
            index.upsert(vectors=[{
                "id": f"{user_id}:{topic}:{category}:{hash(full_message)}",
                "values": vector,
                "metadata": {
                    "user_id": user_id,
                    "topic": topic,
                    "content": full_message,
                    "category": category
                }
            }])
        except Exception as e:
            print(f"[ERROR] Memory store failed: {e}")

def retrieve_memory(user_id: str, query: str, top_k: int = 3):
    vector = get_embedding(query)
    if vector:
        try:
            result = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}}
            )
            return [m["metadata"]["content"] for m in result.get("matches", [])]
        except Exception as e:
            print(f"[ERROR] Memory retrieve failed: {e}")
    return []

# ---------- News ----------
def fetch_news(query: str):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        res = requests.post(
            "https://google.serper.dev/news",
            headers=headers,
            json={"q": query},
            timeout=10
        )
        return res.json().get("news", [])
    except Exception as e:
        print(f"[ERROR] News fetch failed: {e}")
        return []

def summarize_news(news_items: list):
    if not news_items:
        return "No news found."
    headlines = "\n".join([f"{n['title']}: {n['link']}" for n in news_items[:5]])
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": f"Summarize these news headlines concisely:\n{headlines}"}],
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] News summary failed: {e}")
        return "Error summarizing news."

# ---------- Chat ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    uid = req.user_id
    msg = req.message

    try:
        if "news" in msg.lower():
            news = fetch_news(msg)
            summary = summarize_news(news)
            store_memory(uid, "news", summary, "source")
            return {"response": summary}

        memory = retrieve_memory(uid, msg)
        context = "\n".join(memory)
        prompt = f"Context from memory:\n{context}\n\nUser Query:\n{msg}" if context else msg

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        reply = response.choices[0].message.content
        store_memory(uid, "chat", reply, "faq")

        return {"response": reply}

    except Exception as e:
        print(f"[ERROR] Chat failed: {e}")
        return {"response": "Something went wrong. Please try again."}

# ---------- Health ----------
@app.get("/")
def home():
    return {"message": "AI Intel Agent Running 🚀", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
