import os
import requests
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# ✅ Configure Groq (kept as you asked)
groq_client = Groq(api_key="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")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ✅ Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Pinecone (kept key)
pc = Pinecone(api_key="pcsk_6mEt73_2ZH5JtrLugHGaBnASc3aLXFARLNayqijEJHHVhvVJenATzd2d1Wn7oGj5ShCmzn")

index_name = "genai-intel-chat"

# ✅ Ensure index exists (VERY IMPORTANT)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 → 384
        metric="cosine"
    )

index = pc.Index(index_name)

SERPER_API_KEY = "0da379d2affd1fc587d4a472d84265c5f438a83f"

app = FastAPI()

# ⚠️ In-memory (will reset on restart)
user_histories = {}

# ---------- Schema ----------
class HistoryEntry(BaseModel):
    user_id: str
    query: str
    response: str

@app.post("/save_history")
async def save_history(entry: HistoryEntry):
    history = user_histories.get(entry.user_id, [])
    history.append({
        "query": entry.query,
        "response": entry.response
    })
    user_histories[entry.user_id] = history
    return {"status": "saved"}

@app.get("/get_history/{user_id}")
async def get_history(user_id: str):
    return user_histories.get(user_id, [])

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

# ---------- Embedding ----------
def get_embedding(text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None

# ---------- Memory ----------
def store_memory(user_id, topic, full_message, category="general"):
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

def retrieve_memory(user_id, query, top_k=3):
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
def fetch_news(company):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        res = requests.post(
            "https://google.serper.dev/news",
            headers=headers,
            json={"q": company}
        )
        return res.json().get("news", [])
    except Exception as e:
        print(f"[ERROR] News fetch failed: {e}")
        return []

def summarize_news(news_items):
    if not news_items:
        return "No news found."

    headlines = "\n".join([
        f"{n['title']}: {n['link']}" for n in news_items[:5]
    ])

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": f"Summarize:\n{headlines}"}],
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
        # 🧠 News trigger
        if "news" in msg.lower():
            news = fetch_news(msg)
            summary = summarize_news(news)
            store_memory(uid, "news", summary, "source")
            return {"response": summary}

        # 🧠 Memory retrieval
        memory = retrieve_memory(uid, msg)

        prompt = f"""
Context:
{chr(10).join(memory)}

User Query:
{msg}
"""

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
        return {"response": str(e)}  # better debugging

# ---------- Root ----------
@app.get("/")
def home():
    return {"message": "AI Intel Agent Running 🚀"}
