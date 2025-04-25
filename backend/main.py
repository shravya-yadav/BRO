import os
import requests
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from fastapi.middleware.cors import CORSMiddleware

genai.configure(api_key="AIzaSyA62GAHTjnfqI602xCxTzNEnW4k6uQ-LOw")
model = genai.GenerativeModel("gemini-1.5-flash-latest")

pc = Pinecone(api_key="pcsk_5JQvGu_Sexw9S6kQvcq5QuPXkETxKp7dTgWfhaKUgxKoNfyJHU1xWAgoTpvSNQLzhQg1Qo")
index_name = "genai-intel-chat"

index = pc.Index(index_name)

SERPER_API_KEY = "0da379d2affd1fc587d4a472d84265c5f438a83f"

app = FastAPI()

# Dummy history store (use DB or file in real project)
user_histories = {}

# Define the request schema for saving history
class HistoryEntry(BaseModel):
    user_id: str
    query: str
    response: str

# ✅ Add this route to save history
@app.post("/save_history")
async def save_history(entry: HistoryEntry):
    history = user_histories.get(entry.user_id, [])
    history.append({
        "query": entry.query,
        "response": entry.response
    })
    user_histories[entry.user_id] = history
    return {"status": "saved"}

# ✅ Add this route to get history
@app.get("/get_history/{user_id}")
async def get_history(user_id: str):
    return user_histories.get(user_id, [])

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

# ---------- Embedding & Memory ----------
def get_embedding(text):
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return res["embedding"]
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None

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
            return [match["metadata"]["content"] for match in result.get("matches", [])]
        except Exception as e:
            print(f"[ERROR] Memory retrieve failed: {e}")
    return []

# ---------- Session Initialization ----------
@app.get("/start_session/{user_id}")
def start_session(user_id: str):
    try:
        result = index.query(
            vector=[0.0]*768,
            top_k=15,
            include_metadata=True,
            filter={"user_id": {"$eq": user_id}}
        )
        categories = {"faq": [], "preference": [], "source": []}
        for match in result.get("matches", []):
            cat = match["metadata"].get("category", "general")
            if cat in categories:
                categories[cat].append(match["metadata"]["content"])
        return {"session_memory": categories}
    except Exception as e:
        print(f"[ERROR] Session load failed: {e}")
        return {"session_memory": {}}

# ---------- News ----------
def fetch_news(company):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": company}
    try:
        res = requests.post("https://google.serper.dev/news", headers=headers, json=payload)
        res.raise_for_status()
        return res.json().get("news", [])
    except Exception as e:
        print(f"[ERROR] News fetch failed: {e}")
        return []

def summarize_news(news_items):
    if not news_items:
        return "No news found."
    headlines = "\n".join([f"{n['title']}: {n['link']}" for n in news_items[:5]])
    prompt = f"Summarize the following headlines:\n{headlines}"
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        print(f"[ERROR] News summary failed: {e}")
        return "Error summarizing news."

# ---------- Web Search ----------
def fetch_web(query):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    try:
        res = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
        res.raise_for_status()
        return res.json().get("organic", [])
    except Exception as e:
        print(f"[ERROR] Web fetch failed: {e}")
        return []

def summarize_web_results(results):
    if not results:
        return "No relevant results found."
    text = "\n".join([f"{r['title']}: {r['link']}" for r in results[:5]])
    prompt = f"Summarize this web info and include citations:\n{text}"
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        print(f"[ERROR] Web summarization failed: {e}")
        return "Unable to summarize web results."

# ---------- Market Comparison ----------
def compare_market(company, competitors):
    prompt = f"Compare {company} with {', '.join(competitors)} in terms of market share and strategy."
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        return "Error comparing companies."

# ---------- Chat Endpoint ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    uid = req.user_id
    msg = req.message

    try:
        # News Handling
        if "news" in msg.lower() or "latest" in msg.lower():
            target = msg.split("about")[-1].strip() if "about" in msg else msg
            news = fetch_news(target)
            summary = summarize_news(news)
            final_response = f"Here’s the latest news about **{target}**:\n\n{summary}"
            store_memory(uid, target, final_response, category="source")
            return {"response": final_response}

        # Web Search with Source Citation
        elif "web" in msg.lower() or "search" in msg.lower():
            results = fetch_web(msg)
            summary = summarize_web_results(results)
            store_memory(uid, "web_search", summary, category="source")
            return {"response": summary}

        # Market Comparison
        elif "compare" in msg.lower():
            parts = msg.lower().split("compare")[-1].strip().split("with")
            company = parts[0].strip()
            competitors = [c.strip() for c in parts[1].split("and")]
            result = compare_market(company, competitors)
            final_response = f"Market Comparison between **{company}** and **{', '.join(competitors)}**:\n\n{result}"
            store_memory(uid, "comparison", final_response, category="preference")
            return {"response": final_response}

        # General Chat with Memory
        memory = retrieve_memory(uid, msg)
        prompt = f"Context:\n{chr(10).join(memory)}\n\nUser Query:\n{msg}"
        response = model.generate_content(prompt).text
        store_memory(uid, "general_chat", response, category="faq")
        return {"response": response}

    except Exception as e:
        print(f"[ERROR] Chat handling failed: {e}")
        return {"response": "Oops! Something went wrong."}

# ---------- History ----------
@app.get("/history/{user_id}")
def get_user_history(user_id: str):
    try:
        query_res = index.query(
            vector=[0.0] * 768,
            top_k=20,
            include_metadata=True,
            filter={"user_id": {"$eq": user_id}}
        )
        return {
            "history": [match["metadata"]["content"] for match in query_res.get("matches", [])]
        }
    except Exception as e:
        print(f"[ERROR] History fetch failed: {e}")
        return {"history": []}

@app.get("/")
def home():
    return {"message": "AI Intel Agent Running 🚀"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
