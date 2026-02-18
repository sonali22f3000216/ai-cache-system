import time
import hashlib
import numpy as np
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MAX_CACHE_SIZE = 1500
TTL_SECONDS = 86400
SIM_THRESHOLD = 0.95

MODEL_COST = 0.60
AVG_TOKENS = 800



cache = OrderedDict()

analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0
}



class QueryRequest(BaseModel):
    query: str
    application: str



def normalize(text):
    return text.strip().lower()

def hash_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def call_llm(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        temperature=0
    )
    return response.choices[0].message.content

def clean_expired():
    now = time.time()
    expired = []
    for key, value in cache.items():
        if now - value["timestamp"] > TTL_SECONDS:
            expired.append(key)
    for key in expired:
        del cache[key]

def enforce_lru():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)



@app.get("/")
def root():
    return {"message": "Caching API running"}



@app.post("/")
def handle_query(req: QueryRequest):

    start = time.time()
    analytics["totalRequests"] += 1
    clean_expired()

    normalized = normalize(req.query)
    key = hash_key(normalized)

    # Exact Match
    if key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(key)
        latency = int((time.time() - start) * 1000)
        return {
            "answer": cache[key]["answer"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # Semantic Match
    try:
        query_embedding = get_embedding(normalized)
        for k, value in cache.items():
            similarity = cosine_similarity(query_embedding, value["embedding"])
            if similarity > SIM_THRESHOLD:
                analytics["cacheHits"] += 1
                cache.move_to_end(k)
                latency = int((time.time() - start) * 1000)
                return {
                    "answer": value["answer"],
                    "cached": True,
                    "latency": latency,
                    "cacheKey": k
                }
    except:
        pass

    # Miss → LLM
    analytics["cacheMisses"] += 1
    answer = call_llm(normalized)
    embedding = get_embedding(normalized)

    cache[key] = {
        "answer": answer,
        "embedding": embedding,
        "timestamp": time.time()
    }

    enforce_lru()

    latency = int((time.time() - start) * 1000)

    return {
        "answer": answer,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }



@app.get("/analytics")
def analytics_endpoint():

    total = analytics["totalRequests"]
    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]

    hit_rate = hits / total if total > 0 else 0

    baseline_cost = (total * AVG_TOKENS / 1_000_000) * MODEL_COST
    actual_cost = (misses * AVG_TOKENS / 1_000_000) * MODEL_COST
    savings = baseline_cost - actual_cost

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": round(hit_rate * 100, 2),
        "strategies": [
            "exact match caching",
            "semantic caching",
            "LRU eviction",
            "TTL expiration"
        ]
    }
