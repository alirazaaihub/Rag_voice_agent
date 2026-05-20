from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, WorkerOptions
from livekit.agents import room_io
from livekit.agents.voice import ModelSettings
from livekit.plugins import noise_cancellation, silero
from livekit.plugins import openai as lk_openai
from livekit.agents import function_tool   # <-- correct import

# RAG imports
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import asyncio
import os

load_dotenv()

# ─── RAG Setup ────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    context: str
    answer: str

rag_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

vector_db = Chroma(
    persist_directory="vectore_db",
    embedding_function=embeddings
)

def retriever_node(state: State):
    docs = vector_db.similarity_search(state["query"], k=3)
    context = "\n".join([f"Context {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    return {"context": context}

def llm_node(state: State):
    prompt = (
        f"Answer the question based on the following context:\n{state['context']}\n"
        f"Question: {state['query']}\n"
        f"If not relevant to the context, say 'I don't know'."
    )
    answer = rag_llm.invoke(prompt)
    return {"answer": answer}

def build_rag_graph():
    g = StateGraph(State)
    g.add_node("retriever", retriever_node)
    g.add_node("llm", llm_node)
    g.add_edge(START, "retriever")
    g.add_edge("retriever", "llm")
    g.add_edge("llm", END)
    return g.compile()

rag_graph = build_rag_graph()
