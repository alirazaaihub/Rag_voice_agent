# 🎙️ Voice RAG Agent

A real-time voice AI assistant powered by **LiveKit Agents**, **Groq LLM (LLaMA 3.3-70B)**, and a custom **RAG (Retrieval-Augmented Generation)** pipeline. Ask questions out loud — the agent searches your knowledge base and responds naturally in under 3 sentences.

---

## 🧠 Architecture Overview

```
User Voice Input
      │
      ▼
 Groq Whisper STT  ──►  LLaMA 3.3-70B (Groq)  ──►  Cartesia TTS
                               │
                    (if knowledge-base query)
                               │
                               ▼
                      RAG Graph (LangGraph)
                               │
                    Vector Search → Context → Answer
```

---

## ✨ Features

- 🎤 **Real-time voice I/O** via LiveKit WebRTC
- 🔍 **RAG-powered Q&A** using a custom LangGraph pipeline
- 🧠 **LLaMA 3.3-70B** via Groq for ultra-fast inference
- 📝 **Whisper Large v3 Turbo** for accurate speech-to-text
- 🔊 **Cartesia Sonic-3** for natural text-to-speech
- 🔕 **Noise cancellation** (BVC) for clean audio input
- ⚡ **VAD** (Voice Activity Detection) via Silero
- 🛠️ Smart tool-calling: agent decides when to query the knowledge base vs. answer directly

---

## 🗂️ Project Structure

```
├── agent.py          # Main agent logic & LiveKit session setup
├── rag.py            # LangGraph RAG pipeline (rag_graph)
├── .env              # Environment variables (not committed)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/alirazaaihub/Rag_voice_agent.git
cd your-repo-name
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

> Get your Groq API key at [console.groq.com](https://console.groq.com)  
> Get your LiveKit credentials at [cloud.livekit.io](https://cloud.livekit.io)

### 5. Run the Agent

```bash
python agent.py dev
```

---

## 🔧 Configuration

| Component | Model / Provider |
|-----------|-----------------|
| LLM | `llama-3.3-70b-versatile` via Groq |
| STT | `whisper-large-v3-turbo` via Groq |
| TTS | `cartesia/sonic-3` |
| VAD | Silero |
| Noise Cancellation | LiveKit BVC |

---

## 💡 How It Works

1. **User speaks** → Whisper STT transcribes audio in real-time
2. **LLaMA 3.3-70B** receives the transcription and decides:
   - General question → Answers directly
   - Knowledge-base question → Calls `search_knowledge_base` tool
3. **RAG pipeline** (LangGraph) retrieves relevant context from the vector store and generates a grounded answer
4. **Cartesia TTS** converts the answer to speech and plays it back

---

## 📦 Key Dependencies

```txt
livekit-agents
livekit-plugins-openai
livekit-plugins-silero
livekit-plugins-noise-cancellation
langgraph
langchain
python-dotenv
```

> Run `pip freeze > requirements.txt` to generate the full list.

---

## ⚠️ Notes

- The RAG graph (`rag.py`) must be configured separately with your vector store and retriever before running the agent.
- This agent runs as a LiveKit Worker — you need an active LiveKit server (local or cloud) to connect to.
- Never commit your `.env` file. Add it to `.gitignore`.

---

## 📄 License

MIT License — feel free to use and modify.
