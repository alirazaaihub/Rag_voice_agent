from rag import rag_graph
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, WorkerOptions
from livekit.agents import room_io
from livekit.agents.voice import ModelSettings
from livekit.plugins import noise_cancellation, silero
from livekit.plugins import openai as lk_openai
from livekit.agents import function_tool
import asyncio
import os

load_dotenv()

# ─── RAG Setup ────────────────────────────────────────────────────────────────

def _run_rag_sync(query: str) -> str:
    result = rag_graph.invoke({"query": query, "context": "", "answer": ""})
    answer = result["answer"]
    if hasattr(answer, "content"):
        return answer.content
    return str(answer)

# ─── Agent with @function_tool ────────────────────────────────────────────────

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a slightly sarcastic voice AI for tech support. "
                'when a user asks general questions, answer directly with out calling the tool. '
                "Whenever a user question that is related to the knowledge base, use the `search_knowledge_base` tool first. "
                "Never answer from your own knowledge — always use the tool. "
                "After getting the tool result, deliver the answer naturally under 3 sentences."
            ),
        )

    @function_tool
    async def search_knowledge_base(self, query: str) -> str:
        """
        Search the knowledge base to answer user questions.
        Always call this tool when the user asks anything.

        Args:
            query: The user's question to search for in the knowledge base.
        """
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, _run_rag_sync, query)
        return answer


# ─── Server & Session ─────────────────────────────────────────────────────────

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    groq_llm = lk_openai.LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    groq_stt = lk_openai.STT(
        model="whisper-large-v3-turbo",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        language="en",
    )

    session = AgentSession(
        stt=groq_stt,
        llm=groq_llm,
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
