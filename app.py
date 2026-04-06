import gradio as gr
from agent import run_agent, agent
from tools import make_rag_tool

PDF_SOURCE = "https://arxiv.org/pdf/1706.03762"  # "Attention is All You Need"

# Initialize the RAG pipeline once when the app starts
print("Initializing RAG pipeline...")
make_rag_tool(PDF_SOURCE, verbose=False)
print("Ready.\n")

# chat_history is managed by Gradio — we don't use it directly
# because the agent (LangGraph) manages its own conversation state
def respond(message, chat_history):
    return run_agent(message)

gr.ChatInterface(
    fn=respond,
    title="RAG Agent — Attention is All You Need",
    description="Ask anything about the Transformer paper. The agent will search the document, do math, or check the date as needed.",
    examples=[
        "What is multi-head attention?",
        "What BLEU score did the model achieve on English-German translation?",
        "What is 64 divided by 8, then squared?",
        "What is today's date?",
    ],
).launch()
