"""
agent.py
--------
Implements the agentic loop using LangChain + LangGraph.

Instead of manually managing the conversation history and tool dispatch
(like we did with the raw Ollama loop), the framework handles all of that:

  - @tool decorator      →  auto-generates the JSON schema from the docstring
  - ChatOllama           →  connects to our local Ollama model
  - create_react_agent   →  wires the LLM and tools into a loop automatically
  - agent.invoke()       →  runs the full loop and returns all messages

Run this file directly for a demo:
    python agent.py
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from tools import make_rag_tool, search_document, calculator, get_current_date


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL      = "qwen3:8b"
PDF_SOURCE = "https://arxiv.org/pdf/1706.03762"  # "Attention is All You Need"


# ─── Step 1: Define Tools ─────────────────────────────────────────────────────
# The @tool decorator does two things:
#   1. Wraps the function so LangChain can call it.
#   2. Auto-generates the JSON schema from the docstring + type hints.
#      (This is the same JSON you wrote manually in tools.py — now automatic!)

@tool
def search_document_tool(query: str) -> str:
    """
    Search the loaded PDF document for passages relevant to a query.
    Use this when the user asks about the document's content.
    """
    return search_document(query)


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any arithmetic or numeric calculation.
    Example: '(3 + 5) * 2'
    """
    return calculator(expression)


@tool
def get_current_date_tool() -> str:
    """
    Return today's date.
    Use this when the user asks what day or date it is.
    """
    return get_current_date()


TOOLS = [search_document_tool, calculator_tool, get_current_date_tool]


# ─── Step 2: Build the LLM + Agent ───────────────────────────────────────────
# create_react_agent wires the LLM and tools together and manages the loop.
# No need to write the loop manually — it handles tool calls automatically.

llm = ChatOllama(model=MODEL)

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=(
        "You are a helpful assistant with access to tools. "
        "Use search_document_tool to answer questions about the document. "
        "Use calculator_tool for any math. "
        "Use get_current_date_tool when asked about today's date. "
        "Always use a tool when the question requires it."
    ),
)


# ─── Step 3: Helper to strip <think> blocks ───────────────────────────────────
# qwen3 wraps its reasoning in <think>...</think> tags before the final answer.
# We discard that part and return only the answer.

def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


# ─── Step 4: Run the Agent ────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Send a message to the agent and return its final answer.
    Prints each step (tool calls and results) so you can follow along.

    Args:
        user_message: The user's question or request.

    Returns:
        The agent's final answer as a string.
    """
    result = agent.invoke({"messages": [("user", user_message)]})

    # result["messages"] is the full conversation log:
    #   HumanMessage → AIMessage (with tool_calls) → ToolMessage → AIMessage (final)
    # We print each step so students can see what happened behind the scenes.
    print(f"\n{'=' * 60}")
    print(f"  USER: {user_message}")
    print(f"{'=' * 60}")

    for msg in result["messages"][1:]:  # skip the first HumanMessage (already printed)
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            # The model decided to call one or more tools
            for tc in msg.tool_calls:
                print(f"\n  [Tool Called]  {tc['name']}")
                print(f"  [Arguments]    {tc['args']}")

        elif msg_type == "ToolMessage":
            # The result returned by the tool
            preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"  [Tool Result]  {preview}")

        elif msg_type == "AIMessage" and msg.content:
            # Final answer from the model
            print(f"\n  [Final Answer] {strip_thinking(msg.content)}")

    # The last message is always the final AIMessage
    return strip_thinking(result["messages"][-1].content)


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Initialize the RAG pipeline once before running the agent
    print("Initializing... (first run may take ~10 seconds to build the index)\n")
    make_rag_tool(PDF_SOURCE, verbose=False)

    # Demo questions — each one exercises a different tool
    demo_questions = [
        "What is today's date?",                                          # get_current_date_tool
        "What is 1024 divided by 32, then multiplied by 7?",              # calculator_tool
        "What is multi-head attention according to the paper?",           # search_document_tool
        "What BLEU score did the Transformer achieve on English-German?", # search_document_tool
    ]

    print("── Running demo questions ──\n")
    for question in demo_questions:
        run_agent(question)
        print(f"\n{'─' * 60}")

    # Interactive mode
    print("\nInteractive mode — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            run_agent(user_input)
