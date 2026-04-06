"""
tools.py
--------
Defines the tools available to the agent.

Each tool is:
  1. A plain Python function that does the actual work.
  2. A JSON schema (dict) that tells the LLM what the tool is and how to call it.

The TOOL_REGISTRY maps each tool's name to its function so the agent can
call the right function when the model requests it.
"""

import datetime
from Helper import Pipeline


# ─── RAG Pipeline Singleton ───────────────────────────────────────────────────
# We create the pipeline once and reuse it — loading the PDF and embeddings
# model is expensive (~10 seconds) so we never want to do it twice.

_rag_pipeline = None


def make_rag_tool(pdf_path: str, verbose: bool = False) -> None:
    """
    Initialize the RAG pipeline from a PDF file or URL.
    You MUST call this once before running the agent.

    Args:
        pdf_path : Path or URL to the source PDF document.
        verbose  : If True, the RAG pipeline prints its internal steps.
    """
    global _rag_pipeline
    print(f"Loading document: {pdf_path}")
    _rag_pipeline = Pipeline(pdf_path, verbose=verbose)
    print("RAG pipeline ready.")


# ─── Tool Functions ───────────────────────────────────────────────────────────

def search_document(query: str) -> str:
    """
    Search the loaded PDF document for passages relevant to the query.

    Args:
        query: The question or topic to look up in the document.

    Returns:
        The most relevant text passages found, or an error message.
    """
    if _rag_pipeline is None:
        return "ERROR: RAG pipeline not initialized. Call make_rag_tool() first."
    return _rag_pipeline.retrival_with_score(query)


def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, **, parentheses.

    Args:
        expression: A math expression as a string, e.g. '(3 + 5) * 2'.

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        # We pass empty __builtins__ to block any non-math operations.
        # NOTE for students: eval() can be dangerous with untrusted input
        # in production — always sanitize or use a dedicated math library.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"ERROR: Could not evaluate '{expression}'. Reason: {e}"


def get_current_date() -> str:
    """
    Return today's date in a human-readable format.

    Returns:
        Today's date, e.g. 'Sunday, April 06, 2025'.
    """
    return datetime.date.today().strftime("%A, %B %d, %Y")


# ─── Tool Schemas (JSON) ──────────────────────────────────────────────────────
# These dicts follow the OpenAI / Ollama function-calling schema.
# The model reads these to understand what tools exist and how to call them.
# Tip: print(json.dumps(SEARCH_DOCUMENT_TOOL, indent=2)) to inspect any schema.

SEARCH_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search the loaded PDF document for passages relevant to a query. "
            "Use this when the user asks about the document's content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to look up in the document.",
                }
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Use this for any arithmetic or numeric calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '(3 + 5) * 2'.",
                }
            },
            "required": ["expression"],
        },
    },
}

GET_CURRENT_DATE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Return today's date. Use this when the user asks what day or date it is.",
        "parameters": {
            "type": "object",
            "properties": {},  # no arguments needed
            "required": [],
        },
    },
}

# Single list passed to ollama.chat(tools=TOOL_LIST)
TOOL_LIST = [SEARCH_DOCUMENT_TOOL, CALCULATOR_TOOL, GET_CURRENT_DATE_TOOL]

# Maps each tool name (as the model emits it) to the actual Python function.
# The agent uses this as a dispatch table: TOOL_REGISTRY[name](**args)
TOOL_REGISTRY = {
    "search_document":  search_document,
    "calculator":       calculator,
    "get_current_date": get_current_date,
}
