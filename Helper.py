import os
import pandas as pd
import numpy as np
from ollama import chat, ChatResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sklearn.metrics.pairwise import cosine_similarity


def get_response(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the model's reply.
    If the model uses <think>...</think> reasoning blocks (e.g. qwen3),
    only the final answer after </think> is returned.
    """
    response: ChatResponse = chat(
        model='qwen3:8b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ]
    )
    text = response.message.content
    result = text.split('</think>')[-1].strip()
    return result


class Helper:
    def __init__(self, pdf_path, verbose=False):
        # Load the PDF document
        self.documents = PyPDFLoader(pdf_path).load_and_split()
        # Store the last 5 exchanges so the model remembers the conversation
        # Each item is a tuple: (user_message, ai_response)
        self.history = []
        self.k = 5
        # Set verbose=True to print what happens at each step (great for learning!)
        self.verbose = verbose

    def _log(self, title, content):
        # Internal helper — prints a labeled section only when verbose=True
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print('=' * 60)
            print(content)

    def splitter(self, chunk_size=1024, chunk_overlap=64):
        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(self.documents)
        return texts

    def vector_store(self, embedding_model='intfloat/multilingual-e5-large-instruct'):
        HOME = os.getcwd()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        texts = self.splitter()

        # Load the existing vector store if it already exists
        if os.path.exists(os.path.join(HOME, "faiss_index")):
            faiss_index = FAISS.load_local("./faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            return faiss_index
        # Otherwise create a new vector store and save it
        else:
            faiss_index = FAISS.from_documents(texts, self.embeddings)
            faiss_index.save_local("./faiss_index")
            return faiss_index

    def retrival_with_score(self, question, k=2, score=1.0):
        # Retrieve the most relevant documents based on the question
        faiss_index = self.vector_store()
        # Prefix the query with an instruction for better retrieval with e5 models
        prefixed_question = "Instruct: Given a question, retrieve relevant passages\nQuery: " + question
        matched_docs = faiss_index.similarity_search_with_score(prefixed_question, k=k, score_threshold=score)

        context = ""
        log_chunks = ""
        for i, (doc, chunk_score) in enumerate(matched_docs):
            context += doc.page_content + "\n"
            # Build a readable summary of each retrieved chunk for logging
            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            log_chunks += f"\n[Chunk {i+1}] Page: {doc.metadata.get('page', 'N/A')} | Score: {chunk_score:.4f}\n{preview}\n"

        self._log("RETRIEVED CHUNKS", log_chunks if log_chunks else "(none found)")
        # If no relevant documents are found, return a default context
        context = context if context else "No relevant information found; please answer the question directly based on your general knowledge."
        return context

    def build_prompt(self, context, history, input, template=''):
        # Build the prompt using a custom template if provided, otherwise use the default
        if template:
            self.template = template
        else:
            self.template = """
            ### System:
            You are an advanced AI assistant trained to follow instructions meticulously and assist with detailed, accurate responses. Your goal is to leverage the provided context to deliver the most relevant and comprehensive answer possible. If the context does not directly relate to the question, use your general knowledge to formulate an appropriate response.

            ### Current conversation:
            {history}

            ### Context:
            {context}

            ### Question:
            {input}

            ### AI Assistant:
            """
        # Fill in the placeholders and return a plain string ready to send to the model
        return self.template.format(history=history, context=context, input=input)

    def llm_response(self, input):
        self._log("USER QUESTION", input)

        # Retrieve relevant context from the document
        context = self.retrival_with_score(input)

        # Format history as a readable string from the list of past exchanges
        history = ""
        for user_msg, ai_msg in self.history:
            history += f"User: {user_msg}\nAI Assistant: {ai_msg}\n"
        self._log("CONVERSATION HISTORY", history if history else "(no history yet)")

        # Build the full prompt string
        prompt = self.build_prompt(context=context, history=history, input=input)
        self._log("FULL PROMPT SENT TO MODEL", prompt)

        # Get the model's response
        response = get_response(prompt)
        self._log("MODEL RESPONSE", response)

        # Save this exchange and keep only the last k turns
        self.history.append((input, response))
        if len(self.history) > self.k:
            self.history.pop(0)
        return response

    def eval_pipeline(self, questions_path):
        THRESH = 0.5
        df = pd.read_csv(questions_path)
        QA = []
        total_score = 0

        # Get the model's answer for each question and store the embeddings
        for q in df.questions:
            bot_message = self.llm_response(q)
            w1 = self.embeddings.embed_query(q)
            w2 = self.embeddings.embed_query(bot_message)
            QA.append((np.array(w1), np.array(w2)))

        # Calculate the cosine similarity score between each question and its answer
        for q, a in QA:
            # cosine_similarity returns a 2D array, so [0][0] extracts the actual score
            cos_score = cosine_similarity(q.reshape(1, -1), a.reshape(1, -1))[0][0]
            if cos_score > THRESH:
                total_score += 1

        return (total_score / len(QA)) * 100
