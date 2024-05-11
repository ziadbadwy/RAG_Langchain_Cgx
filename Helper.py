import os
import torch
import pandas as pd
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain,LLMChain
from sklearn.metrics.pairwise import cosine_similarity



class helper:
    def __init__(self,pdf_path):
        # Load the PDF document
        self.documents = PyPDFLoader(pdf_path).load_and_split()
        self.memory = ConversationBufferWindowMemory(ai_prefix="AI Assistant", memory_key = 'history',k=5)
        self.device = "gpu" if torch.cuda.is_available() else "cpu"

    def splitter(self,chunck_size=1024,chunk_overlap=64):
        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size,chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(self.documents)
        return texts
    
    def vector_store(self,embbiding_model='sentence-transformers/all-MiniLM-L6-v2'):
        # Load the embeddings model
        HOME = os.getcwd()
        self.embeddings = HuggingFaceEmbeddings(model_name=embbiding_model)
        texts = self.splitter()
        # Create a vector store if it does not exist
        if os.path.exists(os.path.join(HOME,"faiss_index")):
            faiss_index = FAISS.load_local("./faiss_index", self.embeddings,allow_dangerous_deserialization=True)
            return faiss_index
        # Load the existing vector store
        else:
            faiss_index = FAISS.from_documents(texts, self.embeddings)
            faiss_index.save_local("./faiss_index")
            faiss_index = FAISS.load_local("./faiss_index", self.embeddings,allow_dangerous_deserialization=True)
            return faiss_index
            
        
    def retrival_with_score(self,question,k=2,score=1.0):
        # Retrieve the most relevant documents based on the question
        faiss_index = self.vector_store()
        matched_docs = faiss_index.similarity_search_with_score(question, k=k,score_threshold=score)
        context = ""
        for doc in matched_docs:
            context += doc[0].page_content + "\n"
        # If no relevant documents are found, return a default context
        context = context if context else "No relevant information found; please answer the question directly based on your general knowledge."
        return context
    
    def build_prompt(self,context,question,template=''):
        # Build the prompt if a custom template is not provided
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
        prompt = PromptTemplate.from_template(template=self.template).partial(
            question=question, context=context)
        
        return prompt
    
    def build_agent(self,gpt4all_path='Meta-Llama-3-8B-Instruct.Q4_0.gguf',max_tokens=200,streaming=True,verbose=True,repeat_last_n=0):
        # Build the language model
        callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

        llm = GPT4All(model=gpt4all_path
                    ,max_tokens=max_tokens
                    ,device=self.device
                    ,streaming=streaming
                    ,callback_manager=callback_manager
                    , verbose=verbose
                    ,repeat_last_n=repeat_last_n)
        
        return llm
    
    def build_chain(self,llm,prompt):
        # Build the conversation chain
        llm_chain = ConversationChain(
                                    prompt=prompt,
                                    llm=llm,
                                    memory=self.memory
                                )
        return llm_chain
    
    def llm_response(self,input,llm):
        # Generate a response from the language model
        self.question = input
        context = self.retrival_with_score(self.question)
        self.prompt = self.build_prompt(context=context,question=self.question)
        llm_chain = self.build_chain(llm,self.prompt)
        response = llm_chain.predict(input=input)
        return response
    def eval_pipeline(self,llm,qusetions_path):
        # Load the questions
        THRESH = 0.5
        df = pd.read_csv(qusetions_path)
        QA = []
        total_score = 0
        # Evaluate the pipeline
        for q in df.questions:
            bot_message = self.llm_response(q,llm)
            w1 = self.embeddings.embed_query(q)
            w2 = self.embeddings.embed_query(bot_message)
            QA.append((np.array(w1), np.array(w2)))
        # Calculate the cosine similarity score
        for q,a in QA:
            cos_score = cosine_similarity(q.reshape(1,-1),a.reshape(1,-1))
            if cos_score > THRESH:
                total_score += 1
            else:    
                total_score += 0
        return (total_score/len(QA)) * 100

                