# LLAMA 3 RAG Langchain

## Introduction
Welcome to LLAMA 3 RAG Langchain! This chatbot, powered by Meta's LLAMA 3 model, answers questions from PDF documents and remembers past interactions for seamless dialogue. It's designed to be safe and user-friendly, making it easy to get reliable answers quickly.
## Try It on Google Colab

🔗<a target="_blank" href="https://colab.research.google.com/drive/1RZ13Gqk6T0kAZ-s8cjux572EACDb6r6a?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> You can also interact with LLAMA 3 RAG Langchain directly on Google Colab. Just click the link, and start using the chatbot in a Colab notebook with ease!

## Installation

### Setting Up

To get started with LLAMA 3 RAG Langchain, follow these easy steps to set up the project on your local machine:

1. **Clone the Repository** 📥
   - Clone this repo to your local environment using Git:
     bash
     git clone https://github.com/yourusername/LLAMA-3-RAG-Langchain.git
     cd LLAMA-3-RAG-Langchain
     

2. **Install Dependencies** 🛠
   - Ensure you have Python installed and then set up the necessary libraries:
     bash
     pip install -r requirements.txt
     

3. **Set Up Helper.py** 📄
   - Helper.py is a crucial class that contains all the functions you'll need. Here's what it does:
     - *Load the PDF*: Open and read your PDF document.
     - *Chunk Processing*: Split the document into chunks.
     - *Load Embeddings Model*: Prepare the embeddings model for document analysis.
     - *Vector Store Creation*: Create a vector store to hold your document embeddings.
     - *Document Retrieval*: Retrieve the most relevant documents based on your questions.
     - *Prompt Building*: Automatically build prompts or use a custom template if provided.
     - *Response Generation*: Generate responses using the LLAMA 3 model.
     - *Evaluation*: Build an evaluation pipeline to measure the model's performance.

4. **Running the Application** 🚀
   - Start the application to begin interacting with the LLAMA 3 model:
     bash
     python app.py
     
## User Interface with Gradio

To enhance user accessibility:
- **Built with Gradio.io**: This integration allows you to interact directly with the chatbot via a straightforward web interface.
- **No Code Interaction**: Launch the interface and start engaging with the bot immediately—no need to interact with the code or make modifications.
- **User-Friendly**: Designed for ease of use, enabling you to focus on getting answers without any technical Background.

### Note 📝
   - Ensure that your system has at least *8 GB of free RAM* available to run this application smoothly. This requirement is crucial for handling the intensive computational processes involved.

Follow these steps to ensure a smooth setup and start using LLAMA 3 for document-based question answering!
