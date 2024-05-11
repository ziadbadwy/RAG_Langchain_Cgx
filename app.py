from helper import helper
import gradio as gr

# Load the helper class
hp = helper("https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_AI-Index-Report-2024.pdf")
# Load the LLM model
llm = hp.build_agent(verbose=False,streaming=True,max_tokens=200,gpt4all_path='Meta-Llama-3-8B-Instruct.Q4_0.gguf')
# Define the response function
def respond(message, chat_history):
    bot_message = hp.llm_response(message,llm)
    return bot_message
# Define the interface and launch the app
gr.ChatInterface(respond).launch()