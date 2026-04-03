from Helper import Helper
import gradio as gr

# Load the helper class
hp = Helper("https://arxiv.org/pdf/1706.03762", verbose=True)
# Define the response function
# chat_history is passed by Gradio automatically to keep track of the conversation in the UI
def respond(message, chat_history):
    bot_message = hp.llm_response(message)
    return bot_message

# Define the interface and launch the app
gr.ChatInterface(respond).launch()
