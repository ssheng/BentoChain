import os
import torch
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self, generate_speech):
        self.lock = Lock()
        self.generate_speech = generate_speech

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                response = "Please paste your OpenAI key to use"
                history.append((inp, response))
                speech = self.generate_speech(response)
                return history, history, speech
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            speech = self.generate_speech(output)

            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, speech


def create_block(chat: ChatWrapper):
    """Create the gradio block."""

    block = gr.Blocks(css=".gradio-container")

    with block:
        with gr.Row():
            gr.Markdown("<h3><center>BentoML LangChain Demo</center></h3>")

            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )

        chatbot = gr.Chatbot()

        audio = gr.Audio(label="Chatbot Voice", elem_id="chatbox_voice")

        with gr.Row():
            message = gr.Textbox(
                label="What's your question?",
                placeholder="What's the answer to life, the universe, and everything?",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

        gr.Examples(
            examples=[
                "Hi! How's it going?",
                "What should I do tonight?",
                "Whats 2 + 2?",
            ],
            inputs=message,
        )

        gr.HTML("Demo BentoML application of a LangChain chain.")

        gr.HTML(
            "<center>Powered by <a href='https://github.com/bentoml/BentoML'>BentoML 🍱</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        )

        state = gr.State()
        agent_state = gr.State()

        submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state, audio])
        message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state, audio])

        openai_api_key_textbox.change(
            set_openai_api_key,
            inputs=[openai_api_key_textbox],
            outputs=[agent_state],
        )

        return block
