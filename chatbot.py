import os
import gradio as gr
from typing import Optional, Tuple
import bentoml
from datasets import Dataset, Audio
from langchain.chains import ConversationChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from threading import Lock


PLAYBACK_SAMPLE_RATE = 16000


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    tools = load_tools(["wikipedia"], llm=llm)
    chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )
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
    def __init__(self, generate_speech, generate_text):
        self.lock = Lock()
        self.generate_speech = generate_speech
        self.generate_text = generate_text
        self.s2t_processor_ref = bentoml.models.get("whisper_processor:latest")
        self.processor = bentoml.transformers.load_model(self.s2t_processor_ref)

    def __call__(
        self,
        api_key: str,
        audio_path: str,
        text_message: str,
        history: Optional[Tuple[str, str]],
        chain: Optional[ConversationChain],
    ):
        """Execute the chat functionality."""
        self.lock.acquire()

        print(f"audio_path : {audio_path} ({type(audio_path)})")
        print(f"text_message : {text_message} ({type(text_message)})")

        try:
            if audio_path is None and text_message is not None:
                transcription = text_message
            elif audio_path is not None and text_message in [None, ""]:
                audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
                    "audio",
                    Audio(sampling_rate=16000),
                )
                sample = audio_dataset[0]["audio"]

                if sample is not None:
                    input_features = self.processor(
                        sample["array"],
                        sampling_rate=sample["sampling_rate"],
                        return_tensors="pt",
                    ).input_features

                    transcription = self.generate_text(input_features)
                else:
                    transcription = None
                    speech = None

            if transcription is not None:
                history = history or []
                # If chain is None, that is because no API key was provided.
                if chain is None:
                    response = "Please paste your Open AI key."
                    history.append((transcription, response))
                    speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(response))
                    return history, history, speech, None, None
                # Set OpenAI key
                import openai

                openai.api_key = api_key
                # Run chain and append input.
                output = chain.run(input=transcription)
                speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(output))
                history.append((transcription, output))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, speech, None, None


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
            audio_message = gr.Audio(
                label="User voice message",
                source="microphone",
                type="filepath",
            )

            text_message = gr.Text(
                label="User text message",
                placeholder="Give me 5 gift ideas for my mother",
            )

        gr.HTML("Demo BentoML application of a LangChain chain.")

        gr.HTML(
            "<center>Powered by <a href='https://github.com/bentoml/BentoML'>BentoML 🍱</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        )

        state = gr.State()
        agent_state = gr.State()

        audio_message.change(
            chat,
            inputs=[
                openai_api_key_textbox,
                audio_message,
                text_message,
                state,
                agent_state,
            ],
            outputs=[chatbot, state, audio, audio_message, text_message],
            show_progress=False,
        )

        text_message.submit(
            chat,
            inputs=[
                openai_api_key_textbox,
                audio_message,
                text_message,
                state,
                agent_state,
            ],
            outputs=[chatbot, state, audio, audio_message, text_message],
            show_progress=False,
        )

        openai_api_key_textbox.change(
            set_openai_api_key,
            inputs=[openai_api_key_textbox],
            outputs=[agent_state],
            show_progress=False,
        )

        return block
