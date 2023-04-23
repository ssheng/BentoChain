import bentoml
import gradio as gr
from chatbot import create_block, ChatWrapper
from fastapi import FastAPI
from speech2text_runner import Speech2TextRunnable
from text2speech_runner import Text2SpeechRunnable


speech2text_runner = bentoml.Runner(
    Speech2TextRunnable,
    name="speech2text_runner",
)
text2speech_runner = bentoml.Runner(
    Text2SpeechRunnable,
    name="text2speech_runner",
)

svc = bentoml.Service(
    "voicegpt",
    runners=[
        text2speech_runner,
        speech2text_runner,
    ],
)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
def generate_speech(inp: str):
    return text2speech_runner.generate_speech.run(inp)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def generate_text(audio_path: str):
    text = speech2text_runner.transcribe_audio.run(audio_path)
    return text


chat = ChatWrapper(generate_speech, generate_text)
app = FastAPI()
app = gr.mount_gradio_app(app, create_block(chat), path="/chatbot")
svc.mount_asgi_app(app, "/")
