import bentoml
import gradio as gr
from chatbot import create_block, ChatWrapper
from fastapi import FastAPI
from speech2text_runner import s2t_processor_ref, s2t_model_ref, Speech2TextRunnable
from text2speech_runner import (
    t2s_processor_ref,
    t2s_model_ref,
    t2s_vocoder_ref,
    Text2SpeechRunnable,
)


speech2text_runner = bentoml.Runner(
    Speech2TextRunnable,
    name="speech2text_runner",
    models=[s2t_processor_ref, s2t_model_ref],
)
text2speech_runner = bentoml.Runner(
    Text2SpeechRunnable,
    name="text2speech_runner",
    models=[t2s_processor_ref, t2s_model_ref, t2s_vocoder_ref],
)

svc = bentoml.Service(
    "voicegpt",
    runners=[
        text2speech_runner,
        speech2text_runner,
    ],
)


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.Text())
def generate_text(tensor):
    text = speech2text_runner.transcribe_audio.run(tensor)
    return text


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
def generate_speech(inp: str):
    return text2speech_runner.generate_speech.run(inp)


chat = ChatWrapper(generate_speech, generate_text)
app = FastAPI()
app = gr.mount_gradio_app(app, create_block(chat), path="/chatbot")
svc.mount_asgi_app(app, "/")
