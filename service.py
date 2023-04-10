import bentoml
import torch
import gradio as gr

from chatbot import create_block, ChatWrapper
from fastapi import FastAPI
from datasets import load_dataset


processor_ref = bentoml.models.get("speecht5_tts_processor:latest")
model_ref = bentoml.models.get("speecht5_tts_model:latest")
vocoder_ref = bentoml.models.get("speecht5_tts_vocoder:latest")


class SpeechT5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.processor = bentoml.transformers.load_model(processor_ref)
        self.model = bentoml.transformers.load_model(model_ref)
        self.vocoder = bentoml.transformers.load_model(vocoder_ref)
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    @bentoml.Runnable.method(batchable=False)
    def generate_speech(self, inp: str):
        inputs = self.processor(text=inp, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        return speech.numpy()


text2speech_runner = bentoml.Runner(SpeechT5Runnable, name="speecht5_runner", models=[processor_ref, model_ref, vocoder_ref])
svc = bentoml.Service("voicegpt", runners=[text2speech_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
def generate_speech(inp: str):
    return text2speech_runner.generate_speech.run(inp)


chat = ChatWrapper(generate_speech)
app = FastAPI()
app = gr.mount_gradio_app(app, create_block(chat), path="/chatbot")
svc.mount_asgi_app(app, "/")
