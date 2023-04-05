import bentoml
import torch
import gradio as gr

from fastapi import FastAPI

from datasets import load_dataset


processor_ref = bentoml.models.get("speecht5_tts_processor:latest")
model_ref = bentoml.models.get("speecht5_tts_model:latest")
vocoder_ref = bentoml.models.get("speecht5_tts_vocoder:latest")


PLAYBACK_SAMPLE_RATE=16000


class SpeechT5Runnable(bentoml.Runnable):

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
        return (PLAYBACK_SAMPLE_RATE, speech.numpy())


text2speech_runner = bentoml.Runner(SpeechT5Runnable, name="speecht5_runner", models=[processor_ref, model_ref, vocoder_ref])
svc = bentoml.Service("talk_gpt", runners=[text2speech_runner])


from chatbot import create_block, ChatWrapper

chat = ChatWrapper(text2speech_runner.generate_speech.run)
app = FastAPI()
app = gr.mount_gradio_app(app, create_block(chat), path="/chatbot")

svc.mount_asgi_app(app, "/")
