import bentoml
import torch
from datasets import load_dataset

t2s_processor_ref = bentoml.models.get("speecht5_tts_processor:latest")
t2s_model_ref = bentoml.models.get("speecht5_tts_model:latest")
t2s_vocoder_ref = bentoml.models.get("speecht5_tts_vocoder:latest")


class Text2SpeechRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = bentoml.transformers.load_model(t2s_processor_ref)
        self.model = bentoml.transformers.load_model(t2s_model_ref)
        self.vocoder = bentoml.transformers.load_model(t2s_vocoder_ref)
        self.embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
        )
        self.speaker_embeddings = torch.tensor(
            self.embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)
        self.model.to(self.device)
        self.vocoder.to(self.device)

    @bentoml.Runnable.method(batchable=False)
    def generate_speech(self, inp: str):
        inputs = self.processor(text=inp, return_tensors="pt")
        speech = self.model.generate_speech(
            inputs["input_ids"].to(self.device),
            self.speaker_embeddings.to(self.device),
            vocoder=self.vocoder,
        )
        return speech.cpu().numpy()
