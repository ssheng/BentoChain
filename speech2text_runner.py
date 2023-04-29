import torch
import bentoml

s2t_processor_ref = bentoml.models.get("whisper_processor:latest")
s2t_model_ref = bentoml.models.get("whisper_model:latest")


class Speech2TextRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = bentoml.transformers.load_model(s2t_processor_ref)
        self.model = bentoml.transformers.load_model(s2t_model_ref)
        self.model.to(self.device)

    @bentoml.Runnable.method(batchable=False)
    def transcribe_audio(self, tensor):
        if tensor is not None:
            predicted_ids = self.model.generate(tensor.to(self.device))
            transcriptions = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            transcription = transcriptions[0]
            return transcription
