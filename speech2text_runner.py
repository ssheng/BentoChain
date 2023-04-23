import torch
import bentoml
from datasets import Audio, Dataset

s2t_processor_ref = bentoml.models.get("whisper_processor:latest")
s2t_model_ref = bentoml.models.get("whisper_model:latest")


class Speech2TextRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = bentoml.transformers.load_model(s2t_processor_ref)
        self.model = bentoml.transformers.load_model(s2t_model_ref)

    @bentoml.Runnable.method(batchable=False)
    def transcribe_audio(self, audio_path):
        if audio_path is not None:
            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
                "audio",
                Audio(sampling_rate=16000),
            )
            sample = audio_dataset[0]["audio"]

            input_features = self.processor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                return_tensors="pt",
            ).input_features

            predicted_ids = self.model.generate(input_features)
            transcriptions = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            transcription = transcriptions[0]
            return transcription
