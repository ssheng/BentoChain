import bentoml
import logging

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    saved_processor = bentoml.transformers.save_model("speecht5_tts_processor", processor)
    print(f"Saved: {saved_processor}")
    
    saved_model = bentoml.transformers.save_model("speecht5_tts_model", model)
    print(f"Saved: {saved_model}")
    
    saved_vocoder = bentoml.transformers.save_model("speecht5_tts_vocoder", vocoder)
    print(f"Saved: {saved_vocoder}")
