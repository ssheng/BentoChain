import logging
import bentoml
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":
    t5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    t5_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    t5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    )
    whisper_model.config.forced_decoder_ids = None

    saved_t5_processor = bentoml.transformers.save_model(
        "speecht5_tts_processor", t5_processor
    )
    print(f"Saved: {saved_t5_processor}")

    saved_t5_model = bentoml.transformers.save_model(
        "speecht5_tts_model",
        t5_model,
        signatures={"generate_speech": {"batchable": False}},
    )
    print(f"Saved: {saved_t5_model}")

    saved_t5_vocoder = bentoml.transformers.save_model(
        "speecht5_tts_vocoder", t5_vocoder
    )
    print(f"Saved: {saved_t5_vocoder}")

    saved_whisper_processor = bentoml.transformers.save_model(
        "whisper_processor",
        whisper_processor,
    )
    print(f"Saved: {saved_whisper_processor}")

    saved_whisper_model = bentoml.transformers.save_model(
        "whisper_model",
        whisper_model,
    )
    print(f"Saved: {saved_whisper_model}")
