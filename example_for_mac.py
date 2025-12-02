import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)

# NOTE: This model is ENGLISH-ONLY. Vietnamese is not supported.
# For Vietnamese support, see vietnamese-support-plan.md

# Test with English text to verify voice cloning
text = "Hello, this is a test of voice cloning. I hope this sounds like the voice in my audio sample."

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "sampling/my_sample.wav"

# Voice cloning parameters:
# - cfg_weight: Lower values = better voice cloning (use 0.1-0.3, not 0.0)
# - exaggeration: 0.0-1.0 controls emotion/expressiveness
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.5,
    cfg_weight=0.2,
    temperature=0.8,
)
ta.save("test_voice_clone.wav", wav, model.sr)
print("DONE - Output saved to test_voice_clone.wav")
