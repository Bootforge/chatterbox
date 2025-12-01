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
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "sampling/my_sample.wav"

# Voice cloning parameters:
# - cfg_weight: Lower values = better voice cloning (0.0 not supported, use 0.1-0.3)
# - exaggeration: 0.0-1.0 controls emotion/expressiveness (values > 1.0 may cause artifacts)
# wav = model.generate(
#     text,
#     audio_prompt_path="sampling/my_sample.wav",
#     exaggeration=0.5,   # Reduced from 2.0 to avoid artifacts
#     cfg_weight=0.2,     # Low value for good voice cloning (0.0 causes error due to CFG requirement)
#     temperature=0.8,
#     )
# ta.save("test-2.wav", wav, model.sr)

wav = model.generate(
    text,
    audio_prompt_path="sampling/my_sample_2.wav",
    exaggeration=0.5,   # Reduced from 2.0 to avoid artifacts
    cfg_weight=0.1,     # Low value for good voice cloning (0.0 causes error due to CFG requirement)
    temperature=0.8,
    )
ta.save("test-3.wav", wav, model.sr)
print("DONE")
