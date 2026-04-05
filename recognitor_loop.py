import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


print("Available input devices:\n")
devices = sd.query_devices()

input_devices = []
for idx, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        input_devices.append((idx, dev['name']))
        print(f"{idx}: {dev['name']}")

device_index = int(input("\nSelect input device index: "))
print(f"Using device: {devices[device_index]['name']}")




# Load SpeechBrain model once
model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp",
    local_strategy=LocalStrategy.COPY
)

samplerate = 16000
chunk_duration = 10  # seconds

print("\nPress Ctrl+C to stop.\n")

try:
    while True:
        print("Recording 10 seconds...")
        audio = sd.rec(
            int(chunk_duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device_index
        )
        sd.wait()
        print("Recording complete.")

        # Save to temporary WAV
        sf.write("mic_chunk.wav", audio, samplerate)

        # Classify
        logits, _, _, _ = model.classify_file("mic_chunk.wav")
        probs = F.softmax(logits, dim=1)

        # Top‑5 
        top5_prob, top5_idx = torch.topk(probs, k=5)
        top5_prob = top5_prob.squeeze().tolist()
        top5_idx = top5_idx.squeeze().tolist()

        labels = model.hparams.label_encoder.decode_ndim(top5_idx)

        print("\nTop‑5 predictions:")
        for p, lbl in zip(top5_prob, labels):
            print(f"{lbl}: {p:.4f}")
        print("\n---\n")

except KeyboardInterrupt:
    print("\nStopped by user.")
