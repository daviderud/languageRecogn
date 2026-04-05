# 🧩 SpeechBrain + PyTorch + Torchaudio + CUDA (Windows)  
## **Reproducible Installation Guide (Python 3.10)**

This document provides the **exact steps and commands** required to recreate a fully working SpeechBrain environment on any Windows machine with an NVIDIA GPU.

---

## 📦 1. Install Python 3.10

Download and install Python 3.10 from:

- `https://www.python.org/downloads/release/python-31011/` [(python.org in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-31011%2F")

During installation:
- ✔ Enable **Add Python to PATH**

---

## 📁 2. Create and activate the virtual environment

```powershell
py -3.10 -m venv C:\Users\david\Documents\venvdav
C:\Users\david\Documents\venvdav\Scripts\activate.ps1
```

---

## ⬆️ 3. Upgrade pip

```powershell
pip install --upgrade pip
```

---

## ⚡ 4. Install PyTorch + Torchaudio (CUDA 12.1)

### CPU‑only (recommended unless you have CUDA)
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### OR CUDA‑enabled
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧠 5. Install SpeechBrain

```powershell
pip install speechbrain==1.0.3
```

---

## 🔧 6. Install compatible HuggingFace Hub

```powershell
pip install huggingface_hub==0.19.4
```

---

## 🔊 7. Install soundfile backend (required on Windows)

```powershell
pip install soundfile sounddevice
```

---

# 🧪 8. Minimal working SpeechBrain script (Windows‑safe)

```python
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp",
    local_strategy=LocalStrategy.COPY
)

out = model.classify_file("audio.wav")
print(out)
```

---

# 📊 9. Extract top‑5 languages with probabilities

```python
import torch
import torch.nn.functional as F

logits, _, _, _ = model.classify_file("audio.wav")
probs = F.softmax(logits, dim=1)

top5_prob, top5_idx = torch.topk(probs, k=5)
top5_prob = top5_prob.squeeze().tolist()
top5_idx = top5_idx.squeeze().tolist()

labels = model.hparams.label_encoder.decode_ndim(top5_idx)

for p, lbl in zip(top5_prob, labels):
    print(f"{lbl}: {p:.4f}")
```

---

# 🎉 Done

