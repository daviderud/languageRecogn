import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# -----------------------------
# Load SpeechBrain model once
# -----------------------------
model = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp",
    local_strategy=LocalStrategy.COPY
)

samplerate = 16000
chunk_duration = 10  # seconds
audio_queue = queue.Queue()


# -----------------------------
# Audio Quality Metrics
# -----------------------------
def audio_quality_metrics(audio):
    x = audio.flatten()

    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    silence_ratio = np.mean(np.abs(x) < 1e-4)

    noise_floor = np.percentile(np.abs(x), 10)
    snr_est = 20 * np.log10((rms + 1e-8) / (noise_floor + 1e-8))

    return rms, peak, silence_ratio, snr_est


# -----------------------------
# Recording Thread
# -----------------------------
def record_loop(device_index):
    while running_flag[0]:
        audio = sd.rec(
            int(chunk_duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device_index
        )
        sd.wait()
        audio_queue.put(audio)


# -----------------------------
# GUI Update Loop
# -----------------------------
def process_audio():
    if not audio_queue.empty():
        audio = audio_queue.get()

        # Update waveform
        ax.clear()
        ax.plot(audio)
        ax.set_ylim([-1, 1])
        canvas.draw()

        # Compute metrics
        rms, peak, silence_ratio, snr_est = audio_quality_metrics(audio)

        rms_var.set(f"{rms:.4f}")
        peak_var.set(f"{peak:.4f}")
        silence_var.set(f"{silence_ratio:.2f}")
        snr_var.set(f"{snr_est:.1f} dB")

        rms_bar['value'] = rms * 100
        peak_bar['value'] = peak * 100
        silence_bar['value'] = silence_ratio * 100
        snr_bar['value'] = max(0, min(50, snr_est + 10))

        # Save and classify
        sf.write("mic_chunk.wav", audio, samplerate)
        logits, _, _, _ = model.classify_file("mic_chunk.wav")
        probs = F.softmax(logits, dim=1)

        top5_prob, top5_idx = torch.topk(probs, k=5)
        top5_prob = top5_prob.squeeze().tolist()
        top5_idx = top5_idx.squeeze().tolist()
        labels = model.hparams.label_encoder.decode_ndim(top5_idx)

        result_text = "\n".join([f"{lbl}: {p:.4f}" for lbl, p in zip(labels, top5_prob)])
        result_label.config(text=result_text)

    root.after(200, process_audio)


# -----------------------------
# Start Recording
# -----------------------------
def start_recording():
    device_index = int(device_combo.get().split(":")[0])
    running_flag[0] = True
    threading.Thread(target=record_loop, args=(device_index,), daemon=True).start()


# -----------------------------
# Stop Recording
# -----------------------------
def stop_recording():
    running_flag[0] = False


# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("Live Language ID with Audio Quality Monitor")

running_flag = [False]

# Device selection
tk.Label(root, text="Select Input Device:").pack()

devices = sd.query_devices()
input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]

device_combo = ttk.Combobox(root, values=input_devices, width=60)
device_combo.pack()
device_combo.current(0)

ttk.Button(root, text="Start", command=start_recording).pack(pady=5)
ttk.Button(root, text="Stop", command=stop_recording).pack(pady=5)

# Waveform plot
fig = Figure(figsize=(6, 3))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Audio quality indicators
def add_metric(label, var):
    frame = tk.Frame(root)
    frame.pack(fill="x")
    tk.Label(frame, text=label, width=20).pack(side="left")
    tk.Label(frame, textvariable=var).pack(side="left")

rms_var = tk.StringVar()
peak_var = tk.StringVar()
silence_var = tk.StringVar()
snr_var = tk.StringVar()

add_metric("RMS:", rms_var)
add_metric("Peak:", peak_var)
add_metric("Silence Ratio:", silence_var)
add_metric("SNR:", snr_var)

# Bars
def add_bar():
    bar = ttk.Progressbar(root, length=400, maximum=100)
    bar.pack()
    return bar

rms_bar = add_bar()
peak_bar = add_bar()
silence_bar = add_bar()
snr_bar = add_bar()

# Classification results
result_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
result_label.pack(pady=10)

# Start GUI loop
root.after(200, process_audio)
root.mainloop()
