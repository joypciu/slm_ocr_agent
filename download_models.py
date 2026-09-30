"""Fetch the model weights and the llama.cpp server this project runs on (nothing large is stored in git).
   python download_models.py            # everything (about 0.7 GB)
   python download_models.py docs       # text reasoner only
"""
import os, subprocess, sys, zipfile, urllib.request, json

which = sys.argv[1] if len(sys.argv) > 1 else "all"
os.makedirs("models/vision", exist_ok=True)
os.makedirs("runtime", exist_ok=True)
from huggingface_hub import hf_hub_download

# text reasoner (~378 MB)
hf_hub_download("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", local_dir="models")
# vision specialist (~280 MB): language model + image projector
if which != "docs":
    for f in ("SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"):
        hf_hub_download("ggml-org/SmolVLM-256M-Instruct-GGUF", f, local_dir="models/vision")
# llama.cpp server (Windows CPU build). On other systems download the matching build from https://github.com/ggml-org/llama.cpp/releases
if not os.path.exists("runtime/llama/llama-server.exe") and os.name == "nt":
    tag = "b11249"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-cpu-x64.zip"
    zpath = "runtime/llama.zip"
    urllib.request.urlretrieve(url, zpath)
    zipfile.ZipFile(zpath).extractall("runtime/llama")
    os.remove(zpath)
print("done. OCR models download automatically on first use (RapidOCR); to pin PP-OCR ONNX files put them in ocr_models/.")
