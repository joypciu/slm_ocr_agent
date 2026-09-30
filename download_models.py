"""Fetch the model weights and the llama.cpp server this project runs on (nothing large is stored in git).
   python download_models.py            # everything (about 0.7 GB)
   python download_models.py docs       # text reasoner only
   python download_models.py bn         # only the Bengali OCR (Tesseract + Bengali data)
"""
import os, subprocess, sys, zipfile, urllib.request, json

which = sys.argv[1] if len(sys.argv) > 1 else "all"
os.makedirs("models/vision", exist_ok=True)
os.makedirs("runtime", exist_ok=True)
from huggingface_hub import hf_hub_download  # noqa: E402

if which != "bn":
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


def fetch_tesseract():
    """Bengali OCR: Tesseract 5 with the Bengali data. On Windows the official installer needs administrator rights, so it is unpacked instead of run:
    7-Zip's tiny standalone extractor unpacks 7-Zip, which unpacks the Tesseract installer. Everything stays inside runtime/tesseract."""
    root = "runtime/tesseract"
    if os.path.exists(f"{root}/tessdata_best/ben.traineddata") and (os.path.exists(f"{root}/tesseract.exe") or os.name != "nt"):
        return
    if os.name != "nt":
        print("Bengali OCR: install Tesseract and its Bengali data with your package manager (e.g. apt install tesseract-ocr tesseract-ocr-ben), then set TESSERACT_CMD / TESSDATA_DIR.")
        return
    import shutil
    os.makedirs("runtime/dl", exist_ok=True)
    get = lambda url, dest: urllib.request.urlretrieve(url, dest) if not os.path.exists(dest) else None
    get("https://www.7-zip.org/a/7zr.exe", "runtime/dl/7zr.exe")
    get("https://www.7-zip.org/a/7z2409-x64.exe", "runtime/dl/7z-setup.exe")
    subprocess.run([os.path.abspath("runtime/dl/7zr.exe"), "x", os.path.abspath("runtime/dl/7z-setup.exe"), "-o" + os.path.abspath("runtime/dl/7z"), "-y"], check=True, capture_output=True)
    inst = "tesseract-ocr-w64-setup-5.4.0.20240606.exe"
    get(f"https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/{inst}", f"runtime/dl/{inst}")
    subprocess.run([os.path.abspath("runtime/dl/7z/7z.exe"), "x", os.path.abspath(f"runtime/dl/{inst}"), "-o" + os.path.abspath(root), "-y"], check=True, capture_output=True)
    os.makedirs(f"{root}/tessdata_best", exist_ok=True)
    urllib.request.urlretrieve("https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata", f"{root}/tessdata_best/ben.traineddata")
    shutil.copy(f"{root}/tessdata/eng.traineddata", f"{root}/tessdata_best/eng.traineddata")
    shutil.copytree(f"{root}/tessdata/configs", f"{root}/tessdata_best/configs", dirs_exist_ok=True)  # the tsv output mode lives here
    print("Bengali OCR ready in", root)


if which in ("all", "bn"):
    fetch_tesseract()
print("done. OCR models download automatically on first use (RapidOCR); to pin PP-OCR ONNX files put them in ocr_models/.")
