"""Cold-OCR speed vs fidelity for RapidOCR settings. Fidelity = text similarity to the default engine's output on the same page.
   python ocr_speed.py"""
import difflib, glob, os, time
import numpy as np
from PIL import Image
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec

OCR_DIR = "ocr_models"
det, rec = os.path.join(OCR_DIR, "ch_PP-OCRv5_det_mobile.onnx"), os.path.join(OCR_DIR, "PP-OCRv6_rec_small.onnx")
base = {"Det.model_path": det, "Rec.model_path": rec, "Rec.lang_type": LangRec.LATIN}
CONFIGS = {
    "default": {},
    "no-cls": {"Global.use_cls": False},
    "no-cls+side960": {"Global.use_cls": False, "Det.limit_side_len": 960},
    "no-cls+side640": {"Global.use_cls": False, "Det.limit_side_len": 640},
    "no-cls+threads6": {"Global.use_cls": False, "EngineConfig.onnxruntime.intra_op_num_threads": 6, "EngineConfig.onnxruntime.inter_op_num_threads": 1},
}
files = sorted(glob.glob("data/varied_fresh/cord/*.png"))[:6] + sorted(glob.glob("data/varied_fresh/invoice/*.png"))[:4] + sorted(glob.glob("data/varied_fresh/funsd/*.png"))[:4]
imgs = [np.array(Image.open(f).convert("RGB")) for f in files]


def text(r):
    return " ".join(r.txts or [])


ref, rows = None, []
for name, extra in CONFIGS.items():
    eng = RapidOCR(params={**base, **extra})
    eng(imgs[0])  # warm-up
    t0, outs = time.time(), []
    for im in imgs:
        outs.append(text(eng(im)))
    dt = (time.time() - t0) / len(imgs)
    if ref is None:
        ref = outs
    sim = np.mean([difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ref, outs)])
    print(f"{name:18s} {dt:5.2f}s/page   similarity to default {sim:.3f}", flush=True)
