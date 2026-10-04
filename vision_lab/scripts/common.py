import re, torch
from PIL import Image
from transformers import AutoProcessor
def load_proc(path):
    p = AutoProcessor.from_pretrained(path, do_image_splitting=False, size={"longest_edge": 512})
    return p
def build(proc, q, a=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q + "\nAnswer briefly."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return prompt if a is None else (prompt, a + "<end_of_utterance>")
def norm(s):
    s = s.lower().strip().rstrip("."); s = re.sub(r"[^\w\s\.%$]", "", s); return re.sub(r"\s+", " ", s).strip()
def score(pred, gold):
    p, g = norm(pred), norm(gold)
    return 1.0 if (p == g or (len(g) > 1 and g in p.split()) or (len(g) > 2 and g in p)) else 0.0
