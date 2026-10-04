import time, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
torch.set_num_threads(6)
MID = "HuggingFaceTB/SmolVLM-256M-Instruct"
proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=torch.float32).eval()
model.save_pretrained("models/base"); proc.save_pretrained("models/base")
print("params(M):", sum(p.numel() for p in model.parameters())/1e6)
img = Image.new("RGB", (512, 256), "white"); d = ImageDraw.Draw(img)
d.text((20, 40), "INVOICE #4821  Total due: $1,250.00  Date: 2026-03-14", fill="black")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the total due?"}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
inp = proc(text=prompt, images=[img], return_tensors="pt")
t = time.time()
with torch.no_grad(): out = model.generate(**inp, max_new_tokens=32, do_sample=False)
dt = time.time() - t; n = out.shape[1] - inp["input_ids"].shape[1]
print(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
print(f"{n} tokens in {dt:.1f}s -> {n/dt:.1f} tok/s (incl. image encode)")
