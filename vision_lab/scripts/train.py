import json, random, time, torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText
from common import *
torch.set_num_threads(6); torch.manual_seed(0); random.seed(0)
EPOCHS, ACC, LR = 2, 4, 2e-4
proc = load_proc("models/base"); model = AutoModelForImageTextToText.from_pretrained("models/base", torch_dtype=torch.float32)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg); model.print_trainable_parameters(); model.train()
data = json.load(open("data/train.json"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
total = EPOCHS * len(data) // ACC; step = 0; t0 = time.time(); run = []
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 5) * max(0.05, 1 - s / total))
def batch(ex):
    prompt, ans = build(proc, ex["q"], ex["a"])
    inp = proc(text=prompt + ans, images=[Image.open(ex["image"])], return_tensors="pt")
    plen = proc(text=prompt, images=[Image.open(ex["image"])], return_tensors="pt")["input_ids"].shape[1]
    lab = inp["input_ids"].clone(); lab[:, :plen] = -100
    return inp, lab
for ep in range(EPOCHS):
    random.shuffle(data)
    for i, ex in enumerate(data):
        inp, lab = batch(ex)
        loss = model(**inp, labels=lab).loss / ACC; loss.backward(); run.append(loss.item() * ACC)
        if (i + 1) % ACC == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 50 == 0: model.save_pretrained("models/lora_ckpt")
            if step % 10 == 0: print(f"step {step}/{total} loss {sum(run[-40:])/len(run[-40:]):.3f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
model = model.merge_and_unload(); model.save_pretrained("models/tuned"); proc.save_pretrained("models/tuned")
print("DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)
