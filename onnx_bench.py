"""Speed of Qwen3-0.6B on ONNX Runtime (CPU) vs the llama.cpp GGUF: prompt processing and generation, same prompt length.
   python onnx_bench.py [model.onnx]   (model from onnx-community/Qwen3-0.6B-ONNX, files kept under models/onnx)"""
import json, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "models/onnx/onnx/model_q4.onnx"
cfg = json.load(open("models/onnx/config.json"))
tok = Tokenizer.from_file("models/onnx/tokenizer.json")
so = ort.SessionOptions()
so.intra_op_num_threads = 6
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
t0 = time.time()
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
print(f"load {time.time() - t0:.1f}s inputs: {[i.name for i in sess.get_inputs()][:5]}...")
L, H, D = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
kv_names = [i.name for i in sess.get_inputs() if "past_key_values" in i.name]
kv_type = np.float16 if "float16" in sess.get_inputs()[-1].type else np.float32


def run(ids, past, pos0):
    n = len(ids)
    feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, pos0 + n), dtype=np.int64),
            "position_ids": np.arange(pos0, pos0 + n, dtype=np.int64)[None]}
    feed.update(past)
    out = sess.run(None, feed)
    names = [o.name for o in sess.get_outputs()]
    new = {n.replace("present", "past_key_values"): v for n, v in zip(names, out) if n.startswith("present")}
    return out[0], new


text = "The quick brown fox jumps over the lazy dog. " * 60
ids = tok.encode(text).ids[:512]
past = {n: np.zeros((1, H, 0, D), dtype=kv_type) for n in kv_names}
t = time.time()
logits, past = run(ids, past, 0)
pp = time.time() - t
print(f"prompt {len(ids)} tokens: {pp:.2f}s = {len(ids) / pp:.0f} t/s")
nxt = int(np.argmax(logits[0, -1]))
t = time.time()
N = 48
for k in range(N):
    logits, past = run([nxt], past, len(ids) + k)
    nxt = int(np.argmax(logits[0, -1]))
g = time.time() - t
print(f"generate {N} tokens: {g:.2f}s = {N / g:.1f} t/s")
