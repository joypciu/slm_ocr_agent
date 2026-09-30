"""Synthetic benchmark for the vision sub-agents (the name pool was changed after the reported runs: cosmetic, same layouts and ticks): labelled cases are generated, so tuning has real numbers to work with.
   python vision_bench.py gen                 # make the cases (deterministic)
   python vision_bench.py run <task> [variant] # task: checkbox | hand | photo
Splits: the first 60% of each task is 'dev' (tune on it), the last 40% is 'test' (report only)."""
import io, json, os, random, re, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = "data/vision_bench"
FONTS = "C:/Windows/Fonts/"
os.makedirs(OUT, exist_ok=True)

OPTION_SETS = [("Gender", ["Male", "Female"]), ("Gender", ["Male", "Female", "Other"]), ("Marital Status", ["Married", "Single", "Divorced"]),
               ("Residence Status", ["Own", "Family Owned", "Rented", "Govt"]), ("Customer", ["Yes", "No"]), ("Loan Type", ["Personal Loan", "Doctors Loan", "Auto Loan"]),
               ("Segment", ["Salaried", "Businessman", "Self-Employed"]), ("Education", ["SSC", "HSC", "Bachelor", "Masters"])]
FIRST = ["Tanvir", "Rashida", "Karim", "Nusrat", "Sumon", "Rina", "Imran", "Belal", "Lipi", "Jahanara", "Abdul", "Farid", "Sultana", "Rezaul"]
LAST = ["Talukder", "Miah", "Ahmed", "Khan", "Islam", "Rahman", "Akter", "Uddin", "Haque", "Begum", "Chowdhury", "Sarkar"]
PLACES = ["Barishal", "Gulshan", "Dhaka", "Khulna", "Sylhet", "Bogura", "Rajshahi", "Barguna", "Comilla", "Mymensingh"]
COLORS = {"red": (215, 30, 30), "green": (30, 150, 50), "blue": (35, 70, 200), "yellow": (240, 210, 40), "black": (15, 15, 15), "orange": (240, 130, 25), "purple": (130, 50, 170), "white": (250, 250, 250)}


def _finish(img, rng, blur=(0, 0.9), noise=6, rot=1.5):
    img = img.rotate(rng.uniform(-rot, rot), fillcolor=(240,) if img.mode == "L" else (240, 240, 240), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*blur)))
    a = np.array(img).astype(np.int16) + np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, noise, np.array(img).shape).astype(np.int16)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=rng.randint(60, 88))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make_row(rng, label, options, ticked):
    img = Image.new("L", (1150, 96), rng.randint(225, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONTS + "arial.ttf", rng.choice([26, 28, 30]))
    x = 20
    d.text((x, 30), label + ":", font=f, fill=20)
    x += d.textlength(label + ":", font=f) + 30
    w = lambda a=2: rng.uniform(-a, a)
    for i, opt in enumerate(options):
        s, y0 = rng.randint(22, 28), 34 + rng.randint(-2, 2)
        pts = [(x, y0), (x + s, y0), (x + s, y0 + s), (x, y0 + s), (x, y0)]
        d.line([(px + w(1), py + w(1)) for px, py in pts], fill=rng.randint(20, 70), width=rng.choice([2, 2, 3]))
        if i == ticked:
            ink, style = rng.randint(0, 60), rng.choice(["check", "cross", "fill", "check", "cross"])
            if style == "check":
                d.line([(px + w(), py + w()) for px, py in [(x + 3, y0 + s * .55), (x + s * .4, y0 + s - 2), (x + s + 6, y0 - 8)]], fill=ink, width=rng.choice([3, 4]))
            elif style == "cross":
                d.line([(x + 3, y0 + 3), (x + s - 3, y0 + s - 3)], fill=ink, width=3)
                d.line([(x + s - 3, y0 + 3), (x + 3, y0 + s - 3)], fill=ink, width=3)
            else:
                for k in range(6):
                    d.line([(x + 3, y0 + 3 + k * (s - 6) / 5), (x + s - 3, y0 + 5 + k * (s - 6) / 5 + w())], fill=ink, width=2)
        x += s + 10
        d.text((x, 30), opt, font=f, fill=20)
        x += d.textlength(opt, font=f) + 45
    return _finish(img, rng)


def make_hand(rng, text):
    font = ImageFont.truetype(FONTS + rng.choice(["Inkfree.ttf", "LHANDW.TTF", "BRADHITC.TTF"]), rng.randint(38, 54))
    img = Image.new("RGB", (60 + int(len(text) * 34), 96), (rng.randint(228, 252),) * 3)
    d = ImageDraw.Draw(img)
    x = 18
    for ch in text:
        d.text((x, 20 + rng.randint(-5, 5)), ch, font=font, fill=(rng.randint(0, 50), rng.randint(0, 50), rng.randint(30, 110)))
        x += d.textlength(ch, font=font) + rng.uniform(-1, 3)
    return _finish(img, rng, blur=(0.3, 1.1), noise=7, rot=2.5)


def make_shapes(rng, kind):
    bg_name = rng.choice(list(COLORS))
    fg_name = rng.choice([c for c in COLORS if c != bg_name and abs(sum(COLORS[c]) - sum(COLORS[bg_name])) > 120])
    img = Image.new("RGB", (640, 480), COLORS[bg_name])
    d = ImageDraw.Draw(img)
    shape = rng.choice(["circle", "square", "triangle"])

    def draw(cx, cy, r):
        if shape == "circle":
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=COLORS[fg_name])
        elif shape == "square":
            d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=COLORS[fg_name])
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[fg_name])

    if kind == "color":
        draw(rng.randint(200, 440), rng.randint(160, 320), rng.randint(60, 100))
        return img, f"What color is the {shape}?", fg_name
    if kind == "count":
        n = rng.randint(1, 4)
        xs = sorted(rng.sample(range(90, 560, 110), n))
        for cx in xs:
            draw(cx, rng.randint(180, 300), 42)
        return img, f"How many {shape}s are in the image?", str(n)
    side = rng.choice(["left", "right"])
    draw(rng.randint(90, 200) if side == "left" else rng.randint(440, 550), rng.randint(160, 320), 60)
    return img, f"Is the {shape} on the left side or the right side of the image?", side


def gen(n_check=60, n_hand=40, n_photo=45):
    rng = random.Random(2026)
    cases = {"checkbox": [], "hand": [], "photo": []}
    for i in range(n_check):
        label, opts = rng.choice(OPTION_SETS)
        t = rng.randrange(len(opts))
        fn = f"{OUT}/cb_{i:03d}.jpg"
        make_row(rng, label, opts, t).save(fn, quality=95)
        cases["checkbox"].append({"file": fn, "row_text": f"{label}: " + " ".join(opts), "options": opts, "truth": opts[t]})
    for i in range(n_hand):
        kind = ["date", "amount", "name", "place"][i % 4]
        if kind == "date":
            txt = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.choice([1975, 1988, 1992, 2024, 2025, 2026])}"
        elif kind == "amount":
            txt = str(rng.choice([rng.randint(10000, 99999), rng.randint(100000, 9999999)]))
        elif kind == "name":
            txt = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        else:
            txt = rng.choice(PLACES)
        fn = f"{OUT}/hw_{i:03d}.jpg"
        make_hand(rng, txt).save(fn, quality=95)
        cases["hand"].append({"file": fn, "kind": kind, "truth": txt})
    for i in range(n_photo):
        kind = ["color", "count", "side"][i % 3]
        img, q, a = make_shapes(rng, kind)
        fn = f"{OUT}/ph_{i:03d}.jpg"
        img.save(fn, quality=92)
        cases["photo"].append({"file": fn, "q": q, "truth": a, "kind": kind})
    json.dump(cases, open(f"{OUT}/cases.json", "w"), indent=1)
    print({k: len(v) for k, v in cases.items()})


def split(items):
    k = int(len(items) * 0.6)
    return items[:k], items[k:]


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
