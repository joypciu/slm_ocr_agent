"""Bijoy (legacy Bengali font) -> Unicode conversion. Sample lines come from a bank loan application form whose PDF text layer is Bijoy-encoded.
Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni import legacy_bn as L  # noqa: E402
from omni.index import toks  # noqa: E402

PAIRS = [
    ("cÖwZôv‡bi bvg", "প্রতিষ্ঠানের নাম"),                       # pre-base e-kar moved after its cluster; ra-phala; conjunct
    ("†UªW jvB‡mÝ", "ট্রেড লাইসেন্স"),                          # e-kar attaches to the ট্র cluster, not to the next letter
    ("e¨emvq wewb‡qvMK…Z g~jab", "ব্যবসায় বিনিয়োগকৃত মূলধন"),  # ya-phala, ri-kar, uu-kar, o-kar (ে + া)
    ("cÖwZôv‡bi evwl©K weµq (cÖ‡hvR¨ †ÿ‡Î)", "প্রতিষ্ঠানের বার্ষিক বিক্রয় (প্রযোজ্য ক্ষেত্রে)"),  # reph moves in front of its cluster
    ("Av‡e`bKvixi mv‡_ m¤úK©", "আবেদনকারীর সাথে সম্পর্ক"),
    ("wmGgGmGgB D‡`¨v³v", "সিএমএসএমই উদ্যোক্তা"),
    ("b¤^i", "নম্বর"),
    ("¯’vqx m¤ú`", "স্থায়ী সম্পদ"),
]


class LegacyBengali(unittest.TestCase):
    def test_known_lines(self):
        for legacy, expected in PAIRS:
            self.assertEqual(L.convert(legacy), expected, legacy)

    def test_detects_bijoy_page(self):
        page = " ".join(p for p, _ in PAIRS) * 2
        self.assertTrue(L.is_legacy(page))

    def test_leaves_other_text_alone(self):
        for text in ("Loan Amount (BDT): 4,250,000  Date of Birth: 11/03/1992", "Café résumé naïve über Ärger façade — déjà vu, señor", "প্রতিষ্ঠানের নাম (already Unicode Bengali)"):
            self.assertFalse(L.is_legacy(text), text)

    def test_urls_and_numbers_survive_conversion(self):
        self.assertEqual(L.convert("www.ucb.com.bd"), "www.ucb.com.bd")
        self.assertEqual(L.convert("1.10 2026"), "1.10 2026")

    def test_no_unknown_glyphs_on_sample(self):
        self.assertEqual(L.unknown_glyphs(" ".join(p for p, _ in PAIRS)), [])


class BengaliSearchTokens(unittest.TestCase):
    def test_bengali_and_ascii_digits_match(self):
        self.assertEqual(toks("১.৭ আয়"), toks("1.7 আয়"))

    def test_precomposed_and_decomposed_letters_match(self):
        self.assertEqual(toks("আয়"), toks("আয়"))  # য় as য + nukta vs the precomposed letter


if __name__ == "__main__":
    unittest.main()
