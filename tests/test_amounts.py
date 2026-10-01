"""Receipt/invoice row reading. Rows are typical OCR output of public sample receipts (typos included).
Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni import amounts as A  # noqa: E402

R1 = "TRAD KY TOAST CARTE 28.182\nITEMS 1.00\nSUBTTL 28.182\nPB-1 10% 2.818\nTOTAL 31.000\nCASH 31.000"
R2 = "ISMINE MT (L) 24,000\nCOCONUT JELLY (L) 4,000\nSUB TOTAI 28,000\nTOTAL SALES 28.000\nTOTAL ITEMS\nCASH 100,000"
R3 = "Y.B.BAT 46000\nTOTAL\n91000\nCASH 91000"


class Rows(unittest.TestCase):
    def val(self, q, text):
        r = A.answer(q, text)
        return A.digits(r[0]) if r else None

    def test_total_subtotal_tax(self):
        self.assertEqual(self.val("What is the total amount?", R1), "31000")
        self.assertEqual(self.val("What is the subtotal?", R1), "28182")
        self.assertEqual(self.val("What is the tax amount?", R1), "2818")

    def test_typos_in_labels(self):
        self.assertEqual(self.val("What is the subtotal?", R2), "28000")

    def test_value_on_the_next_row(self):
        self.assertEqual(self.val("What is the total amount?", R3), "91000")

    def test_price_of_item_uses_its_own_row(self):
        self.assertEqual(self.val("What is the price of Y.B.BAT?", R3), "46000")
        self.assertEqual(self.val("What is the price of JASMINE MT ( L )?", R2.replace("ISMINE", "JASMINE")), "24000")

    def test_unknown_or_open_questions_fall_through(self):
        self.assertIsNone(A.answer("What is the total loan amount requested by the applicant?", R1))
        self.assertIsNone(A.answer("What is the price of caviar?", R1))
        self.assertIsNone(A.answer("What is the tax amount?", "TOTAL 100"))


if __name__ == "__main__":
    unittest.main()
