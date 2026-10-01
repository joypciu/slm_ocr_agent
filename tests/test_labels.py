"""Reading the value of a named form label. Rows/segments mimic OCR output of scanned business forms (typos included).
Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni import labels as L  # noqa: E402

B = (0, 0, 1, 0.02)


def page(*rows):
    """rows: lists of segments (text, x0, x1) -> (lines, segs)"""
    lines = [(" ".join(t for t, _, _ in r), (r[0][1], 0, r[-1][2], 0.02)) for r in rows]
    return lines, [[list(s) for s in r] for r in rows]


class Labels(unittest.TestCase):
    def test_value_ends_at_the_box_boundary(self):
        lines, segs = page([("cc: S. Willinger (3)", 0.10, 0.30), ("K. A. Hutchison/S. A. Howard", 0.35, 0.60)])
        self.assertEqual(L.read("What is the value of 'cc'?", lines, segs)[0], "S. Willinger (3)")

    def test_ocr_typo_in_a_quoted_label(self):
        lines, segs = page([("SUBMITTER'S SSf:407-64-3484", 0.01, 0.40), ("Status (1993)", 0.5, 0.7)])
        self.assertEqual(L.read("What is the value of 'SUBMITTER'S SS#'?", lines, segs)[0], "407-64-3484")

    def test_value_in_the_next_box_or_under_the_label(self):
        lines, segs = page([("Brand:", 0.02, 0.10), ("RALEIGH (BELAIR portion not tested)", 0.12, 0.50)])
        self.assertEqual(L.read("What is the value of 'Brand'?", lines, segs)[0], "RALEIGH (BELAIR portion not tested)")
        lines, segs = page([("TO:", 0.10, 0.15)], [("The Corporation Trust Company", 0.10, 0.45)])
        self.assertEqual(L.read("What is the value of 'TO'?", lines, segs)[0], "The Corporation Trust Company")

    def test_a_following_label_ends_the_value_without_boxes(self):
        lines = [("DATE: June 21, 1993DEPARTMENT:R&D Library", B)]
        self.assertEqual(L.read("What is the value of 'DATE'?", lines)[0], "June 21, 1993")
        self.assertEqual(L.read("What is the value of 'DEPARTMENT'?", lines)[0], "R&D Library")

    def test_unquoted_label_must_match_exactly(self):
        lines = [("Booking Branch: North Gulshan", B), ("Branch Code: 0123", B)]
        self.assertEqual(L.read("What is the booking branch?", lines)[0], "North Gulshan")
        self.assertIsNone(L.read("What is the branch?", lines))          # no exact label 'branch'
        self.assertIsNone(L.read("What is the name?", [("Name: Karim", B)]))  # too vague without quotes

    def test_uncertain_cases_fall_through(self):
        self.assertIsNone(L.read("What is the value of 'Gender'?", [("Gender: ☐ Male ☑ Female", B)]))   # tick boxes
        two = [("Date: 1/2/90", B), ("Date: 5/6/91", B)]
        self.assertIsNone(L.read("What is the value of 'Date'?", two))   # two different values
        self.assertIsNone(L.read("Who signed the proposal?", [("Signed: X", B)]))


if __name__ == "__main__":
    unittest.main()
