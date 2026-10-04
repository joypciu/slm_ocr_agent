"""Follow-up questions resolved from the previous turn, and everything else left as asked.
Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni.followup import resolve  # noqa: E402


class FollowUps(unittest.TestCase):
    def r(self, q, prev):
        return resolve(q, prev)[0]

    def test_short_follow_ups_borrow_the_previous_frame(self):
        self.assertEqual(self.r("and the subtotal?", "What is the total amount?"), "What is the subtotal?")
        self.assertEqual(self.r("what about 'DATE'?", "What is the value of 'FAX'?"), "What is the value of 'DATE'?")
        self.assertEqual(self.r("invoice date?", "What is the invoice number?"), "What is the invoice date?")
        self.assertEqual(self.r("and the price of Kroket?", "What is the price of Arem Arem?"), "What is the price of Kroket?")

    def test_pronoun_takes_the_person_named_before(self):
        self.assertEqual(self.r("What is his phone number?", "What is Karim Rahman's address?"), "What is Karim Rahman's phone number?")
        self.assertEqual(self.r("and his salary?", "What is the loan amount?"), "What is the salary?")  # no name before: the form's person

    def test_everything_else_is_left_alone(self):
        for q in ("Who signed the proposal?", "Summarize this.", "Thanks!", "ok", "ঋণের পরিমাণ?"):
            self.assertEqual(self.r(q, "What is the total?"), q)
        self.assertEqual(self.r("and the subtotal?", None), "and the subtotal?")   # no previous turn


if __name__ == "__main__":
    unittest.main()
