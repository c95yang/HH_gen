import unittest

from tridi.utils.interaction import parse_action_from_seq, parse_action_from_video


class TestInteractionParser(unittest.TestCase):
    def test_parse_action_from_seq_examples(self):
        self.assertEqual(parse_action_from_seq("s03_Hit_7"), "Hit")
        self.assertEqual(parse_action_from_seq("s04_Handshake_8"), "Handshake")
        self.assertEqual(parse_action_from_seq("s03_HoldingHands_6"), "HoldingHands")
        self.assertEqual(parse_action_from_seq("s03_Posing_7"), "Posing")
        self.assertEqual(parse_action_from_seq("s03_Push_8"), "Push")

    def test_parse_action_from_video(self):
        self.assertEqual(parse_action_from_video("Grab 16.mp4"), "Grab")
        self.assertEqual(parse_action_from_video("Handshake_3.MP4"), "Handshake")


if __name__ == "__main__":
    unittest.main()
