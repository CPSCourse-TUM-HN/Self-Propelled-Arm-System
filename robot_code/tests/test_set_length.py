from __future__ import print_function

import unittest

from set_length import set_square_lengths


class SetLengthTest(unittest.TestCase):
    def test_updates_all_routine_two_and_three_drive_segments(self):
        data = {
            "routines": [
                {"id": 2, "description": "old", "steps": [
                    {"motion": "drive"}, {"motion": "turn"}, {"motion": "drive"},
                    {"motion": "drive"}, {"motion": "drive"}, {"motion": "drive"},
                ]},
                {"id": 3, "steps": [
                    {"motion": "drive"}, {"motion": "turn"}, {"motion": "drive"},
                    {"motion": "drive"}, {"motion": "drive"}, {"motion": "drive"},
                    {"motion": "drive"}, {"motion": "drive"},
                ]},
            ]
        }

        set_square_lengths(data, 40)

        routine2 = [r for r in data["routines"] if r["id"] == 2][0]
        routine3 = [r for r in data["routines"] if r["id"] == 3][0]
        self.assertEqual(
            [s["distance_m"] for s in routine2["steps"] if s["motion"] == "drive"],
            [0.2, 0.4, 0.4, 0.4, 0.2],
        )
        self.assertEqual(
            [s["distance_m"] for s in routine3["steps"] if s["motion"] == "drive"],
            [0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2],
        )

    def test_rejects_non_positive_length(self):
        with self.assertRaises(ValueError):
            set_square_lengths({"routines": []}, 0)


if __name__ == "__main__":
    unittest.main()
