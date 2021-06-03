import unittest

from logipy.graphs.timestamps import *
from logipy.monitor.time_source import get_zero_locked_timesource


class TestTimestampSequenceMatches(unittest.TestCase):

    def test_positive_with_two_absolute_sequences(self):
        seq1 = [Timestamp(4), Timestamp(7), Timestamp(12), Timestamp(32)]
        seq2 = [Timestamp(4), Timestamp(7), Timestamp(12), Timestamp(32)]

        self.assertTrue(timestamp_sequences_matches(seq1, seq2))

    def test_negative_with_two_absolute_sequences(self):
        seq1 = [Timestamp(4), Timestamp(7), Timestamp(12), Timestamp(32)]
        seq2 = [Timestamp(4), Timestamp(7), Timestamp(11), Timestamp(32)]

        self.assertFalse(timestamp_sequences_matches(seq1, seq2))

    def test_positive_with_relative_and_absolute_sequence(self):
        timesource = get_zero_locked_timesource()
        seq1 = [LesserThanRelativeTimestamp(-1, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                RelativeTimestamp(0, timesource)]
        seq2 = [Timestamp(4), Timestamp(7), Timestamp(32), Timestamp(32)]

        self.assertTrue(timestamp_sequences_matches(seq1, seq2))

    def test_negative_with_relative_and_absolute_sequence(self):
        timesource = get_zero_locked_timesource()
        seq1 = [LesserThanRelativeTimestamp(-1, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                RelativeTimestamp(0, timesource)]
        seq2 = [Timestamp(4), Timestamp(7), Timestamp(31), Timestamp(32)]

        self.assertFalse(timestamp_sequences_matches(seq1, seq2))

    def test_positive_with_relative_sequences(self):
        timesource = get_zero_locked_timesource()
        seq1 = [LesserThanRelativeTimestamp(-2, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                RelativeTimestamp(0, timesource)]
        seq2 = [LesserThanRelativeTimestamp(-1, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                RelativeTimestamp(0, timesource)]

        self.assertTrue(timestamp_sequences_matches(seq1, seq2))

    def test_negative_with_relative_sequences(self):
        timesource = get_zero_locked_timesource()
        seq1 = [LesserThanRelativeTimestamp(-2, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                RelativeTimestamp(0, timesource)]
        seq2 = [LesserThanRelativeTimestamp(-1, timesource),
                LesserThanRelativeTimestamp(-1, timesource),
                RelativeTimestamp(0, timesource),
                GreaterThanRelativeTimestamp(1, timesource)]

        self.assertFalse(timestamp_sequences_matches(seq1, seq2))

    def test_equal_between_absolutes(self):
        t1 = Timestamp(32)
        t2 = Timestamp(32)
        t3 = Timestamp(44)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_equal_between_relatives(self):
        t1 = LesserThanRelativeTimestamp(-1)
        t2 = LesserThanRelativeTimestamp(-1)
        t3 = LesserThanRelativeTimestamp(-2)
        t4 = GreaterThanRelativeTimestamp(1)
        t5 = RelativeTimestamp(0)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(t1, t4)
        self.assertNotEqual(t1, t5)

    def test_equal_between_absolutes_relatives(self):
        t1 = Timestamp(22)
        t2 = LesserThanRelativeTimestamp(0)
        t3 = GreaterThanRelativeTimestamp(3)
        t4 = RelativeTimestamp(0)
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(t1, t4)

    def test_hashability(self):
        t1 = Timestamp(22)
        t2 = Timestamp(22)
        t3 = LesserThanRelativeTimestamp(-1)
        t4 = LesserThanRelativeTimestamp(-1)

        s = {t1, t3}
        self.assertEqual(len(s), 2)
        self.assertIn(t2, s)
        self.assertIn(t4, s)
