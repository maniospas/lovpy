from copy import copy
import logipy.monitor.time_source as time_source


PLUS_INFINITE = "inf"
MINUS_INFINITE = "-inf"


class Timestamp:
    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return str(self._value)

    def __lt__(self, other):
        return self.get_validity_interval()[1] < other.get_validity_interval()[1]

    def __le__(self, other):
        return self.get_validity_interval()[1] <= other.get_validity_interval()[1]

    def __gt__(self, other):
        return self.get_validity_interval()[1] > other.get_validity_interval()[1]

    def __ge__(self, other):
        return self.get_validity_interval()[1] >= other.get_validity_interval()[1]

    def __copy__(self):
        return type(self)(self._value)

    def is_absolute(self):
        return type(self) is Timestamp

    def get_absolute_value(self):
        return self._value

    def get_validity_interval(self):
        return [self._value, self._value]

    def get_shifted_timestamp(self, shift):
        return Timestamp(self.get_absolute_value()+shift)

    def matches(self, other):
        # Check if intervals overlap.
        a = self.get_validity_interval()
        b = other.get_validity_interval()

        # WARNING: [..,-inf] and [inf,..] cases are not supported yet.
        if a[0] == MINUS_INFINITE and b[0] != MINUS_INFINITE:
            a[0] = b[0] - 1
        if b[0] == MINUS_INFINITE and a[0] != MINUS_INFINITE:
            b[0] = a[0] - 1
        if a[1] == PLUS_INFINITE and b[1] != PLUS_INFINITE:
            a[1] = b[1] + 1
        if b[1] == PLUS_INFINITE and a[1] != PLUS_INFINITE:
            b[1] = a[1] + 1

        # Cover the cases where both lower or upper limits are -inf or inf respectively.
        if a[0] == MINUS_INFINITE and b[0] == MINUS_INFINITE:
            if a[1] != MINUS_INFINITE and b[1] != MINUS_INFINITE:
                return True
            else:
                return False
        if a[1] == PLUS_INFINITE and b[1] == PLUS_INFINITE:
            if a[0] != PLUS_INFINITE and b[0] != PLUS_INFINITE:
                return True
            else:
                return False

        return max(a[0], b[0]) <= min(a[1], b[1])


class RelativeTimestamp(Timestamp):
    def __init__(self, value, time_source=None):
        super().__init__(value)
        self.time_source = time_source

    def __repr__(self):
        return "::" + str(self.get_relative_value())

    def __copy__(self):
        return type(self)(super().get_absolute_value(), time_source=self.time_source)

    def get_absolute_value(self):
        return self.time_source.get_current_time() + self.get_relative_value()

    def get_relative_value(self):
        return super().get_absolute_value()

    def get_validity_interval(self):
        absolute = self.get_absolute_value()
        return [absolute, absolute]

    def set_time_source(self, time_source):
        self.time_source = time_source

    def get_time_source(self):
        return self.time_source

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.get_relative_value() != other.get_relative_value():
            return False
        return True


class LesserThanRelativeTimestamp(RelativeTimestamp):
    def __repr__(self):
        return "<= " + RelativeTimestamp.__repr__(self)

    def get_validity_interval(self):
        return [MINUS_INFINITE, self.get_absolute_value()]


class GreaterThanRelativeTimestamp(RelativeTimestamp):
    def __repr__(self):
        return ">= " + RelativeTimestamp.__repr__(self)

    def get_validity_interval(self):
        return [self.get_absolute_value(), PLUS_INFINITE]


def timestamp_sequences_matches(seq1, seq2):
    """Check if two timestamp sequences matches."""
    if len(seq1) != len(seq2):
        raise RuntimeError("Timestamp sequences lengths should match.")

    # Align sequences based on the first pair of absolute and relative occurence.
    shift = 0

    for i in reversed(range(len(seq1))):
        t1 = seq1[i]
        t2 = seq2[i]

        if t1.is_absolute() and not t2.is_absolute():
            shift = t2.get_relative_value() - t1.get_absolute_value()
            break
        elif t2.is_absolute() and not t1.is_absolute():
            shift = t1.get_relative_value() - t2.get_absolute_value()
            break

    matches = True
    for i in range(len(seq1)):
        t1 = seq1[i]
        t2 = seq2[i]

        if t1.is_absolute():
            t1 = t1.get_shifted_timestamp(shift)
        else:
            t1 = copy(t1)
            t1.set_time_source(time_source.get_zero_locked_timesource())

        if t2.is_absolute():
            t2 = t2.get_shifted_timestamp(shift)
        else:
            t2 = copy(t2)
            t2.set_time_source(time_source.get_zero_locked_timesource())

        if not t1.matches(t2):
            matches = False
            break

    return matches
