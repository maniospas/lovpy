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

    def get_absolute_value(self):
        return self._value

    def get_validity_interval(self):
        return [self._value, self._value]

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

    def get_absolute_value(self):
        return self.time_source.get_current_time() + self.get_relative_value()

    def get_relative_value(self):
        return super().get_absolute_value()

    def get_validity_interval(self):
        absolute = self.get_absolute_value()
        return [absolute, absolute]

    def set_time_source(self, time_source):
        self.time_source = time_source


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
