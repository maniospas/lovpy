from unittest import TestCase

from lovpy.monitor.wrappers import LogipyPrimitive


class TestLogipyPrimitive(TestCase):

    def test_repr(self) -> None:
        class MyClass:
            def __repr__(self):
                return "my_class_repr"

        obj = MyClass()

        self.assertEqual(repr(LogipyPrimitive(obj)), repr(obj))
