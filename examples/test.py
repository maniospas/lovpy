from logipy.wrappers import LogipyPrimitive, logipy_call
class ValueContainer:
    def __init__(self, x):
        self.x = x

    def increase(self):
        self.x = self.x + 10


def print_counter(i):
    logipy_call(print,logipy_call(str,i), ' iteration')


for i in logipy_call(range,10):
    value = logipy_call(ValueContainer,i)
    logipy_call(value.increase,)
    logipy_call(print,logipy_call(repr,value.x))
    logipy_call(print_counter,value.x)
