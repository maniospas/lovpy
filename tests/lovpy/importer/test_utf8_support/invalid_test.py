class ValueContainer:
    def __init__(self, x):
        self.x = x

    def increase(self):
        self.x = self.x + 10


def print_counter(i):
    print(str(i), ' iteration')


for i in range(10):
    value = ValueContainer(i)
    value.increase()
    print_counter(value.x)
