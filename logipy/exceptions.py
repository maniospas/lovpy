class PropertyNotHoldsException(Exception):
    def __init__(self, property_text, last_proved_stacktrace):
        self.last_proved_stacktrace = last_proved_stacktrace
        message = "A property found not to hold:\n\t"
        message += property_text
        super().__init__(message)


class ModelNotFoundException(Exception):
    pass
