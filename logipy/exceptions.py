class PropertyNotHoldsException(Exception):
    def __init__(self, property_text):
        message = "A property found not to hold:\n\t"
        message += property_text
        super().__init__(message)


class ModelNotFoundException(Exception):
    pass
