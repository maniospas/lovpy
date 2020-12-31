global_properties = list()  # Storage for graph properties that apply everywhere.


class LogipyPropertyException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_global_properties():
    return global_properties


def add_global_property(property_graph):
    global_properties.append(property_graph)


# All code below, is deprecated.
def empty_properties():
    return set()


def combine(property_set1, property_set2):
    """Extends the first set with the properties contained in second set."""
    for property in property_set2:
        property_set1.add(property)


def has_property(property_set, property):
    """"Checks whether given property set has the given property."""
    positive = True
    if property.startswith("NOT "):
        property = property[4:]
        positive = False
    if property == "TRUE":
        return True
    if property == "FALSE":
        return False
    return (property in property_set) == positive


def add_property(property_set, given_rules, properties):
    if given_rules is not None:
        for rule in given_rules.split(" AND "):
            if not has_property(property_set, rule):
                return

    for property in properties.split(" AND "):
        if property.startswith("SHOULD "):
            # For SHOULD it's enough to check that property already belongs to given set.
            if not has_property(property_set, property[len("SHOULD "):]):
                raise LogipyPropertyException(property)
        elif property.startswith("NOT "):
            # Adding a property preceded by NOT means removing it from the set.
            if property[4:] in property_set:
                property_set.remove(property[4:])
        elif property.startswith("ERROR"):
            raise LogipyPropertyException(property[5:])
        elif property.startswith("PRINT"):
            print(property[5:])
        else:
            property_set.add(property)
