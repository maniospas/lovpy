import logging

import logipy.logic.properties as properties


quantifiers = dict()
rules = dict()
rules["call"] = dict()
rules["called by"] = dict()
rules["returned by"] = dict()


def add_rule(text):
    """Adds given rule to the global rules monitor."""

    # Find position of Gherkin step keywords.
    text = " " + text + " "
    given_pos = text.find(" GIVEN ")
    when_pos = text.find(" WHEN ")
    then_pos = text.find(" THEN ")
    if when_pos == -1 or then_pos == -1:
        raise Exception(
            "WHEN and THEN are mandatory in Gherkin syntax (syntax is case sensitive): "+text)

    # Assign GIVEN rules and THEN graph_logic to monitors separated by WHEN rules.
    given_rules = None if given_pos == -1 else text[given_pos+7:when_pos]
    when_text = text[when_pos+6:then_pos]
    then_text = text[then_pos+6:].strip()
    found_rule_type = None
    for rule_type, rule_dict in rules.items():
        # Check whether WHEN rule matches a defined one.
        if when_text.startswith(rule_type+" "):
            found_rule_type = rule_type
            # Create a monitor for the corresponding method under found rule.
            method_name = when_text[len(rule_type+" "):]
            if method_name not in rule_dict:
                rule_dict[method_name] = list()
            rule_dict[method_name].append({"given": given_rules, "property": then_text})

    if found_rule_type is None:
        raise Exception("Invalid WHEN timing: "+when_text)


def evalOrNan(expression, vars):
    try:
        return str(eval(expression, globals(), vars))
    except Exception as e:
        return str(e)


def apply_method_rules(method_name, logipy_primitive_instance, rule_type, args, kwargs):
    _apply_method_base_rules(method_name, logipy_primitive_instance, rule_type, args, kwargs)


def _apply_method_base_rules(method_name, logipy_primitive_instance, rule_type, args, kwargs):

    rule_dict = rules[rule_type]
    if method_name in rule_dict:
        # Apply all rules defined for the given method.
        for rule in rule_dict[method_name]:
            property = _transform_property(rule["property"], method_name, logipy_primitive_instance,
                                           args, kwargs)
            given = _transform_property(rule["given"], method_name, logipy_primitive_instance,
                                        args, kwargs)
            properties.add_property(logipy_primitive_instance.get_logipy_properties(), given, property)


def _transform_property(property, method_name, logipy_primitive_instance, args, kwargs):
    if property is None:
        return None
    evaluated_property = ""
    expression = ""
    level = 0
    for c in property:
        if level == 0:
            evaluated_property += c
        elif c != "]":
            expression += c
        if c == "[":
            level += 1
        if c == "]":
            level -= 1
            if level < 0:
                raise properties.LogipyPropertyException("Malformed property: " + property)
            if level == 0:
                evaluated_property += evalOrNan(expression, {"VAR": logipy_primitive_instance, "ARGS": args, "KWARGS": kwargs, "METHOD": method_name}) + "]"
    logging.debug("Eval expression: " + evaluated_property)

    if level != 0:
        raise properties.LogipyPropertyException("Malformed property: " + property)
    return evaluated_property





