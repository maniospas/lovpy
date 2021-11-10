from lovpy.importer.gherkin_importer import convert_specification_to_graph


def get_counter_sample_properties():
    rule1 = """WHEN returned by __add__
               THEN NOT is_counter
            """.replace('\n', '')
    rule1 = ' '.join(rule1.split())

    rule2 = """WHEN returned by range
               THEN is_iterated
               AND is_counter
            """.replace('\n', '')
    rule2 = ' '.join(rule2.split())

    spec1 = convert_specification_to_graph(rule1)
    spec2 = convert_specification_to_graph(rule2)
    return [spec1, spec2]


def get_threading_sample_properties():
    rule1 = """
        WHEN call acquire
        THEN SHOULD NOT locked
        AND locked
        AND PRINT locked [VAR.lovpy_value()]
    """.replace('\n', '')
    rule1 = ' '.join(rule1.split())

    rule2 = """
        GIVEN locked
        WHEN call release
        THEN PRINT released by [METHOD]
        AND NOT locked
    """.replace('\n', '')
    rule2 = ' '.join(rule2.split())

    spec1 = convert_specification_to_graph(rule1)
    spec2 = convert_specification_to_graph(rule2)
    return [spec1, spec2]
