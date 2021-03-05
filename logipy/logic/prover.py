import os


class PropertyNotHoldsException(Exception):
    def __init__(self, property_text):
        message = "A property found not to hold:\n\t"
        message += property_text
        super().__init__(message)


def prove_set_of_properties(property_graphs, execution_graph):
    """A very simple and somewhat silly prover."""
    property_graphs = [p.get_copy() for p in property_graphs]  # Don't modify the original graphs.

    execution_graph.visualize("Execution Graph")

    always_proved_properties = []
    properties_to_prove = []
    for p in property_graphs:
        if p.is_uniform_timestamped():
            always_proved_properties.append(p)
        else:
            properties_to_prove.append(p)
    # In always proved properties, also add the proved parts of complex properties.
    for p in properties_to_prove:
        proved_part = p.get_present_time_subgraph()
        if proved_part and proved_part.is_implication_graph():
            _, proved_part_conclusion = proved_part.get_top_level_implication_subgraphs()
            always_proved_properties.append(proved_part)
            p.remove_subgraph(proved_part_conclusion)

    execution_graph = execution_graph.get_copy()

    # Try to apply all always proved properties, until no more property can be applied.
    more_to_be_applied = True
    properties_applied = 0
    while more_to_be_applied and properties_applied < 2:
        # TODO: Do it only one time
        # TODO: Sort always proved properties based on the complexity of p.
        more_to_be_applied = False
        for p in always_proved_properties:
            assumption, conclusion = p.get_top_level_implication_subgraphs()
            if execution_graph.contains_property_graph(assumption):
                execution_graph.replace_subgraph(assumption, conclusion)
                more_to_be_applied = True
                properties_applied += 1
                # p.visualize("Property Applied")
                # execution_graph.visualize("New Execution Graph")

    # Check that its not possible to prove the negation of all the rest properties.
    for p in properties_to_prove:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        conclusion = conclusion.get_copy()
        conclusion.logical_not()
        if execution_graph.contains_property_graph(assumption) and \
                execution_graph.contains_property_graph(conclusion):
            raise PropertyNotHoldsException(p.get_property_textual_representation())

    # # Visualization implementation.
    # base_path = "./test_runs/{}/".format(prove_set_of_properties.exported_counter)
    # property_id = 0
    # for p in properties_graphs:
    #     dir_path = base_path + "properties/"
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
    #     p.export_to_graphml_file(dir_path+str(property_id)+".graphml")
    #     property_id += 1
    # execution_graph.export_to_graphml_file(base_path+"execution_graph.graphml")
    # prove_set_of_properties.exported_counter += 1


prove_set_of_properties.exported_counter = 0
