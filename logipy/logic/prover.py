import os


class PropertyNotHoldsException(Exception):
    def __init__(self, property_text):
        message = "A property found not to hold:\n\t"
        message += property_text
        super().__init__(message)


def prove_set_of_properties(property_graphs, execution_graph):
    """A very simple and somewhat silly prover."""
    always_proved_properties = []
    properties_to_prove = []
    for p in property_graphs:
        if p.is_uniform_timestamped():
            always_proved_properties.append(p)
        else:
            properties_to_prove.append(p)

    execution_graph = execution_graph.get_copy()

    # Try to apply all always proved properties.
    # TODO: Make it work for properties that contain an always proved part and a part that
    # should be proved.
    for p in always_proved_properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        if execution_graph.contains_property_graph(assumption):
            execution_graph.replace_subgraph(assumption, conclusion)

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
