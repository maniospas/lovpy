import os


def prove_set_of_properties(property_graphs, execution_graph):
    """A very simple and somewhat silly prover."""
    always_proved_properties = []
    for i in range(len(property_graphs)):
        if property_graphs[i].is_uniform_timestamped():
            always_proved_properties.append(property_graphs.remove(i))

    execution_graph = execution_graph.get_copy()

    # Try to apply all always proved properties.
    for p in always_proved_properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        if execution_graph.contains_property_graph(assumption):
            execution_graph.replace(assumption, conclusion)

    # Check that its not possible to prove the negation of all the rest properties.
    for p in property_graphs:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        conclusion = conclusion.get_copy()
        conclusion.logical_not()
        if execution_graph.contains_property_graph(assumption) and \
                execution_graph.contains_property_graph(conclusion):
            raise Exception("Property found to not hold.")

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
