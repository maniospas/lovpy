from lovpy.graphs.timed_property_graph import NoPositiveAndNegativePredicatesSimultaneously
from lovpy.exceptions import PropertyNotHoldsException
from .next_theorem_selectors import get_default_theorem_selector
from .properties import split_into_theorems_and_properties_to_prove, \
    negate_conclusion_part_of_properties

PROVE_IF_FULLY_REDUCED = True  # A property is proved only if reduced graph perfectly matches it.
MAX_PROOF_PATH = 10  # Max number of theorems to be applied in order to prove a property.

full_visualization_enabled = False
prover_invocations = 0


def prove_set_of_properties(property_graphs, execution_graph, theorem_selector=None):
    """Tries to prove that given set of properties hold into given execution graph."""
    # TODO: DEPRECATED: Remove it in future update.

    # Don't modify the original properties.
    property_graphs = [p.get_copy() for p in property_graphs]

    # execution_graph.visualize("Execution Graph")

    theorems, properties_to_prove = split_into_theorems_and_properties_to_prove(property_graphs)

    for p in negate_conclusion_part_of_properties(properties_to_prove):
        proved, theorems_applied, intermediate_graphs = \
            prove_property(execution_graph, p, theorems, theorem_selector)

        if proved:
            if full_visualization_enabled:
                visualize_proving_process(intermediate_graphs, theorems_applied, p)
            raise PropertyNotHoldsException(p.get_property_textual_representation(), None)


def prove_property(execution_graph,
                   property_graph,
                   theorems,
                   theorem_selector=None,
                   prove_if_fully_reduced=PROVE_IF_FULLY_REDUCED):
    """Proves that given property holds into given execution graph by utilizing given theorems.

    :param execution_graph: Execution graph into which given property should be proved.
    :param property_graph: Graph of property to be proved.
    :param theorems: Premises to be used for proving given property.
    :param theorem_selector: Defines the method used for choosing next theorem. Can be either
        an instance of NextTheoremSelector or list of NextTheoremSelector instances.
        If such a list is provided, all contained theorem selectors are utilized, until
        one is able to prove given property. Their order into the list defines their order
        of utilization. If no selector is able to prove property, data from the first one
        are returned.
    :param prove_if_fully_reduced: If set to True, a property is considered proved only if
        execution graph can be converted to its form. If set to False, it is enough for a
        property to be contained into execution graph to be considered proved.

    :return:
        -proved: True in case given property has been proved, otherwise False.
        -theorems_applied: A list containing all theorems applied during proving process.
            This list is always returned, no matter if proving process was successful or not.
        -intermediate_graphs: A list containing modified execution graphs, right after each
            theorem application.
    """
    if not theorem_selector:
        theorem_selector = get_default_theorem_selector()

    if not isinstance(theorem_selector, list):
        theorem_selector = [theorem_selector]

    all_theorems_applied = []
    all_intermediate_graphs = []

    for ts in theorem_selector:
        proved, theorems_applied, intermediate_graphs = _prove_property_with_selector(
            execution_graph,
            property_graph,
            theorems,
            theorem_selector=ts,
            prove_if_fully_reduced=prove_if_fully_reduced
        )

        if proved:
            return proved, theorems_applied, intermediate_graphs

        all_theorems_applied.append(theorems_applied)
        all_intermediate_graphs.append(intermediate_graphs)

    return False, all_theorems_applied[0], all_intermediate_graphs[0]


def get_all_possible_modus_ponens(graph, properties):
    # TODO: Implemented in TimedPropertyGraph. Remove it from here.
    possible_modus_ponens = {}

    for p in properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        matching_cases, _, cases_timestamps, _ = graph.find_equivalent_subgraphs(assumption)
        if matching_cases:
            possible_modus_ponens[p] = matching_cases

    return possible_modus_ponens


def find_possible_theorem_applications(graph, theorems):
    # TODO: Move this function to the methods of TimedPropertyGraph.
    possible_theorem_applications = []
    for theorem in theorems:
        possible_theorem_applications.extend(graph.find_all_possible_modus_ponens(theorem))
    return possible_theorem_applications


def apply_theorem(graph, theorem_application):
    return graph.apply_modus_ponens(theorem_application)


def visualize_proving_process(execution_graphs, theorems_applied, proved_property,
                              display_assumption=True):
    """Visualizes a proving process by displaying a sequence of figures.

    :param execution_graphs: A sequence of TimedPropertyGraph objects with each one containing
            a single evolution of execution graph through the proving process. List starts
            with initial execution graph, just before applying any theorem.
    :param theorems_applied: A sequence of TimedPropertyGraph objects, containing the theorems
            applied in order to evolve execution graph. theorems_applied[i] should be the
            theorem used on execution_graphs[i], in order to obtain execution_graphs[i+1].
    :param proved_property: The property that was finally proved, in the form of a
            TimedPropertyGraph object.
    :param display_assumption: If set to True, assumption matching of theorems is displayed
            onto execution graphs with a different color.
    """
    if display_assumption:
        execution_graphs[0].colorize_subgraph(
            theorems_applied[0].actual_implication.get_top_level_implication_subgraphs()[0])
    execution_graphs[0].visualize("Initial graph for proving process.")
    execution_graphs[0].clear_colorization()

    # Visualize proving process.
    for i in range(len(theorems_applied)):
        theorems_applied[i].actual_implication.visualize(f"Theorem Applied #{i}")
        if display_assumption and i < len(theorems_applied)-1:
            execution_graphs[i+1].colorize_subgraph(
                theorems_applied[i+1].actual_implication.get_top_level_implication_subgraphs()[0])
        execution_graphs[i+1].visualize(f"Graph after applying theorem #{i}")
        execution_graphs[i+1].clear_colorization()

    # Visualize how the property was found not to hold.
    matching_cases, _, _, _ = execution_graphs[-1].find_equivalent_subgraphs(proved_property)
    for path in matching_cases[0]:
        execution_graphs[-1].graph.colorize_path(path)
    proved_property.visualize("Property that not holds.")
    execution_graphs[-1].visualize("Graph where property does not hold.", show_colorization=True)


def enable_full_visualization():
    global full_visualization_enabled
    full_visualization_enabled = True


def _prove_property_with_selector(execution_graph,
                                  property_graph,
                                  theorems,
                                  theorem_selector,
                                  prove_if_fully_reduced=PROVE_IF_FULLY_REDUCED):
    global prover_invocations

    temp_graph = execution_graph.get_copy()  # Modify a separate graph for each property.
    temp_graph.add_constant_property(NoPositiveAndNegativePredicatesSimultaneously(temp_graph))

    proved = False
    theorems_applied = []
    intermediate_graphs = [temp_graph.get_copy()]
    while len(theorems_applied) < MAX_PROOF_PATH:
        possible_theorems = find_possible_theorem_applications(temp_graph, theorems)
        if not possible_theorems:
            break

        next_theorem = theorem_selector.select_next(
            temp_graph, possible_theorems, property_graph, theorems_applied, prover_invocations)
        if not next_theorem:
            break
        # next_theorem.implication_graph.visualize("Next theorem to apply.")
        apply_theorem(temp_graph, next_theorem)
        # temp_graph.visualize("New execution graph.")
        theorems_applied.append(next_theorem)
        intermediate_graphs.append(temp_graph.get_copy())

        if prove_if_fully_reduced:
            # Check after each theorem application if property has been proved.
            matches = temp_graph.find_subgraph_matches(property_graph)
            proved = True if (matches and matches[0] == temp_graph) else False
            if proved:
                break

    if not prove_if_fully_reduced:
        proved = True if temp_graph.contains_property_graph(property_graph) else False

    prover_invocations += 1

    return proved, theorems_applied, intermediate_graphs
