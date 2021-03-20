import random

import logipy.logic.prover as prover


INVALID_THEOREMS_PER_VALID_THEOREM = 10


class DatasetEntity:
    def __init__(self, current_graph, next_theorem, goal, is_correct):
        self.current_graph = current_graph
        self.next_theorem = next_theorem
        self.goal = goal
        self.is_correct = is_correct


def generate_next_theorem_dataset(properties, max_depth):
    """Generates a sequence of graph cases along with theorems that lead or not to a given goal."""
    theorems, properties_to_prove = prover.split_into_theorems_and_properties_to_prove(properties)

    # The properties I actually want to prove that hold are the negated ones.
    properties_to_prove = prover.negate_conclusion_part_of_properties(properties_to_prove)

    yield from _recursively_generate_next_theorem_dataset(properties_to_prove, theorems, max_depth)


def _recursively_generate_next_theorem_dataset(properties, theorems, depth):

    if depth == 0:
        # The simplest case is the current graph to match the goal graph (self-proving).
        for goal in properties:
            yield DatasetEntity(goal, None, goal, True)
    else:
        simpler_entities = _recursively_generate_next_theorem_dataset(properties, theorems,
                                                                      depth-1)

        for entity in simpler_entities:
            if entity.is_correct:
                # Expand each valid shallower entity in any possible way.
                for theorem in theorems:
                    new_graph = reverse_apply_theorem(entity.current_graph, theorem)
                    if new_graph:
                        # The theorem used to expand the simpler graph is considered to be the
                        # only valid one for reaching the final goal. All the rest theorems are
                        # considered to be invalid into reaching the goal.
                        yield DatasetEntity(new_graph, theorem, entity.goal, True)

                        remaining_theorems = theorems
                        remaining_theorems.remove(theorem)
                        if len(remaining_theorems) > INVALID_THEOREMS_PER_VALID_THEOREM:
                            remaining_theorems = random.sample(
                                remaining_theorems, INVALID_THEOREMS_PER_VALID_THEOREM)
                        for remaining_theorem in remaining_theorems:
                            yield DatasetEntity(new_graph, remaining_theorem, entity.goal, False)

            yield entity  # Always yield the shallower entities too.


def reverse_apply_theorem(graph, theorem):
    reversed_theorem = theorem.get_copy()
    reversed_theorem.switch_implication_assumption_parts()

    possible_modus_ponens = prover.get_all_possible_modus_ponens(graph, [reversed_theorem])

    expanded_graph = None
    if possible_modus_ponens:
        expanded_graph = graph.get_copy()
        expanded_graph.apply_modus_ponens[0]

    return expanded_graph
