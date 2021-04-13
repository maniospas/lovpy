import copy
import logging
import random
import string

import logipy.logic.prover as prover
from logipy.logic.monitored_predicate import Call, ReturnedBy, CalledBy
from logipy.logic.timed_property_graph import TimedPropertyGraph
from logipy.logic.timestamps import Timestamp, is_interval_subset
from logipy.monitor.time_source import TimeSource


LOGGER_NAME = "logipy.trainer.dataset_generator"
INVALID_THEOREMS_PER_VALID_THEOREM = 10


class DatasetEntity:
    def __init__(self):
        # Attributes referring to the current state of execution graph.
        self.current_graph = TimedPropertyGraph()  # Current execution graph.
        self.current_goal_predicates = []          # Current predicates in execution graph.
        self.current_validity_intervals = []       # Intervals during which current predicates hold.
        self.timesource = TimeSource()             # A local timesource for building exec graph.

        # Attributes referring to the property that should be finally proved.
        self.goal = None                   # Property to finally be proved.
        self.goal_predicates = []          # Predicates that should hold to prove goal.
        self.goal_validity_intervals = []  # Intervals where goal predicates should hold.

        # Attributes referring to theorem application sequence.
        self.next_theorem = None        # Next theorem to be applied.
        self.application_sequence = []  # Theorems reversely applied so far.
        # Indicates whether next theorem should be applied to reach the final goal.
        self.is_correct = True

    def __deepcopy__(self, memodict={}):
        pass  # TODO: Implement

    def add_property_to_prove(self, property_to_prove):
        """Adds a new goal property into the graph of the sample.

        Currently, only one property to prove is supported. Trying to add a second one, will
        raise a RuntimeError.

        :param property_to_prove: An implication TimedPropertyGraph.
        """
        if self.goal:
            raise RuntimeError("Trying to add a second property to prove into the sample graph.")

        self.goal = property_to_prove
        property_instance, self.goal_predicates, self.goal_validity_intervals = \
            self._generate_newer_absolute_property_instance(property_to_prove)

        self.current_graph.logical_and(property_instance)
        self._update_timesource()

    def add_properties_of_theorem(self, theorem):
        """Adds an instance of the assumption of given theorem to current graph.

        There are two possible cases:
            1. Predicates of the newly added instance invalidate some predicates needed to
                prove the goal theorem, so it cannot be proved anymore. In that case, the
                theorem application process should stop, as the goal cannot be proved.
            2. Predicates of the newly added instance do not affect the possibility to prove
                the final goal, so current next theorem to be applied still remains the one
                that should be applied next.

        :param theorem: An implication TimedPropertyGraph object.
        """
        theorem_instance, basic_predicates, validity_intervals = \
            self._generate_newer_absolute_property_instance(theorem)
        assumption, _ = theorem_instance.get_top_level_implication_subgraphs()

        if self._predicates_invalidate_goal(basic_predicates, validity_intervals):
            # If goal predicates do not hold anymore, goal cannot be proved, so the best
            # option is to stop applying theorems.
            self.next_theorem = None
            self._update_current_predicates(basic_predicates, validity_intervals)
        else:  # Otherwise, the best theorem is still the last used one.
            # Well, if a predicate was replaced by the newly added instance, then that's the best
            # theorem to use.
            if not prover.find_possible_theorem_applications(self.current_graph, self.next_theorem):
                logger = logging.getLogger(LOGGER_NAME)
                logger.info("Sample Info: Theorem application sequence broke.")
                self.next_theorem = theorem

        self.current_graph.logical_and(assumption)
        self._update_timesource()

    def contains_property_to_prove(self):
        """Returns True if a goal property has been added to current graph."""
        return bool(self.goal)

    def get_reverse_theorem_applications(self, theorems):
        """Returns all possible applications of the reverse of given theorems."""
        reversed_theorems = []
        for t in theorems:
            reversed_theorem = t.get_copy()
            reversed_theorem.switch_implication_parts()
            reversed_theorems.append(reversed_theorem)
        return prover.find_possible_theorem_applications(self.current_graph, reversed_theorems)

    def generate_negative_sample(self, next_invalid_theorem):
        pass  # TODO: Implement

    def expand_with_theorem(self, reverse_theorem_application):
        """Expands current graph by reversely applying a theorem."""
        self.application_sequence.append(reverse_theorem_application)
        self.next_theorem = reverse_theorem_application.implication_graph
        self.current_graph.apply_modus_ponens(reverse_theorem_application)
        self._update_timesource()

    def expand_with_random_predicates(self):
        """Expands current graph by adding predicates that do not belong to any property."""
        predicate_name = _generate_random_text(random.randint(8, 16))
        option = random.randint(1, 3)
        if option == 1:
            predicate = Call(predicate_name).convert_to_graph()
        elif option == 2:
            predicate = ReturnedBy(predicate_name).convert_to_graph()
        else:
            predicate = CalledBy(predicate_name).convert_to_graph()

        self._randomly_shift_timesource()
        timestamp = Timestamp(self.timesource.get_current_time())
        predicate.set_timestamp(timestamp)

        self.current_graph.logical_and(predicate)

    def _generate_newer_absolute_property_instance(self, property_to_prove):
        """Converts a property with relative timestamps, to one with absolute timestamps.

        New absolute timestamps are calculated to be newer than most recent timestamp in graph.

        :param property_to_prove: A property in the form of TimedPropertyGraph.

        :return:
            -absolute_instance: A copy of given property with absolute timestamps.
            -basic_predicates
            -validity_intervals
        """
        # TODO: Support instance generation with invalid timestamps (negative samples).
        basic_predicates = property_to_prove.get_basic_predicates()
        absolute_instance = property_to_prove.get_copy()

        # Intervals during which predicates should have been applied in order to be valid.
        possible_intervals = self._find_newer_possible_intervals(basic_predicates)
        # Intervals during which predicates hold in the graph.
        validity_intervals = []
        for i in range(len(basic_predicates)):
            new_time = get_random_value_in_interval(possible_intervals[i])
            # Validity interval starts from the newly assigned timestamp.
            validity_intervals.append([new_time, "inf"])  # TODO: Rewrite it more elegantly.
            absolute_timestamp = Timestamp(new_time)
            absolute_instance.update_subgraph_timestamp(basic_predicates[i], absolute_timestamp)

        return absolute_instance, basic_predicates, validity_intervals

    def _find_newer_possible_intervals(self, predicates):
        """Find possible intervals that are newer than current most recent timestamp.

        :param predicates: A set of predicates in the form of TimedPropertyGraph objects.

        :return: A list of intervals inside which given predicates can be timestamped, in order
                all of them to concurrently hold.
        """
        validity_intervals = []

        self._randomly_shift_timesource()

        for predicate in predicates:
            predicate_paths = predicate.get_all_paths()
            predicate_timestamp = min([p.timestamp for p in predicate_paths])
            if not predicate_timestamp.is_absolute():
                predicate_timestamp.set_time_source(self.timesource)
            validity_intervals.append(predicate_timestamp.get_validity_interval())

        return validity_intervals

    def _predicates_invalidate_goal(self, predicates, predicates_interval):
        """Checks whether given predicates invalidate goal predicates if added to graph."""
        if not self.goal_predicates:
            raise RuntimeError("No goal property has been added to the graph yet.")

        remaining_predicates, remaining_intervals = \
            self._find_predicates_that_still_hold(predicates, predicates_interval)

        return not predicates_in_predicates_set(
            self.goal_predicates, self.goal_validity_intervals,
            remaining_predicates, remaining_intervals
        )

    def _find_predicates_that_still_hold(self, predicates, predicates_interval):
        """Finds predicates from current graph that still hold after adding given predicates."""
        invalidated_predicates_indexes = set()

        for i in range(len(self.goal_predicates)):
            for j in range(len(predicates)):
                if predicate_invalidates_predicate(
                        predicates[j], predicates_interval[j],
                        self.goal_predicates[i], self.goal_validity_intervals[i]):
                    invalidated_predicates_indexes.add(i)

        remaining_current_predicates = [self.current_goal_predicates[i]
                                        for i in range(len(self.current_goal_predicates))
                                        if i not in invalidated_predicates_indexes]
        remaining_current_intervals = [self.current_validity_intervals[i]
                                       for i in range(len(self.current_validity_intervals))
                                       if i not in invalidated_predicates_indexes]

        return remaining_current_predicates, remaining_current_intervals

    def _update_current_predicates(self, predicates, predicates_interval):
        """Updates current set of predicates that hold with given predicates."""
        if self.current_goal_predicates:
            # Only when current predicate set is not empty, it is meaningful to search for
            # the ones that still hold.
            self.current_goal_predicates, self.current_validity_intervals = \
                self._find_predicates_that_still_hold(predicates, predicates_interval)

        self.current_goal_predicates.extend(predicates)
        self.current_validity_intervals.extend(predicates_interval)

    def _update_timesource(self):
        """Updates local timesource to match the most recent timestamp in current graph."""
        latest_timestamp = self.current_graph.get_most_recent_timestamp()
        if not latest_timestamp.is_absolute():
            raise RuntimeError("Relative timestamp found in sample's execution graph.")
        while self.timesource.get_current_time() < latest_timestamp.get_absolute_value():
            self.timesource.stamp_and_increment()

    def _randomly_shift_timesource(self):
        """Shifts local timesource by a random number of time steps in [1, 10]."""
        shift = random.randint(1, 10)
        for i in range(shift):
            self.timesource.stamp_and_increment()


class DatasetGenerator:

    def __init__(self, properties, max_depth, total_samples,
                 random_expansion_probability=0.7,
                 add_new_property_probability=0.2):
        self.max_depth = max_depth
        self.total_samples = total_samples
        self.samples_generated = 0
        self.random_expansion_probability = random_expansion_probability
        self.add_new_property_probability = add_new_property_probability

        self.theorems, properties_to_prove = \
            prover.split_into_theorems_and_properties_to_prove(properties)
        # The properties I actually want to prove that hold are the negated ones.
        self.properties_to_prove = \
            prover.negate_conclusion_part_of_properties(properties_to_prove)

    def __iter__(self):
        return DatasetIterator(self)

    def next_sample(self):
        """Generates a new training sample with depth in range(1,max_depth+1)."""
        if self.samples_generated < self.total_samples:
            self.samples_generated += 1
            depth = random.randint(1, self.max_depth)
            return self.generate_sample(depth)
        else:
            return None

    def generate_sample(self, depth):
        """Generates a training sample with given depth."""
        sample = DatasetEntity()

        for i in range(depth):  # Apply 'depth' number of sample expansions.
            random_expansion = random.random() < self.random_expansion_probability

            if random_expansion:
                sample.expand_with_random_predicates()
            else:
                if sample.contains_property_to_prove():
                    add_new_properties = random.random() < self.add_new_property_probability

                    reverse_theorems = sample.get_reverse_theorem_applications(self.theorems)
                    if reverse_theorems and not add_new_properties:
                        sample.expand_with_theorem(random.choice(reverse_theorems))
                    else:
                        extra_theorems = [t for t in self.theorems if t not in reverse_theorems]
                        sample.add_properties_of_theorem(random.choice(extra_theorems))
                else:
                    sample.add_property_to_prove(random.choice(self.properties_to_prove))
        return sample

    # def _recursively_generate_next_theorem_dataset(self, depth):
    #     # TODO: Convert timestamps of generated graph to absolute ones.
    #
    #     if depth == 0:
    #         # The simplest case is the current graph to match the goal graph (self-proving).
    #         yield from self._generate_goals()
    #
    #     else:
    #         simpler_entities = self._recursively_generate_next_theorem_dataset(depth-1)
    #         for entity in simpler_entities:
    #             yield from self._reversely_expand_dataset_entity(entity)
    #             yield entity  # Always yield the shallower entities too.
    #
    # def _generate_goals(self):
    #     for goal in self.properties_to_prove:
    #         absolute_goal = convert_to_absolute(goal)
    #         yield DatasetEntity(absolute_goal, None, goal, True)
    #
    # def _reversely_expand_dataset_entity(self, entity):
    #     if entity.is_correct:
    #         # Expand each valid shallower entity in any possible way.
    #         for theorem in self.theorems:
    #             new_graph = reverse_apply_theorem(entity.current_graph, theorem)
    #             if new_graph:
    #                 # The theorem used to expand the simpler graph is considered to be the
    #                 # only valid one for reaching the final goal. All the rest theorems are
    #                 # considered to be invalid into reaching the goal.
    #                 yield DatasetEntity(new_graph, theorem, entity.goal, True)
    #
    #                 remaining_theorems = self.theorems.copy()
    #                 remaining_theorems.remove(theorem)
    #                 if len(remaining_theorems) > INVALID_THEOREMS_PER_VALID_THEOREM:
    #                     remaining_theorems = random.sample(
    #                         remaining_theorems, INVALID_THEOREMS_PER_VALID_THEOREM)
    #                 for remaining_theorem in remaining_theorems:
    #                     yield DatasetEntity(
    #                         new_graph, remaining_theorem, entity.goal, False)
    #             else:
    #                 noisy_graph = add_theorem_assumption(entity.current_graph, theorem)
    #                 yield DatasetEntity(noisy_graph, entity.next_theorem, entity.goal, True)


class DatasetIterator:
    """Iterator for DatasetGenerator."""
    def __init__(self, generator):
        self.generator = generator

    def __next__(self):
        next_sample = self.generator.next_sample()
        if not next_sample:
            raise StopIteration
        return next_sample
#
#
# def reverse_apply_theorem(graph, theorem):
#     reversed_theorem = theorem.get_copy()
#     reversed_theorem.switch_implication_parts()
#
#     possible_modus_ponens = prover.get_all_possible_modus_ponens(graph, [reversed_theorem])
#
#     expanded_graph = None
#     if possible_modus_ponens:
#         expanded_graph = graph.get_copy()
#         expanded_graph.apply_modus_ponens[0]
#
#     return expanded_graph
#
#
# def add_theorem_assumption(graph, theorem):
#     pass
#
#
# def convert_to_absolute(property_graph):
#     pass


def get_random_value_in_interval(interval):
    """Returns a random value in the given interval."""
    lower_bound = interval[0] if interval[0] != "-inf" else max(interval[1]-20, 0)
    upper_bound = interval[1] if interval[1] != "inf" else interval[0] + 20
    return random.randint(lower_bound, upper_bound)


def predicate_invalidates_predicate(new_predicate, new_validity_interval,
                                    old_predicate, old_validity_interval):
    new_predicate = new_predicate.get_copy()
    new_predicate.logical_not()
    old_predicate = old_predicate.get_copy()

    # Set the same timestamp to both predicates, so to only compare them structurally.
    mock_timestamp = Timestamp(0)
    new_predicate.set_timestamp(mock_timestamp)
    old_predicate.set_timestamp(mock_timestamp)

    matching_cases, _, _, _ = new_predicate.find_equivalent_subgraphs(old_predicate)
    if matching_cases and new_validity_interval[0] >= old_validity_interval[0]:
        return True
    else:
        return False


def predicates_in_predicates_set(subject_predicates, subject_intervals,
                                 predicate_set, intervals_set):
    """Checks whether given set of subject predicates belong to another predicate set.

    In order for two structurally equivalent predicates to be matched, the interval of subject
    predicate should be a subset of the interval of the predicate in predicates set.

    :param subject_predicates: Predicates in the form of TimedPropertyGraph objects, whose
            matching in predicate_set should be checked.
    :param subject_intervals: Intervals during which subject predicates hold.
    :param predicate_set: The set of predicates in which subject predicates should be matched.
    :param intervals_set: Intervals during which predicates of predicate set hold.

    :return: True if all subject predicates are matched into predicate set, otherwise False.
    """
    mock_timestamp = Timestamp(0)

    for i in range(len(subject_predicates)):
        subject_p = subject_predicates[i].get_copy()
        subject_p.set_timestamp(mock_timestamp)

        for j in range(len(predicate_set)):
            p = predicate_set[j].get_copy()
            p.set_timestamp(mock_timestamp)

            matching_cases, _, _, _ = p.find_equivalent_subgraphs(subject_p)
            if matching_cases and is_interval_subset(subject_intervals[i], intervals_set[i]):
                break
        else:
            return False  # Not matching one subject predicate is enough for matching to fail.

    return True


def _generate_random_text(length):
    return "".join(random.choices(string.ascii_lowercase, k=length))
