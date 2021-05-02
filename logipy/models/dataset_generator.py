import logging
import random
import string

from matplotlib import pyplot as plt
from matplotlib import image as mpimage

import logipy.config
import logipy.logic.prover as prover
from logipy.graphs.monitored_predicate import Call, ReturnedBy, CalledBy
from logipy.graphs.timed_property_graph import TimedPropertyGraph, PredicateNode
from logipy.graphs.timed_property_graph import TIMESTAMP_PROPERTY_NAME
from logipy.graphs.timed_property_graph import NoPositiveAndNegativePredicatesSimultaneously
from logipy.logic.timestamps import Timestamp, is_interval_subset
from logipy.graphs.logical_operators import NotOperator
from logipy.monitor.time_source import TimeSource
from logipy.config import get_scratchfile_path


LOGGER_NAME = "logipy.models.dataset_generator"
INVALID_THEOREMS_PER_VALID_THEOREM = 10


class DatasetEntity:
    def __init__(self):
        # Attributes referring to the current state of execution graph.
        self.current_graph = TimedPropertyGraph()  # Current execution graph.
        self.current_graph.add_constant_property(
            NoPositiveAndNegativePredicatesSimultaneously(self.current_graph))
        self.current_goal_predicates = []          # Current predicates in execution graph.
        self.current_validity_intervals = []       # Intervals during which current predicates hold.
        self.timesource = TimeSource()             # A local timesource for building exec graph.
        self.suppressed_predicates = set()

        # Attributes referring to the property that should be finally proved.
        self.goal = None                   # Property to finally be proved.
        self.is_provable = True            # Indicates if it is possible to prove goal.
        self.goal_predicates = []          # Predicates that should hold to prove goal.
        self.goal_validity_intervals = []  # Intervals where goal predicates should hold.

        # Attributes referring to theorem application sequence.
        self.next_theorem = None        # Next theorem to be applied.
        self.application_sequence = []  # Theorems reversely applied so far.
        # Indicates whether next theorem should be applied to reach the final goal.
        self.is_correct = True

    def __deepcopy__(self, memodict={}):
        pass  # TODO: Implement

    def add_property_to_prove(self, property_to_prove, goal=None, is_provable=True):
        """Adds a new goal property into the graph of the sample.

        Currently, only one property to prove is supported. Trying to add a second one, will
        raise a RuntimeError.

        :param property_to_prove: An implication TimedPropertyGraph.
        :param goal:
        :param is_provable:
        """
        if self.goal:
            raise RuntimeError("Trying to add a second property to prove into the sample graph.")

        self.goal = property_to_prove if not goal else goal
        self.is_provable = is_provable
        property_instance, self.goal_predicates, self.goal_validity_intervals = \
            self._generate_newer_absolute_property_instance(property_to_prove)

        self.current_graph.logical_and(property_instance)

        self._update_timesource()
        self._shift_current_graph_timestamps()
        non_monitored_predicates, non_monitored_intervals = \
            get_non_monitored_predicates(self.goal_predicates, self.goal_validity_intervals)
        self._update_suppressed_predicates(non_monitored_predicates, non_monitored_intervals)

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
        assumption, conclusion = theorem_instance.get_top_level_implication_subgraphs()

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

        self.current_graph.logical_and(prover.convert_implication_to_and(theorem_instance))
        self._update_timesource()

    def add_suppressed_predicate(self, suppressed_predicate):
        """Adds an instance of given suppressed predicate to current graph.

        When adding given suppressed predicate, the internal log of suppressed predicates is
        updated in order to consider as suppressed the negation of the added one for time
        moments older than the one of added instance.

        When adding the suppressed predicate, it is not checked if it really has been suppressed,
        so it is expected to have been built for example using get_suppressed_predicates()
        method.

        :param suppressed_predicate: A valid SuppressedPredicate object.
        """
        instance = suppressed_predicate.generate_instance()
        self.current_graph.logical_and(instance)
        self._update_suppressed_predicates(
            [instance], [[instance.get_most_recent_timestamp().get_absolute_value(), "inf"]]
        )
        self._shift_current_graph_timestamps()
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

    def get_suppressed_predicates(self):
        return list(self.suppressed_predicates)

    def generate_negative_sample(self, next_invalid_theorem):
        pass  # TODO: Implement

    def expand_with_theorem(self, reverse_theorem_application):
        """Expands current graph by reversely applying a theorem."""
        self.application_sequence.append(reverse_theorem_application)
        self.next_theorem = reverse_theorem_application.implication_graph
        self.current_graph.apply_modus_ponens(reverse_theorem_application)
        self._shift_current_graph_timestamps()
        self._update_timesource()

        basic_predicates = self.current_graph.get_basic_predicates()
        intervals = [[pred.get_most_recent_timestamp().get_absolute_value(), "inf"]
                     for pred in basic_predicates]
        self._update_suppressed_predicates(basic_predicates, intervals)

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

    def visualize(self, title="Sample"):
        """Visualizes sample in a single figure.

        Figure is consisted of three subplots:
         -The leftmost subplot is the instance of execution graph contained into the sample.
         -The central subplot is the goal property that should be proved.
         -The rightmost subplot is the next theorem to be applied.

        :param str title: A supertitle for the whole figure.
        """
        acurrent = self.current_graph.to_agraph("Current Graph")
        provable_text = "Provable" if self.is_provable else "Not Provable"
        agoal = self.goal.to_agraph(f"Goal Property - {provable_text}")
        anext = self.next_theorem.to_agraph("Next Theorem") if self.next_theorem else None

        # Export to disk temp jpg images of the three graphs.
        current_path = get_scratchfile_path("temp_current.jpg")
        goal_path = get_scratchfile_path("temp_goal.jpg")
        next_path = get_scratchfile_path("temp_next.jpg")
        acurrent.layout("dot")
        acurrent.draw(current_path)
        agoal.layout("dot")
        agoal.draw(goal_path)
        if anext:
            anext.layout("dot")
            anext.draw(next_path)

        # Plot the three graph images side by side.
        f, axarr = plt.subplots(1, 3, num=None, figsize=(54, 18), dpi=80,
                                facecolor='w', edgecolor='w')
        f.tight_layout()
        f.suptitle(title, fontsize=40, fontweight='bold')
        axarr[0].imshow(mpimage.imread(current_path))
        axarr[1].imshow(mpimage.imread(goal_path))
        if anext:
            axarr[2].imshow(mpimage.imread(next_path))
        for axes in axarr:
            axes.axis('off')
        plt.show()

        # Cleanup temp images.
        logipy.config.remove_scratchfile(current_path)
        logipy.config.remove_scratchfile(goal_path)
        if anext:
            logipy.config.remove_scratchfile(next_path)

    # def get_negated_theorem_applications(self, theorems):
    #     negated_theorems = []
    #     for t in theorems:
    #         assumption, conclusion = t.get_top_level_implication_subgraphs()
    #         conclusion.logical_not()
    #         assumption.logical_implication_conclusion()
    #         negated_theorems.append(assumption)
    #     return prover.find_possible_theorem_applications(self.current_graph, negated_theorems)

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

        previous_time = self.timesource.get_current_time()
        self._randomly_shift_timesource()

        for predicate in predicates:
            predicate_paths = predicate.get_all_paths()
            predicate_timestamp = min([p.timestamp for p in predicate_paths])
            if not predicate_timestamp.is_absolute():
                predicate_timestamp.set_time_source(self.timesource)
            validity_intervals.append(_constraint_lower_bound_of_interval(
                    predicate_timestamp.get_validity_interval(), previous_time))

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

    def _update_suppressed_predicates(self, new_predicates, validity_intervals):
        """Updates the record of suppressed predicates, when given predicates added.

        This update is based on the property that when a predicate exists both in positive
        and negative forms, the one with the newer timestamp is retained.

        :param new_predicates: A sequence of the new predicates added to the graph.
        :param validity_intervals: A sequence of the intervals of new predicates during
                which they hold.
        """

        update_predicates, update_intervals = get_non_monitored_predicates(new_predicates,
                                                                           validity_intervals)
        update_predicates, update_intervals = _find_non_suppressed_predicates(update_predicates,
                                                                              update_intervals)

        for i in range(len(update_predicates)):
            suppressed = SuppressedPredicateBuilder(
                update_predicates[i], update_intervals[i][0]).build()

            if suppressed not in self.suppressed_predicates:
                self.suppressed_predicates.add(suppressed)
            else:
                # Suppressed version is always considered to be the older one.
                grabber = SuppressedPredicateEqualGrabber(suppressed)
                assert grabber in self.suppressed_predicates
                old = grabber.actual
                self.suppressed_predicates.remove(old)
                self.suppressed_predicates.add(min(old, suppressed, key=lambda p: p.suppressed_at))

    def _randomly_shift_timesource(self):
        """Shifts local timesource by a random number of time steps in [1, 10]."""
        shift = random.randint(1, 10)
        for i in range(shift):
            self.timesource.stamp_and_increment()

    def _shift_current_graph_timestamps(self):
        paths = self.current_graph.get_all_paths()
        paths.sort(key=lambda path: path.timestamp)
        min_shift = 0
        if paths[0].timestamp.get_absolute_value() < 0:
            min_shift = -paths[0].timestamp.get_absolute_value()

        shift = random.randint(min_shift, min_shift + 10)

        for e in self.current_graph.graph.edges:
            old_timestamp = self.current_graph.graph.edges[
                e[0], e[1], e[2]][TIMESTAMP_PROPERTY_NAME]
            new_timestamp = Timestamp(old_timestamp.get_absolute_value()+shift)
            self.current_graph.graph.edges[
                e[0], e[1], e[2]][TIMESTAMP_PROPERTY_NAME] = new_timestamp
        # for p in paths:
        #     self.current_graph.update_path_timestamp(
        #         p.path, Timestamp(p.timestamp.get_absolute_value()+shift))
        for sup in self.suppressed_predicates:
            sup.suppressed_at += shift

        self._update_timesource()


class SuppressedPredicate:
    def __init__(self, graph, suppressed_at):
        self.graph = graph
        self.suppressed_at = suppressed_at
        self.pred_name = None
        self.is_negated = None

        if isinstance(graph.get_root_node(), NotOperator):
            self.pred_name = str(list(graph.graph.successors(graph.get_root_node()))[0])
            self.is_negated = True
        else:
            self.pred_name = str(graph.get_root_node())
            self.is_negated = False

    def __hash__(self):
        return hash(self.pred_name)

    def __eq__(self, other):
        try:
            return self.pred_name == other.pred_name
        except AttributeError:
            return NotImplemented

    def generate_instance(self):
        """Generates an instance of suppressed predicate with timestamp before suppression.

        :return: Suppressed predicate in the form of a TimedPropertyGraph object, with
                absolute timestamps.
        """
        instance = self.graph.get_copy()
        timestamp = Timestamp(random.randint(min(0, self.suppressed_at-1), self.suppressed_at-1))
        instance.set_timestamp(timestamp)
        return instance


class SuppressedPredicateBuilder:
    def __init__(self, positive_predicate, suppressed_at):
        self.positive_predicate = positive_predicate
        self.suppressed_at = suppressed_at

    def build(self):
        property_graph = self.positive_predicate.get_copy()
        property_graph.add_constant_property(
            NoPositiveAndNegativePredicatesSimultaneously(property_graph)
        )
        property_graph.logical_not()

        return SuppressedPredicate(property_graph, self.suppressed_at)


class SuppressedPredicateEqualGrabber:
    def __init__(self, suppressed_predicate):
        self.suppressed_predicate = suppressed_predicate
        self.actual = None

    def __hash__(self):
        return hash(self.suppressed_predicate)

    def __eq__(self, other):
        if self.suppressed_predicate == other:
            self.actual = other
            return True
        return False


class DatasetGenerator:

    def __init__(self, properties, max_depth, total_samples,
                 random_expansion_probability=0.7,
                 add_new_property_probability=0.2,
                 verbose=False):
        self.max_depth = max_depth
        self.total_samples = total_samples
        self.samples_generated = 0
        self.random_expansion_probability = random_expansion_probability
        self.add_new_property_probability = add_new_property_probability

        self.theorems, properties_to_prove = \
            prover.split_into_theorems_and_properties_to_prove(properties)
        # The properties I want to prove that hold are the negated ones.
        self.valid_properties_to_prove = \
            prover.negate_conclusion_part_of_properties(properties_to_prove)
        # Also keep the negation of properties to prove, to create negative samples.
        self.invalid_properties = [prover.convert_implication_to_and(p)
                                   for p in properties_to_prove]
        self.verbose = verbose

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
            random_expansion = False #random.random() < self.random_expansion_probability

            if random_expansion:
                sample.expand_with_random_predicates()
                if self.verbose:
                    sample.current_graph.visualize(f"#{i+1} Random expanded.")
            else:
                if sample.contains_property_to_prove():
                    reverse_theorems = sample.get_reverse_theorem_applications(self.theorems)
                    if reverse_theorems:
                        sample.expand_with_theorem(random.choice(reverse_theorems))
                        if self.verbose:
                            sample.current_graph.visualize(f"#{i+1} Reverse theorem expansion.")
                    else:
                        suppressed_predicates = sample.get_suppressed_predicates()
                        sample.add_suppressed_predicate(random.choice(suppressed_predicates))
                        if self.verbose:
                            sample.current_graph.visualize(
                                f"#{i+1} Expansions by suppressed predicate")
                else:
                    add_valid_property = bool(random.randint(0, 1))

                    if add_valid_property:
                        sample.add_property_to_prove(random.choice(self.valid_properties_to_prove))
                        if self.verbose:
                            sample.current_graph.visualize(f"#{i+1} Added property to prove.")
                    else:
                        invalid_index = random.randint(0, len(self.invalid_properties)-1)
                        valid_property = self.valid_properties_to_prove[invalid_index]
                        invalid_property = self.invalid_properties[invalid_index]
                        sample.add_property_to_prove(invalid_property, valid_property, False)
                        if self.verbose:
                            sample.current_graph.visualize(f"#{i + 1} Added negation of property.")
        return sample

    # def generate_sample_old_method(self, depth):
    #     """Generates a training sample with given depth."""
    #     sample = DatasetEntity()
    #     last_new_theorem_added = None
    #
    #     for i in range(depth):  # Apply 'depth' number of sample expansions.
    #         random_expansion = False #random.random() < self.random_expansion_probability
    #
    #         if random_expansion:
    #             sample.expand_with_random_predicates()
    #             sample.current_graph.visualize(f"#{i+1} Random expanded.")
    #         else:
    #             if sample.contains_property_to_prove():
    #                 add_new_properties = random.random() < self.add_new_property_probability
    #
    #                 reverse_theorems = sample.get_reverse_theorem_applications(self.theorems)
    #                 if reverse_theorems and not add_new_properties:
    #                     sample.expand_with_theorem(random.choice(reverse_theorems))
    #                     sample.current_graph.visualize(f"#{i+1} Reverse theorem expansion.")
    #                 else:
    #                     reverse_theorems_graphs = [t.implication_graph for t in reverse_theorems]
    #                     extra_theorems = [
    #                         t for t in self.theorems
    #                         if t not in reverse_theorems_graphs and t != last_new_theorem_added
    #                     ]
    #                     sample.add_properties_of_theorem(random.choice(extra_theorems))
    #                     sample.current_graph.visualize(f"#{i+1} New property expansion.")
    #             else:
    #                 valid_property = random.choice(self.valid_properties_to_prove)
    #                 sample.add_property_to_prove()
    #                 sample.current_graph.visualize(f"#{i+1} Added property to prove.")

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
        # TODO: Fix bug in find_equivalent_subgraphs() about getting wrong timestamps
        #  and remove while loop.
        while True:
            try:
                next_sample = self.generator.next_sample()

                if not next_sample:
                    raise StopIteration
                return next_sample
            except RuntimeError:
                pass
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


def get_non_monitored_predicates(predicates_set, validity_intervals):
    non_monitored = []
    non_monitored_intervals = []

    for i in range(len(predicates_set)):
        predicate = predicates_set[i]

        for n in predicate.graph.nodes:
            node_name = str(n)
            if isinstance(n, PredicateNode) and (
                    node_name.startswith("call") or node_name.startswith("returned_by") or
                    node_name.startswith("called_by")):
                break
        else:
            non_monitored.append(predicate)
            non_monitored_intervals.append(validity_intervals[i])

    return non_monitored, non_monitored_intervals


def _generate_random_text(length):
    return "".join(random.choices(string.ascii_lowercase, k=length))


def _constraint_lower_bound_of_interval(interval, constraint):
    if constraint != "-inf" and (interval[0] == "-inf" or interval[0] < constraint):
        return [constraint, interval[1]]
    else:
        return interval


def _find_non_suppressed_predicates(predicates, validity_intervals):
    """Returns the predicates that are left after suppression operations.

    :param predicates: A sequence of predicates in the form of TimedPropertyGraph objects.
    :param validity_intervals: The intervals during which corresponding predicates hold.
    """
    preds_by_name = {}
    for p in predicates:
        if isinstance(p.get_root_node(), NotOperator):
            pred_name = str(list(p.graph.successors(p.get_root_node()))[0])
            is_negated = True
        else:
            pred_name = str(p.get_root_node())
            is_negated = False

        if not pred_name in preds_by_name:
            preds_by_name[pred_name] = []
        preds_by_name[pred_name].append({"timestamp": p.get_most_recent_timestamp(),
                                         "is_negated": is_negated,
                                         "index": predicates.index(p)})

    non_suppressed_predicates = []
    non_suppressed_validity_intervals = []

    for base_name, preds in preds_by_name.items():
        preds.sort(reverse=True, key=lambda d: d["timestamp"])

        for p in preds:
            if p["is_negated"] == preds[0]["is_negated"]:
                non_suppressed_predicates.append(predicates[p["index"]])
                non_suppressed_validity_intervals.append(validity_intervals[p["index"]])
            else:
                break

    return non_suppressed_predicates, non_suppressed_validity_intervals
