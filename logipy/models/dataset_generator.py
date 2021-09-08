from copy import copy
import random
import string

import logipy.logic.prover as prover
import logipy.logic.properties as logipy_properties
from logipy.monitor.monitored_predicate import Call, ReturnedBy, CalledBy
from logipy.graphs.timed_property_graph import TimedPropertyGraph, PredicateNode
from logipy.graphs.timed_property_graph import (NoPositiveAndNegativePredicatesSimultaneously,
                                                NoComparisonRelativeTimestampAlone)
from logipy.graphs.timestamps import Timestamp, RelativeTimestamp, is_interval_subset
from logipy.graphs.logical_operators import NotOperator
from logipy.monitor.time_source import TimeSource
from . import io


LOGGER_NAME = "logipy.models.dataset_generator"


class DatasetEntity:
    def __init__(self, theorems):
        # Attributes referring to the current state of execution graph.
        self.current_graph = TimedPropertyGraph()  # Current execution graph.
        self.current_graph.add_constant_property(
                NoPositiveAndNegativePredicatesSimultaneously(self.current_graph))
        self.current_graph.add_constant_property(
                NoComparisonRelativeTimestampAlone(self.current_graph))
        self.current_goal_predicates = []          # Current predicates in execution graph.
        self.current_validity_intervals = []       # Intervals during which current predicates hold.
        self.timesource = TimeSource()             # A local timesource for building exec graph.
        self.suppressed_predicates = set()
        self.shifts_history = []                   # List of performed timestamp shifts.

        # Attributes referring to the property that should be finally proved.
        self.goal = None                   # Property to finally be proved.
        self.is_provable = True            # Indicates if it is possible to prove goal.
        self.goal_predicates = []          # Predicates that should hold to prove goal.
        self.goal_validity_intervals = []  # Intervals where goal predicates should hold.

        # Attributes referring to theorem application sequence.
        self.next_theorem = None        # Next theorem to be applied.
        self.application_sequence = []  # Theorems reversely applied so far.
        # Indicates whether next theorem should be applied to reach the final goal.
        self.next_theorem_correct = True
        self.all_theorems = theorems  # A list of all theorems out of which sample is built.

    def __copy__(self):
        new_entity = type(self)(self.all_theorems)

        # Attributes referring to the current state of execution graph.
        new_entity.current_graph = self.current_graph.get_copy()
        new_entity.current_goal_predicates = self.current_goal_predicates.copy()
        new_entity.current_validity_intervals = self.current_validity_intervals.copy()
        new_entity.timesource = copy(self.timesource)
        new_entity.suppressed_predicates = self.suppressed_predicates
        new_entity.shifts_history = copy(self.shifts_history)

        # Attributes referring to the property that should be finally proved.
        new_entity.goal = self.goal.get_copy()
        new_entity.is_provable = self.is_provable
        new_entity.goal_predicates = self.goal_predicates.copy()
        new_entity.goal_validity_intervals = self.goal_validity_intervals.copy()

        # Attributes referring to theorem application sequence.
        new_entity.next_theorem = self.next_theorem.get_copy() if self.next_theorem else None
        new_entity.application_sequence = self.application_sequence.copy()
        # Indicates whether next theorem should be applied to reach the final goal.
        new_entity.next_theorem_correct = self.next_theorem_correct

        return new_entity

    def add_property_to_prove(self, property_to_prove, goal=None, is_provable=True):
        """Adds a new goal property into the graph of the sample.

        Currently, only one property to prove is supported. Trying to add a second one, will
        raise a RuntimeError.

        :param property_to_prove: The property that should finally be proved, in the form of
                an implication TimedPropertyGraph.
        :param goal: When property_to_prove should be provable, goal should be set to the
                same value as property_to_prove. If None is provided, it is automatically
                set to the same value as property_to_prove.
        :param is_provable: Defines whether given goal can be proved from given initial
                property_to_prove.
        """
        if self.goal:
            raise RuntimeError("Trying to add a second property to prove into the sample graph.")

        self.goal = property_to_prove if not goal else goal
        self.is_provable = is_provable
        property_instance, self.goal_predicates, self.goal_validity_intervals = \
            self._generate_newer_absolute_property_instance(property_to_prove)

        self.current_graph.logical_and(property_instance)

        self._update_timesource()
        self.shift_current_graph_timestamps()
        non_monitored_predicates, non_monitored_intervals = \
            get_non_monitored_predicates(self.goal_predicates, self.goal_validity_intervals)
        self._update_suppressed_predicates(non_monitored_predicates, non_monitored_intervals)

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
        self.shift_current_graph_timestamps()
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

    def get_non_suppressible_suppressed_predicates(self):
        non_suppressible = [s for s in self.suppressed_predicates
                            if not self._is_suppressed_predicate_still_suppressible(s)]
        return non_suppressible

    def generate_negative_samples(self):
        """Generates all possible negative samples from a positive one.

        Negative samples are considered to be all samples having a different next theorem
        application from the one contained in current valid sample. Possible theorem
        applications are generated from theorems sequence provided in 'all_theorems'
        attribute.

        :raises RuntimeError: When generate_negative_samples is called on a negative sample.

        :return: A list of all possible negative samples.
        """
        if not self.next_theorem_correct:
            raise RuntimeError("Cannot generate a negative sample out of a negative sample.")

        available_modus_ponenses = prover.find_possible_theorem_applications(
                self.current_graph, self.all_theorems)
        # Remove the correct next modus ponens from available ones for negative samples.
        if self.next_theorem:
            available_modus_ponenses = [modus_ponens for modus_ponens in available_modus_ponenses
                                        if modus_ponens.actual_implication != self.next_theorem]

        negative_samples = []

        for modus_ponens in available_modus_ponenses:
            negative_sample = copy(self)
            negative_sample.next_theorem = modus_ponens.actual_implication
            negative_sample.next_theorem_correct = False
            negative_samples.append(negative_sample)

        return negative_samples

    def expand_with_theorem(self, reverse_theorem_application):
        """Expands current graph by reversely applying a theorem."""
        self.application_sequence.append(reverse_theorem_application)
        next_theorem = reverse_theorem_application.actual_implication
        self.next_theorem = next_theorem
        self.current_graph.apply_modus_ponens(reverse_theorem_application)
        self.shift_current_graph_timestamps()
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

    def is_next_theorem_correct(self):
        """Returns True if contained next theorem is the right one in proving sequence."""
        assert bool(self.next_theorem)
        return self.next_theorem_correct

    def should_proving_process_terminate(self):
        """Returns True if proving process of goal into current graph should terminate."""
        return not self.is_provable or not self.application_sequence

    def shift_current_graph_timestamps(self, shift=None):
        if not shift:
            paths = self.current_graph.get_all_paths()
            paths.sort(key=lambda path: path.timestamp)
            min_shift = 0
            if paths[0].timestamp.get_absolute_value() < 0:
                min_shift = -paths[0].timestamp.get_absolute_value()

            shift = random.randint(min_shift, min_shift + 10)

        self.current_graph.shift_graph_timestamps(shift)
        if self.next_theorem:
            self.next_theorem.shift_graph_timestamps(shift)

        # for p in paths:
        #     self.current_graph.update_path_timestamp(
        #         p.path, Timestamp(p.timestamp.get_absolute_value()+shift))
        for sup in self.suppressed_predicates:
            sup.suppressed_at += shift

        self._update_timesource()
        self.shifts_history.append(shift)

    def visualize(self, title="Sample", export_path=None, visualize_next_assumption=True):
        """Visualizes sample in a single figure.

        Figure is consisted of three subplots:
         -The leftmost subplot is the instance of execution graph contained into the sample.
         -The central subplot is the goal property that should be proved.
         -The rightmost subplot is the next theorem to be applied.

        :param str title: A supertitle for the whole figure.
        :param Path export_path: If this argument is given, then instead of displaying the
                sample figure on screen, it is exported to pointed location.
        :param bool visualize_next_assumption: When set to True, visualizes assumption part
                of next theorem into current graph.
        """
        provable_text = "Provable" if self.is_provable else "Not Provable"
        goal_title = f"Goal Property - {provable_text}"
        negative_text = "Positive" if self.next_theorem_correct else "Negative"
        next_title = f"Next Theorem - {negative_text}"

        io.visualize_three_graphs(
            self.current_graph, self.goal, self.next_theorem,
            title=title,
            export_path=export_path,
            goal_title=goal_title,
            next_title=next_title,
            visualize_next_assumption_in_current=(bool(self.next_theorem) and
                                                  visualize_next_assumption)
        )

    # def add_properties_of_theorem(self, theorem):
    #     """Adds an instance of the assumption of given theorem to current graph.
    #
    #     There are two possible cases:
    #         1. Predicates of the newly added instance invalidate some predicates needed to
    #             prove the goal theorem, so it cannot be proved anymore. In that case, the
    #             theorem application process should stop, as the goal cannot be proved.
    #         2. Predicates of the newly added instance do not affect the possibility to prove
    #             the final goal, so current next theorem to be applied still remains the one
    #             that should be applied next.
    #
    #     :param theorem: An implication TimedPropertyGraph object.
    #     """
    #     theorem_instance, basic_predicates, validity_intervals = \
    #         self._generate_newer_absolute_property_instance(theorem)
    #     assumption, conclusion = theorem_instance.get_top_level_implication_subgraphs()
    #
    #     if self._predicates_invalidate_goal(basic_predicates, validity_intervals):
    #         # If goal predicates do not hold anymore, goal cannot be proved, so the best
    #         # option is to stop applying theorems.
    #         self.next_theorem = None
    #         self._update_current_predicates(basic_predicates, validity_intervals)
    #     else:  # Otherwise, the best theorem is still the last used one.
    #         # Well, if a predicate was replaced by the newly added instance, then that's the best
    #         # theorem to use.
    #         if not prover.find_possible_theorem_applications(self.current_graph,
    #                                                          self.next_theorem):
    #             logger = logging.getLogger(LOGGER_NAME)
    #             logger.info("Sample Info: Theorem application sequence broke.")
    #             self.next_theorem = theorem
    #
    #     self.current_graph.logical_and(prover.convert_implication_to_and(theorem_instance))
    #     self._update_timesource()

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
        property_to_prove = property_to_prove.get_copy()
        basic_predicates = property_to_prove.get_basic_predicates()
        absolute_instance = property_to_prove.get_copy()

        # Intervals during which predicates should have been applied in order to be valid.
        possible_intervals = self._find_newer_possible_intervals(basic_predicates)
        # Intervals during which predicates hold in the graph.
        validity_intervals = []
        for basic_pred, interval in zip(basic_predicates, possible_intervals):
            new_time = get_random_value_in_interval(interval)
            # Validity interval starts from the newly assigned timestamp.
            validity_intervals.append([new_time, "inf"])  # TODO: Rewrite it more elegantly.
            absolute_timestamp = Timestamp(new_time)
            absolute_instance.update_subgraph_timestamp(basic_pred, absolute_timestamp)

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

    def _is_suppressed_predicate_still_suppressible(self, suppressed_predicate):
        """Checks whether given suppressed predicate will be suppressed again if added to graph.

        :param SuppressedPredicate suppressed_predicate: A suppressed predicate of current graph.

        :return: True if given predicate will be suppressed again if added to current graph.
                Otherwise, it returns False.
        """
        inv_pred_graph = suppressed_predicate.graph.get_copy()
        inv_pred_graph.logical_not()
        inv_pred_graph.set_timestamp(RelativeTimestamp(0))
        matches, _, _, _ = self.current_graph.find_equivalent_subgraphs(inv_pred_graph)
        return True if matches else False


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


class EntitiesSequenceVisualizer:
    """Visualizer of evolutions of an entity that have been individually shifted in time."""
    def __init__(self):
        self.entities = []
        self.entities_total_shift = []
        self.max_total_shift = 0
        self.entities_title = []

    def add(self, entity: DatasetEntity, title=""):
        self.entities.append(copy(entity))
        total_shift = sum(entity.shifts_history)
        self.entities_total_shift.append(total_shift)
        self.max_total_shift = max(self.max_total_shift, total_shift)
        self.entities_title.append(title)

    def visualize(self):
        for e, total_shift, title in zip(self.entities, self.entities_total_shift,
                                         self.entities_title):
            e.shift_current_graph_timestamps(self.max_total_shift - total_shift)
            e.visualize(title=title, visualize_next_assumption=False)


class DatasetGenerator:

    def __init__(self, properties, max_depth, total_samples,
                 random_expansion_probability=0.7,
                 add_new_property_probability=0.2,
                 negative_samples_percentage=0.8,
                 verbose=False):

        self.max_depth = max_depth
        self.total_samples = total_samples
        self.samples_generated = 0
        self.random_expansion_probability = random_expansion_probability
        self.add_new_property_probability = add_new_property_probability
        self.negative_samples_percentage = negative_samples_percentage
        self.verbose = verbose

        self.theorems, properties_to_prove = \
            prover.split_into_theorems_and_properties_to_prove(properties)
        # The properties I want to prove that hold are the negated ones.
        self.valid_properties_to_prove = \
            prover.negate_conclusion_part_of_properties(properties_to_prove)
        # Also keep the negation of properties to prove, to create negative samples.
        self.invalid_properties = [logipy_properties.convert_implication_to_and(p)
                                   for p in properties_to_prove]

        self.negative_samples = []  # Contains all possible negative samples so far.

    def __iter__(self):
        return DatasetIterator(self)

    def next_sample(self):
        """Generates a new training sample with depth in range(1,max_depth+1)."""
        sample = None

        if self.samples_generated < self.total_samples:  # Max samples not reached yet.
            # Firstly the percentage of positive samples is covered.
            if (self.samples_generated <
                    int(self.total_samples*(1-self.negative_samples_percentage))
                    or len(self.negative_samples) == 0):

                # As long as generated sample doesn't contain next theorem, discard it and
                # keep only the negative samples generated out of it.
                sample_contains_next = False
                while not sample_contains_next:
                    depth = random.randint(1, self.max_depth)
                    sample = self.generate_sample(depth)
                    sample_contains_next = bool(sample.next_theorem)

                    negative_samples = sample.generate_negative_samples()
                    if self.verbose:
                        sample.visualize(f"Sample #{self.samples_generated+1}")
                        for i, s in enumerate(negative_samples):
                            s.visualize(f"Negative sample #{i+1}")
                    self.negative_samples.extend(negative_samples)
            else:
                sample = random.choice(self.negative_samples)
                self.negative_samples.remove(sample)

            self.samples_generated += 1

        return sample

    def generate_sample(self, depth):
        """Generates a training sample with given depth."""
        sample = DatasetEntity(self.theorems)
        visualizer = EntitiesSequenceVisualizer()

        no_further_expansion_possible = False

        for i in range(depth):  # Apply 'depth' number of sample expansions.
            random_expansion = random.random() < self.random_expansion_probability

            if random_expansion:
                sample.expand_with_random_predicates()
                if self.verbose:
                    visualizer.add(sample, title=f"#{i+1} Random expanded.")
            elif not no_further_expansion_possible:
                if sample.contains_property_to_prove():
                    added_suppressed_predicate, no_further_expansion_possible = \
                        self._expand_sample_with_theorem(sample)
                    if no_further_expansion_possible:
                        continue
                    if self.verbose:
                        if added_suppressed_predicate:
                            visualizer.add(
                                sample, title=f"#{i+1} Expansions by suppressed predicate")
                        else:
                            visualizer.add(sample, title=f"#{i+1} Reverse theorem expansion.")
                else:
                    add_valid_property = bool(random.randint(0, 1))
                    self._add_property_to_sample(sample, add_valid_property)
                    if self.verbose:
                        if add_valid_property:
                            visualizer.add(sample, title=f"#{i+1} Added property to prove.")
                        else:
                            visualizer.add(sample, title=f"#{i+1} Added negation of property.")

        if self.verbose:
            visualizer.visualize()

        return sample

    def _add_property_to_sample(self, sample: DatasetEntity, add_valid_property: bool):
        if add_valid_property:
            sample.add_property_to_prove(random.choice(self.valid_properties_to_prove))
        else:
            invalid_index = random.randint(0, len(self.invalid_properties) - 1)
            valid_property = self.valid_properties_to_prove[invalid_index]
            invalid_property = self.invalid_properties[invalid_index]
            sample.add_property_to_prove(invalid_property, valid_property, False)

    def _expand_sample_with_theorem(self, sample: DatasetEntity):
        """Expands given sample using a theorem or by a suppressed predicate.

        If a theorem can be reversely applied, it is always preferred. However, when
        no more theorems can be reversely applied, expansion by a suppressed predicate
        is attempted. If there are no suppressed predicates to be added, further expansion
        by theorems is not possible and such value is returned.

        :return:
            -added_suppressed_predicate: boolean
            -no_further_expansion_possible: boolean
        """
        reverse_theorems = sample.get_reverse_theorem_applications(self.theorems)
        added_suppressed_predicate = False
        no_further_expansion_possible = False

        if reverse_theorems:
            expansion_theorem = random.choice(reverse_theorems)
            sample.expand_with_theorem(expansion_theorem)
            reverse_theorems.remove(expansion_theorem)
        else:  # When no theorems can be reversely applied, add a suppressed predicate.
            suppressed_predicates = sample.get_non_suppressible_suppressed_predicates()
            if suppressed_predicates:
                sample.add_suppressed_predicate(random.choice(suppressed_predicates))
                added_suppressed_predicate = True
            else:
                no_further_expansion_possible = True

        return added_suppressed_predicate, no_further_expansion_possible

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
        next_sample = self.generator.next_sample()
        if not next_sample:
            raise StopIteration
        return next_sample


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

        if pred_name not in preds_by_name:
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
