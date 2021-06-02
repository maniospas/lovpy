import unittest

from tests.logipy.importer.sample_properties import get_threading_sample_properties
from logipy.monitor.monitored_predicate import Call, ReturnedBy
from logipy.graphs.timestamps import Timestamp
from logipy.logic.prover import *


class TestProver(unittest.TestCase):

    def test_erroneous_with_sebsequent_calls_to_lock_acquire(self):
        returned_by_allocate_lock = ReturnedBy("allocate_lock").convert_to_graph()
        returned_by_allocate_lock.set_timestamp(Timestamp(1))

        call_acquire = Call("acquire").convert_to_graph()
        call_acquire.set_timestamp(Timestamp(4))

        call_acquire2 = Call("acquire").convert_to_graph()
        call_acquire2.set_timestamp(Timestamp(7))

        total_graph = returned_by_allocate_lock
        total_graph.logical_and(call_acquire)
        total_graph.logical_and(call_acquire2)

        # total_graph.visualize("Graph on which properties should hold.")

        properties = get_threading_sample_properties()

        # properties[0].visualize("Property 1")
        # properties[1].visualize("Property 2")

        self.assertRaises(PropertyNotHoldsException, prove_set_of_properties,
                          properties, total_graph)

    def test_correct_with_sebsequent_calls_to_lock_acquire_and_release(self):
        returned_by_allocate_lock = ReturnedBy("allocate_lock").convert_to_graph()
        returned_by_allocate_lock.set_timestamp(Timestamp(1))

        call_acquire = Call("acquire").convert_to_graph()
        call_acquire.set_timestamp(Timestamp(4))

        call_release = Call("release").convert_to_graph()
        call_release.set_timestamp(Timestamp(6))

        call_acquire2 = Call("acquire").convert_to_graph()
        call_acquire2.set_timestamp(Timestamp(7))

        total_graph = returned_by_allocate_lock
        total_graph.logical_and(call_acquire)
        total_graph.logical_and(call_release)
        total_graph.logical_and(call_acquire2)

        # total_graph.visualize("Graph on which properties should hold.")

        properties = get_threading_sample_properties()

        # properties[0].visualize("Property 1")
        # properties[1].visualize("Property 2")

        exception_raised = False

        try:
            prove_set_of_properties(properties, total_graph)
        except PropertyNotHoldsException:
            exception_raised = True

        if exception_raised:
            self.fail("PropertyNotHoldsException raised.")

    def test_erroneous_with_subsequent_calls_to_lock_acquire_and_delayed_call_to_release(self):
        returned_by_allocate_lock = ReturnedBy("allocate_lock").convert_to_graph()
        returned_by_allocate_lock.set_timestamp(Timestamp(1))

        call_acquire = Call("acquire").convert_to_graph()
        call_acquire.set_timestamp(Timestamp(4))

        call_acquire2 = Call("acquire").convert_to_graph()
        call_acquire2.set_timestamp(Timestamp(6))

        call_release = Call("release").convert_to_graph()
        call_release.set_timestamp(Timestamp(7))

        total_graph = returned_by_allocate_lock
        total_graph.logical_and(call_acquire)
        total_graph.logical_and(call_acquire2)
        total_graph.logical_and(call_release)

        # total_graph.visualize("Graph on which properties should hold.")

        properties = get_threading_sample_properties()

        # properties[0].visualize("Property 1")
        # properties[1].visualize("Property 2")

        self.assertRaises(PropertyNotHoldsException, prove_set_of_properties,
                          properties, total_graph)



