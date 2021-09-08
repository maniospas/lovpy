import unittest
from copy import deepcopy

from tests.logipy.importer.sample_properties import get_threading_sample_properties
from logipy.monitor.monitored_predicate import Call, ReturnedBy
from logipy.graphs.timestamps import Timestamp
from logipy.logic.prover import *
from logipy.logic.properties import split_into_theorems_and_properties_to_prove
from logipy.importer.gherkin_importer import convert_gherkin_to_graphs


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

    def test_prove_property_with_provable_property(self):
        call_acquire = Call("acquire").convert_to_graph()
        call_acquire.set_timestamp(Timestamp(4))

        call_release = Call("release").convert_to_graph()
        call_release.set_timestamp(Timestamp(11))

        call_acquire2 = Call("acquire").convert_to_graph()
        call_acquire2.set_timestamp(Timestamp(127))

        call_release2 = Call("release").convert_to_graph()
        call_release2.set_timestamp(Timestamp(229))

        call_acquire3 = Call("acquire").convert_to_graph()
        call_acquire3.set_timestamp(Timestamp(310))

        total_graph = deepcopy(call_acquire)
        total_graph.logical_and(call_release)
        total_graph.logical_and(call_acquire2)
        total_graph.logical_and(call_release2)
        total_graph.logical_and(call_acquire3)
        # total_graph.visualize("Base Graph")

        theorems, properties_to_prove = split_into_theorems_and_properties_to_prove(
            get_threading_sample_properties())
        # for i, t in enumerate(theorems):
        #     t.visualize(f"Theorem #{i+1}")
        # for i, p in enumerate(properties_to_prove):
        #     p.visualize(f"Property #{i+1}")

        proved, theorems, intermediate = prove_property(total_graph, properties_to_prove[0],
                                                        theorems)
        self.assertTrue(proved)
        
        # visualize_proving_process(intermediate, theorems, properties_to_prove[0],
        #                           display_assumption=False)

    def test_find_possible_theorem_applications(self):
        call_acquire = Call("acquire").convert_to_graph()
        call_acquire.set_timestamp(Timestamp(4))

        call_release = Call("release").convert_to_graph()
        call_release.set_timestamp(Timestamp(11))

        call_acquire2 = Call("acquire").convert_to_graph()
        call_acquire2.set_timestamp(Timestamp(127))

        call_release2 = Call("release").convert_to_graph()
        call_release2.set_timestamp(Timestamp(229))

        call_acquire3 = Call("acquire").convert_to_graph()
        call_acquire3.set_timestamp(Timestamp(310))

        total_graph = deepcopy(call_acquire)
        total_graph.logical_and(call_release)
        total_graph.logical_and(call_acquire2)
        total_graph.logical_and(call_release2)
        total_graph.logical_and(call_acquire3)
        # total_graph.visualize("Base Graph")

        theorems, properties_to_prove = split_into_theorems_and_properties_to_prove(
            get_threading_sample_properties())

        next_theorems = find_possible_theorem_applications(total_graph, theorems)
        # for i, t in enumerate(next_theorems):
        #     t.actual_implication.visualize(f"Theorem Application #{i+1}")

        self.assertEqual(len(next_theorems), 3)

    def test_prover_with_non_chronological_theorems(self):
        gherkin = """
            SCENARIO:
                WHEN call perform_maintainance
                THEN performing_maintainance
            
            SCENARIO:
                WHEN call checkpoint
                THEN checkpoint_reached
            
            SCENARIO:
                GIVEN performing_maintainance
                AND checkpoint_reached
                WHEN low_resources
                THEN NOT able_to_report
            
            SCENARIO:
                GIVEN performing_maintainance
                WHEN checkpoint_reached
                THEN able_to_report
            
            SCENARIO:
                GIVEN call receive_big_data
                WHEN call request_heavy_processing
                THEN low_resources
            
            SCENARIO:
                GIVEN low_resources
                WHEN call offload
                THEN NOT low_resources
            
            SCENARIO:
                WHEN call visualize
                THEN SHOULD able_to_report
        """
        graphs = convert_gherkin_to_graphs(gherkin)

        theorems, properties_to_prove = split_into_theorems_and_properties_to_prove(graphs)

        for i, t in enumerate(theorems):
            t.visualize(f"Theorem #{i+1}")
        for i, p in enumerate(properties_to_prove):
            p.visualize(f"Property #{i+1}")

        call_maint = Call("perform_maintainance").convert_to_graph()
        call_maint.set_timestamp(Timestamp(3))
        call_rec_big_data = Call("receive_big_data").convert_to_graph()
        call_rec_big_data.set_timestamp(Timestamp(6))
        call_checkpoint = Call("checkpoint").convert_to_graph()
        call_checkpoint.set_timestamp(Timestamp(9))
        call_proc = Call("request_heavy_processing").convert_to_graph()
        call_proc.set_timestamp(Timestamp(12))
        call_vis = Call("visualize").convert_to_graph()
        call_vis.set_timestamp(Timestamp(15))
        call_offload = Call("offload").convert_to_graph()
        call_offload.set_timestamp(Timestamp(20))
        base_graph = deepcopy(call_maint)
        base_graph.logical_and(call_rec_big_data)
        base_graph.logical_and(call_checkpoint)
        base_graph.logical_and(call_proc)
        base_graph.logical_and(call_vis)
        # base_graph.logical_and(call_offload)
        # base_graph.visualize("Base Graph")

        proved, theorems, intermediate = prove_property(base_graph, properties_to_prove[0],
                                                        theorems)
        # self.assertTrue(proved)

        # visualize_proving_process(intermediate, theorems, properties_to_prove[0],
        #                           display_assumption=False)

        # # Manually apply theorems in correct order.
        # theorem_application_sequence = [theorems[0], theorems[1], theorems[4],
        #                                 theorems[2]]
        # for i, t in enumerate(theorem_application_sequence):
        #     modus_ponenses = base_graph.find_all_possible_modus_ponens(t)
        #     # base_graph.colorize_subgraph(
        #     #     modus_ponenses[0].actual_implication.get_top_level_implication_subgraphs()[0])
        #     modus_ponenses[0].actual_implication.visualize(f"Theorem Application #{i+1}")
        #     base_graph.visualize(f"Graph before applying theorem #{i + 1}", show_colorization=False)
        #     base_graph.clear_colorization()
        #     base_graph.apply_modus_ponens(modus_ponenses[0])
        #
        # base_graph.visualize("Final graph")
