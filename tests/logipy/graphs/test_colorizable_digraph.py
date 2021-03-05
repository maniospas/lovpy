import unittest

from logipy.graphs.colorizable_digraph import ColorizableDiGraph


class TestColorizableDiGraph(unittest.TestCase):

    def test_disconnect_fully_colorized_sub_dag_with_simple_5_node_tree_removing_1_branch(self):
        graph = ColorizableDiGraph()
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 3), (1, 4), (2, 5), (2, 6)
        ])
        graph.colorize_path([(0, 1), (1, 3)])
        graph.build_colorization_scheme()

        graph.disconnect_fully_colorized_sub_dag()

        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(0, 2))
        self.assertTrue(graph.has_edge(1, 4))
        self.assertTrue(graph.has_edge(2, 5))
        self.assertTrue(graph.has_edge(2, 6))

        self.assertFalse(graph.has_edge(1, 3))

    def test_disconnect_fully_colorized_sub_dag_with_12_node_dag(self):
        graph = ColorizableDiGraph()
        edges = [
            (0, 1), (0, 8),
            (1, 2), (1, 7),
            (2, 3), (2, 5),
            (3, 4),
            (4, 6), (4, 7),
            (5, 6), (5, 8),
            (6, 9), (6, 10),
            (7, 10), (7, 11),
            (8, 10), (8, 12)
        ]
        graph.add_edges_from(edges)
        graph.colorize_path([(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9)])
        graph.colorize_path([(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10)])
        graph.colorize_path([(0, 8), (8, 10)])
        graph.colorize_path([(0, 8), (8, 12)])
        graph.build_colorization_scheme()

        graph.disconnect_fully_colorized_sub_dag()

        edges.remove((4, 6))
        edges.remove((0, 8))
        for e in edges:
            self.assertTrue(graph.has_edge(e[0], e[1]))

        self.assertFalse(graph.has_edge(4, 6))
        self.assertFalse(graph.has_edge(0, 8))

    def test_disconnect_fully_colorized_sub_dag_with_8_node_dag(self):
        graph = ColorizableDiGraph()
        edges = [
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6),
            (5, 8), (5, 7),
            (6, 4),
            (7, 8),
            (8, 4)
        ]
        graph.add_edges_from(edges)
        graph.colorize_path([(0, 2), (2, 5), (5, 8), (8, 4)])
        graph.colorize_path([(0, 2), (2, 6), (6, 4)])
        graph.build_colorization_scheme()

        graph.disconnect_fully_colorized_sub_dag()

        edges.remove((2, 6))
        edges.remove((6, 4))
        edges.remove((5, 8))
        for e in edges:
            self.assertTrue(graph.has_edge(e[0], e[1]))

        self.assertFalse(graph.has_edge(2, 6))
        self.assertFalse(graph.has_edge(6, 4))
        self.assertFalse(graph.has_edge(5, 8))

    def test_colorize_path(self):
        graph = ColorizableDiGraph()
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 3), (1, 4), (2, 5), (2, 6)
        ])
        graph.colorize_path([(0, 1), (1, 3)])

        self.assertTrue(graph.is_edge_colorized(0, 1))
        self.assertTrue(graph.is_edge_colorized(1, 3))

    def test_out_colorize_nodes_with_simple_5_node_tree(self):
        graph = ColorizableDiGraph()
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 3), (1, 4), (2, 5), (2, 6)
        ])
        graph.colorize_path([(0, 1), (1, 3)])
        graph.out_colorize_nodes()

        self.assertTrue(graph.is_node_out_colorized(3))
        self.assertTrue(graph.is_node_out_colorized(4))
        self.assertTrue(graph.is_node_out_colorized(5))
        self.assertTrue(graph.is_node_out_colorized(6))

    def test_out_colorize_nodes_with_8_nodes_dag(self):
        graph = ColorizableDiGraph()
        edges = [
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6),
            (5, 8), (5, 7),
            (6, 4),
            (7, 8),
            (8, 4)
        ]
        graph.add_edges_from(edges)
        graph.colorize_path([(0, 2), (2, 5), (5, 8), (8, 4)])
        graph.colorize_path([(0, 2), (2, 6), (6, 4)])
        graph.out_colorize_nodes()

        self.assertTrue(graph.is_node_out_colorized(3))
        self.assertTrue(graph.is_node_out_colorized(4))
        self.assertTrue(graph.is_node_out_colorized(8))
        self.assertTrue(graph.is_node_out_colorized(6))

    def test_in_colorize_nodes_with_simple_5_node_tree(self):
        graph = ColorizableDiGraph()
        graph.add_edges_from([
            (0, 1), (0, 2),
            (1, 3), (1, 4), (2, 5), (2, 6)
        ])
        graph.colorize_path([(0, 1), (1, 3)])
        graph.in_colorize_nodes()

        self.assertTrue(graph.is_node_in_colorized(0))
        self.assertTrue(graph.is_node_in_colorized(1))
        self.assertTrue(graph.is_node_in_colorized(3))

    def test_in_colorize_nodes_with_8_nodes_dag(self):
        graph = ColorizableDiGraph()
        edges = [
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6),
            (5, 8), (5, 7),
            (6, 4),
            (7, 8),
            (8, 4)
        ]
        graph.add_edges_from(edges)
        graph.colorize_path([(0, 2), (2, 5), (5, 8), (8, 4)])
        graph.colorize_path([(0, 2), (2, 6), (6, 4)])
        graph.in_colorize_nodes()

        self.assertTrue(graph.is_node_in_colorized(0))
        self.assertTrue(graph.is_node_in_colorized(2))
        self.assertTrue(graph.is_node_in_colorized(5))
        self.assertTrue(graph.is_node_in_colorized(6))
