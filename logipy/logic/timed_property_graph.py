import itertools
import logging
import copy

import networkx
from networkx.readwrite.graphml import write_graphml
from networkx.algorithms.simple_paths import all_simple_edge_paths
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from networkx.drawing.nx_pylab import draw_networkx, draw_networkx_edge_labels, draw_networkx_edges
from matplotlib import pyplot as plt

from logipy.graphs.colorizable_digraph import ColorizableDiGraph
from logipy.logic.logical_operators import *
from logipy.logic.timestamps import *
from logipy.monitor.time_source import get_global_time_source


TIMESTAMP_PROPERTY_NAME = "timestamp"
IMPLICATION_PROPERTY_NAME = "implication"
TASK_PROPERTY_NAME = "task_property"
ASSUMPTION_GRAPH = "assumption"
CONCLUSION_GRAPH = "conclusion"

LOGGER_NAME = "logipy.logic.timed_property_graph"


class TimedPath:
    def __init__(self, path):
        self.timestamp = find_path_timestamp(path)
        self.edges = [(e[0], e[1]) for e in path]

    def get_timestamp(self):
        return self.timestamp

    def get_edges(self):
        return self.edges


class TimedPropertyGraph:

    class ModusPonensApplication:
        def __init__(self, graph, implication_graph, matching_paths, matching_paths_timestamps):
            self.graph = graph  # The graph on which modus ponens will be applied.
            self.implication_graph = implication_graph  # A --> B property graph.
            self.matching_paths = matching_paths  # Paths on graph that match A.
            self.matching_paths_timestamps = matching_paths_timestamps

    def __init__(self, time_source=get_global_time_source()):
        self.graph = ColorizableDiGraph()
        self.root_node = None
        self.time_source = time_source
        self.property_textual_representation = None

    def logical_and(self, property_graph, timestamp=None):
        if property_graph.graph.number_of_nodes() == 0:
            # Nothing to do if given graph is empty.
            return

        was_empty = self.graph.number_of_nodes() == 0

        timestamp1 = timestamp
        timestamp2 = timestamp

        if not timestamp:
            timestamp1 = self.get_most_recent_timestamp()
            timestamp2 = property_graph.get_most_recent_timestamp()
        if isinstance(timestamp1, RelativeTimestamp):
            timestamp1.set_time_source(self.time_source)
        if isinstance(timestamp2, RelativeTimestamp):
            timestamp2.set_time_source(self.time_source)

        self.graph.add_edges_from(property_graph.graph.edges(data=True))
        if not was_empty:
            # TODO: Implement recursive naming.
            and_node = AndOperator(self.get_root_node(), property_graph.get_root_node())
            self._add_edge(and_node, self.get_root_node(), {TIMESTAMP_PROPERTY_NAME: timestamp1})
            self._add_edge(and_node, property_graph.get_root_node(),
                           {TIMESTAMP_PROPERTY_NAME: timestamp2})
        else:
            self.root_node = property_graph.get_root_node()

    def logical_not(self, timestamp=None):
        if not timestamp:
            timestamp = self.get_most_recent_timestamp()
        if timestamp and isinstance(timestamp, RelativeTimestamp):
            timestamp.set_time_source(self.time_source)
        # TODO: Implement recursive naming.
        not_node = NotOperator(self.get_root_node())
        self._add_edge(not_node, self.get_root_node(), {TIMESTAMP_PROPERTY_NAME: timestamp})
        self._fix_orphan_logical_operators()

    def logical_implication(self, property_graph, timestamp=None):
        if not self.get_root_node():
            raise Exception("Implication cannot be performed with an empty assumption.")
        if not property_graph.get_root_node():
            raise Exception("Implication cannot be performed with an empty conclusion.")

        # TODO: Implement recursive naming.
        impl_node = ImplicationOperator(self.get_root_node(), property_graph.get_root_node())

        if not timestamp:
            assumption_timestamp = self.get_most_recent_timestamp()
            conclusion_timestamp = property_graph.get_most_recent_timestamp()
        else:
            timestamp.set_time_source(self.time_source)
            assumption_timestamp = timestamp
            conclusion_timestamp = timestamp

        self.graph.add_edges_from(property_graph.graph.edges(data=True))
        self._add_edge(impl_node, self.get_root_node(),
                       {TIMESTAMP_PROPERTY_NAME: assumption_timestamp,
                       IMPLICATION_PROPERTY_NAME: ASSUMPTION_GRAPH})
        self._add_edge(impl_node, property_graph.get_root_node(),
                       {TIMESTAMP_PROPERTY_NAME: conclusion_timestamp,
                       IMPLICATION_PROPERTY_NAME: CONCLUSION_GRAPH})

    def apply_modus_ponens(self, modus_ponens):
        """Applies given modus ponens operation on current graph.

        In order to apply a modus ponens operation, it should have been generated
        using current graph, for example by utilizing find_all_possible_modus_ponens()
        method.

        :param modus_ponens: A TimedPropertyGraph.ModusPonensApplication object that has
                been generated by current graph.
        """
        matching_paths = modus_ponens.matching_paths
        matching_timestamps = modus_ponens.matching_paths_timestamps

        assumption, conclusion = \
            modus_ponens.implication_graph.get_top_level_implication_subgraphs()
        assumption_timestamp = max(matching_timestamps)  # first moment assumption holds

        # Remove assumption from the graph.
        self._logically_remove_path_set(matching_paths)

        # Add conclusion as an unconnected component.
        for edge in conclusion.get_graph().edges(data=TIMESTAMP_PROPERTY_NAME):
            # TODO: Consider if it is required to assign timestamps according to relative ones.
            # Replace relative timestamps with the absolute timestamp when assumption
            # firstly holds.
            timestamp = edge[2]
            if not timestamp.is_absolute():
                timestamp = assumption_timestamp
            self._add_edge(edge[0], edge[1], {TIMESTAMP_PROPERTY_NAME: timestamp})

        # Intervene an AND node to connect the unconnected component of conclusion.
        deeper_common_node = self._find_deeper_common_node_in_graph(matching_paths)
        if deeper_common_node:
            and_node = AndOperator(deeper_common_node, conclusion.get_root_node())
            and_node.disable_hashing_by_structure()
            # Move all incoming edges from deeper common node to the new AND node.
            predecessors = list(self.graph.predecessors(deeper_common_node))
            for predecessor in predecessors:
                t = self.graph.edges[predecessor, deeper_common_node][TIMESTAMP_PROPERTY_NAME]
                self.graph.remove_edge(predecessor, deeper_common_node)
                self._add_edge(predecessor, and_node, {TIMESTAMP_PROPERTY_NAME: t})
            # Connect the new AND node with deeper common node and
            # the unconnected conclusion component.
            old_part_timestamp = self._find_subgraph_most_recent_timestamp(deeper_common_node)
            new_part_timestamp = self._find_subgraph_most_recent_timestamp(conclusion.get_root_node())
            self._add_edge(and_node, deeper_common_node,
                           {TIMESTAMP_PROPERTY_NAME: old_part_timestamp})
            self._add_edge(and_node, conclusion.get_root_node(),
                           {TIMESTAMP_PROPERTY_NAME: new_part_timestamp})

            self._fix_orphan_logical_operators()

        # # Fix matching paths to include newly added and node (if really added).
        # for p in matching_paths:
        #     for i in range(len(p)):
        #         # Intervene the newly added and node between upper common node and its
        #         # predecessors.
        #         edge = p[i]
        #         if edge[0] == deeper_common_node:
        #             p.insert(i, (and_node, edge[0], and_timestamp))
        #             if i > 0:
        #                 previous_edge = p.pop(i-1)
        #                 p.insert(i-1, (previous_edge[0], and_node, and_timestamp))
        #             break

    def find_all_possible_modus_ponens(self, implication_graph):
        assumption, conclusion = implication_graph.get_top_level_implication_subgraphs()
        matching_cases, _, _, cases_timestamps = \
            self.find_equivalent_subgraphs(assumption)

        possible_modus_ponens = []

        for i in range(len(matching_cases)):
            new_modus_ponen = TimedPropertyGraph.ModusPonensApplication(
                self, implication_graph, matching_cases[i], cases_timestamps[i]
            )
            possible_modus_ponens.append(new_modus_ponen)

        return possible_modus_ponens

    def set_timestamp(self, timestamp):
        """Sets given timestamp, as the timestamp of all edges of the graph.

        Set timestamp should not be used on a property graph after it has been used
        as an operand on a logical operation with another graph.
        """
        if not isinstance(timestamp, Timestamp):
            raise Exception("Only instances of Timestamp and its subclasses are allowed.")

        # Current implementation provides a single time source for all relative timestamps.
        if isinstance(timestamp, RelativeTimestamp):
            timestamp.set_time_source(self.time_source)

        for u, v in self.graph.edges():
            self.graph[u][v].update({TIMESTAMP_PROPERTY_NAME: timestamp})

    def is_uniform_timestamped(self, timestamp=None):
        """Checks whether current graph contains only timestamps that match.

        :param timestamp: If this argument is provided, then in order to consider the
                graph as uniform timestamped, all timestamps should additionally match
                given timestamp.

        :returns: True if graph is uniform timestamped, otherwise False.
        """
        edges = list(self.graph.edges(data=TIMESTAMP_PROPERTY_NAME))
        if not timestamp:
            timestamp = edges[0][2]
        for edge in edges:
            if timestamp.is_absolute() != edge[2].is_absolute():
                # All timestamps should either be absolute or relative.
                return False

            if timestamp.is_absolute() and (
                    timestamp.get_absolute_value() != edge[2].get_absolute_value()):
                # Absolute timestamps should have exact the same value.
                return False
            elif not timestamp.is_absolute() and timestamp != edge[2]:
                # Relative timestamps should have a common interval.
                return False
        return True

    def get_root_node(self):
        return self.root_node

    def set_time_source(self, time_source):
        self.time_source = time_source
        for edge in self.graph.edges(data=TIMESTAMP_PROPERTY_NAME):
            if isinstance(edge[2], RelativeTimestamp):
                edge[2].set_time_source(time_source)

    def get_most_recent_timestamp(self):
        timestamps = [e[2] for e in self.get_graph().edges(data=TIMESTAMP_PROPERTY_NAME)]
        return max(timestamps) if timestamps else None

    def get_top_level_implication_subgraphs(self):
        assumption = None
        conclusion = None

        if isinstance(self.root_node, ImplicationOperator):
            root_edges = list(self.graph.edges(self.root_node, data=IMPLICATION_PROPERTY_NAME))
            for edge in root_edges:
                if edge[2] == ASSUMPTION_GRAPH:
                    assumption = self.graph.subgraph(networkx.dfs_postorder_nodes(
                        self.graph, edge[1]))
                elif edge[2] == CONCLUSION_GRAPH:
                    conclusion = self.graph.subgraph(networkx.dfs_postorder_nodes(
                        self.graph, edge[1]))

        return self._inflate_property_graph_from_subgraph(assumption), \
            self._inflate_property_graph_from_subgraph(conclusion)

    def remove_subgraph(self, subgraph):
        # TODO: Implement using find_equivalent_subgraphs()
        _, matching_groups, found = self._find_equivalent_path_structure(subgraph)
        self._logically_remove_path_set([g[0] for g in matching_groups])
        self._fix_orphan_logical_operators()

    def contains_property_graph(self, property_graph):
        matching_cases, _, _, _ = self.find_equivalent_subgraphs(property_graph)
        return bool(matching_cases)

        # property_leaves = _get_leaf_nodes(property_graph)
        #
        # # Start from the leaves in property graph and make sure they exist in current graph.
        # for property_leaf in property_leaves:
        #     if not self.get_graph().has_node(property_leaf):
        #         return False
        #
        # # Make sure that for every path from a leaf node to the root node in property
        # # graph, there exists an equivalent and time-matching path from a leaf node
        # # to the root node in current graph. Equivalent means that between two
        # # connected nodes in property graph, whose depth differs by 1, only AND nodes
        # # can be inserted in current graph.
        # for property_leaf in property_leaves:
        #     _, _, matched = self.find_time_matching_paths_from_node_to_root(
        #         property_leaf, property_graph, property_leaf)
        #     if not matched:
        #         return False
        #
        # return True

    def export_to_graphml_file(self, path):
        write_graphml(self.get_graph(), path)

    def get_graph(self):
        return self.graph

    def get_copy(self):
        # TODO: Fix copying for subclasses with arguments in __init__.
        copy_obj = type(self)()
        copy_obj.graph = self.graph.copy()
        copy_obj.root_node = self.root_node  # Node references remain the same.
        copy_obj.time_source = self.time_source
        copy_obj.property_textual_representation = self.property_textual_representation
        return copy_obj

    def get_property_textual_representation(self):
        return self.property_textual_representation if self.property_textual_representation else ""

    def set_property_textual_representation(self, textual_representation):
        self.property_textual_representation = textual_representation

    def get_leaves(self):
        return [n for n, d in self.graph.out_degree() if d == 0]

    def get_present_time_subgraph(self):
        present_time_edges = [(e[0], e[1]) for e in self.graph.edges(data=TIMESTAMP_PROPERTY_NAME)
                              if e[2].matches(Timestamp(self.time_source.get_current_time()))]
        subgraph = self.graph.edge_subgraph(present_time_edges).copy()
        # TODO: Fix subgraph by removing AND nodes with single out edge and adjacent NOT nodes.
        present_time_graph = self._inflate_property_graph_from_subgraph(subgraph)
        if present_time_graph:
            present_time_graph._fix_orphan_logical_operators()
        return present_time_graph

    def is_implication_graph(self):
        return isinstance(self.root_node, ImplicationOperator) and \
               self.graph.out_degree(self.root_node) > 1

    def visualize(self, title="", show_colorization=False):
        plt.figure(num=None, figsize=(18, 18), dpi=80, facecolor='w', edgecolor='w')
        plt.title(title, fontsize=22, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        pos = networkx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        edge_labels = {
            e: str(self.graph.edges[e[0], e[1]][TIMESTAMP_PROPERTY_NAME])
            for e in self.graph.edges
        }
        edge_colors = {
            e: 'red' if self.graph.is_edge_colorized(e[0], e[1]) else 'black'
            for e in self.graph.edges
        }
        node_labels = {
            n: n.get_operator_symbol() if isinstance(n, LogicalOperator) else str(n)
            for n in self.graph.nodes
        }
        node_colors = []
        for n in self.graph.nodes:
            if show_colorization:
                if self.graph.is_node_in_colorized(n) and self.graph.is_node_out_colorized(n):
                    node_colors.append('red')
                elif self.graph.is_node_in_colorized(n):
                    node_colors.append('orange')
                elif self.graph.is_node_out_colorized(n):
                    node_colors.append('purple')
                else:
                    node_colors.append('teal')
            else:
                node_colors.append('teal')

        draw_networkx(self.graph, pos=pos, font_size=18, node_size=5000, node_color=node_colors,
                      labels=node_labels, font_weight='bold')
        if show_colorization:
            draw_networkx_edges(self.graph, pos=pos, edgelist=list(edge_colors.keys()),
                                edge_color=list(edge_colors.values()))
        draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels,
                                  font_size=18, font_color='red')
        plt.show()

    def find_equivalent_subgraphs(self, other):
        matched_paths, matching_groups, found = self._find_equivalent_path_structure(other)
        if not found:
            return [], [], [], []

        original_timestamps = [find_path_timestamp(p) for p in matched_paths]
        groups_timestamps = [[find_path_timestamp(p) for p in group] for group in matching_groups]
        cases = list(itertools.product(*matching_groups))  # all subgraphs that match other graph
        cases_timestamps = list(itertools.product(*groups_timestamps))
        sorted_by_original_timestamps = list(zip(*sorted(zip(
            original_timestamps, matched_paths, *cases_timestamps, *cases), key=lambda row: row[0]
        )))
        original_timestamps = sorted_by_original_timestamps[0]
        matched_paths = sorted_by_original_timestamps[1]
        cases_timestamps = sorted_by_original_timestamps[2:len(cases)+2]
        cases = sorted_by_original_timestamps[len(cases)+2:]

        cases_to_remove = set()

        for i in range(len(cases)):
            case = cases[i]
            case_timestamps = cases_timestamps[i]

            if not timestamp_sequences_matches(original_timestamps, case_timestamps):
                cases_to_remove.add(i)

        matching_cases = [cases[i] for i in range(len(cases)) if i not in cases_to_remove]
        matching_cases_timestamps = [cases_timestamps[i] for i in range(len(cases))
                                     if i not in cases_to_remove]

        return matching_cases, matched_paths, original_timestamps, matching_cases_timestamps

    def switch_implication_parts(self):
        """Switches assumption with conclusion and vice-versa."""
        if not isinstance(self.get_root_node(), ImplicationOperator):
            raise RuntimeError("Cannot switch implication parts on non implication graph.")

        assumption_edge, conclusion_edge = self._get_top_level_implication_edges()
        self.graph.edges[
            assumption_edge[0], assumption_edge[1]][IMPLICATION_PROPERTY_NAME] = CONCLUSION_GRAPH
        self.graph.edges[
            conclusion_edge[0], conclusion_edge[1]][IMPLICATION_PROPERTY_NAME] = ASSUMPTION_GRAPH

    def get_basic_predicates(self):
        """Returns the basic predicates that form the graph.

        :return: A sequence of all basic predicates as PredicateGraph objects.
        """
        basic_predicates = []
        predicate_nodes = [n for n in self.graph.nodes if isinstance(n, PredicateNode)]

        for n in predicate_nodes:
            paths_to_predicate = all_simple_edge_paths(self.graph, self.get_root_node(), n)

            # Each path to a predicate defines a basic predicate graph.
            for p in paths_to_predicate:
                predicate_graph = self.get_copy()
                predicate_graph._retain_only_edges_that_starts_with(p)
                predicate_graph._fix_orphan_logical_operators()
                basic_predicates.append(predicate_graph)

        return basic_predicates

    def get_all_paths(self):
        """Returns all paths from root node to leaf nodes.

        :return: A list of paths in the form of TimestampedPath objects.
        """
        leaf_nodes = _get_leaf_nodes(self)
        paths = all_simple_edge_paths(self.graph, self.get_root_node(), leaf_nodes)
        return [TimestampedPath(p, self.find_path_timestamp(p)) for p in paths]

    def update_subgraph_timestamp(self, subgraph, new_timestamp):
        """Sets timestamps of given subgraph to the given timestamp.

        Update happens only on the first occurrence of subgraph. The rest remain intact.

        If no occurrence of given subgraph can be found, a RuntimeError is raised.

        :param subgraph: A TimedPropertyGraph to be matched into current graph.
        :param new_timestamp: A Timestamp to be set on the occurrence of subgraph.
        """
        matching_cases, _, _, _ = self.find_equivalent_subgraphs(subgraph)
        if len(matching_cases) > 1:
            logger = logging.getLogger(LOGGER_NAME)
            logger.warning("More than a single case found while updating subgraph timestamp.")
        elif not matching_cases:
            raise RuntimeError("Failed to update timestamps of given subgraph: subgraph not found.")

        case_to_update = matching_cases[0]
        for path in case_to_update:
            self.update_path_timestamp(path, new_timestamp)

    def update_path_timestamp(self, path, new_timestamp):
        """Sets timestamps of given path to given timestamp

        :param path: A path represented as a sequence of edges represented as two-tuples.
        :param new_timestamp: The new timestamp to be set on given path.
        """
        for e in path:
            # TODO: Consider if timestamp copying is needed.
            self.graph.edges[e[0], e[1]][TIMESTAMP_PROPERTY_NAME] = new_timestamp

    def find_path_timestamp(self, path):
        """Returns the timestamp of a path.

        The timestamp of a path is considered to be the oldest timestamp of its edges.

        :param path: A path represented as an iterable of the edges belonging to the path.
                Each edge is represented as a two-tuple, containing the source node as first
                item and the target node as second item.

        :return: The timestamp of the path in the form of a Timestamp object.
        """
        return min([self.graph.edges[e[0], e[1]].get(TIMESTAMP_PROPERTY_NAME) for e in path])

    def _get_top_level_implication_edges(self):
        assumption_edge = None
        conclusion_edge = None
        root_edges = list(self.graph.edges(self.get_root_node(), data=True))

        for e in root_edges:
            edge_data = e[2]
            if edge_data.get(IMPLICATION_PROPERTY_NAME, None) == ASSUMPTION_GRAPH:
                assumption_edge = e
            elif edge_data.get(IMPLICATION_PROPERTY_NAME, None) == CONCLUSION_GRAPH:
                conclusion_edge = e

        return assumption_edge, conclusion_edge

    def _add_node(self, node):
        self.graph.add_node(node)
        if self.root_node is None:
            self.root_node = node

    def _add_edge(self, start_node, end_node, data_dict=dict()):
        data_dict = {k: v for k, v in data_dict.items() if v}  # Remove None arguments.
        self.graph.add_edge(start_node, end_node, **data_dict)
        if self.get_root_node() is None or end_node == self.get_root_node():
            self.root_node = start_node

    def _inflate_property_graph_from_subgraph(self, subgraph):
        if len(subgraph) == 0:
            return None
        property_graph = TimedPropertyGraph()
        property_graph.graph = subgraph
        property_graph.time_source = self.time_source
        property_graph.property_textual_representation = self.property_textual_representation
        for node_in_degree in subgraph.in_degree():
            if node_in_degree[1] == 0:
                property_graph.root_node = node_in_degree[0]
        if not property_graph.root_node:
            raise Exception("Provided subgraph doesn't contain any root node.")
        return property_graph

    def _find_equivalent_path_structure(self, other):
        """Finds all paths in current graph that combined logically forms the other graph.

        If a logical matching subgraph of other graph is found in current graph, returns all
        paths in current graph, that when combined together, logically forms the other graph.

        Parameters:
            :other: A TimedPropertyGraph object to be logically matched in current object.

        Returns:
            :matched_paths: A list of all the paths contained in other graph, from node to root
                    when a logically matching subgraph of other graph has been found
                    in current graph, else an empty list.
            :matching_groups: A list containing a list of logically matching paths in current
                    graph, for every path contained in matched_paths upon success, otherwise
                    an empty list.
            :found: True when a logically matching subgraph of other graph has been found
                    in current graph, otherwise False.
        """

        # Check that all leaves of other graph, are also leaves of current one.
        other_leaves = other.get_leaves()
        for leaf in other_leaves:
            if not self.get_graph().has_node(leaf):
                return [], [], False

        # Obtain all simple paths from root to leaves in other graph.
        other_paths = all_simple_edge_paths(other.get_graph(), other.get_root_node(), other_leaves)
        # Networkx doesn't include edge data, so manually add timestamps.
        other_paths = _fill_paths_with_timestamps(other.get_graph(), other_paths)

        # Obtain all simple paths from root to leaves of other graph in current graph.
        current_paths = all_simple_edge_paths(self.get_graph(), self.get_root_node(), other_leaves)
        current_paths = list(_fill_paths_with_timestamps(self.get_graph(), current_paths))

        # Find the paths in current graph that logically matches every path of other graph.
        matched_paths = []
        matching_groups = []
        for other_path in other_paths:
            matchings_paths = []
            for cur_path in current_paths:
                if _paths_logically_match(cur_path, other_path):
                    matchings_paths.append(cur_path)
            if not matchings_paths:
                return [], [], False
            matched_paths.append(other_path)
            matching_groups.append(matchings_paths)

        return matched_paths, matching_groups, True

    def _logically_remove_path_set(self, paths):
        for p in paths:
            self.graph.colorize_path(p)
        self.graph.build_colorization_scheme()
        # self.visualize(title="Graph before performing removal", show_colorization=True)
        self.graph.disconnect_fully_colorized_sub_dag()
        self.graph.clear_colorization()

    def _find_deeper_common_node_in_graph(self, paths):
        """Returns the deeper common node of all paths, that also belongs in current graph."""
        last_common_node = paths[0][0][0]  # Consider top node of all paths to be the same.
        if not self.graph.has_node(last_common_node):
            return None  # Either graph is empty or path belongs to a different graph.

        for i in range(min([len(p) for p in paths])):
            edge = paths[0][i]  # Use the first path as a guideline.
            for path in paths:
                if not _edges_match(edge, path[i]):
                    break
            else:
                if self.graph.has_node(paths[0][i][1]):
                    last_common_node = paths[0][i][1]
                    continue
            break
        return last_common_node

    def _find_subgraph_most_recent_timestamp(self, source):
        return max([self.graph.edges[e[0], e[1]][TIMESTAMP_PROPERTY_NAME]
                    for e in bfs_edges(self.graph, source)])

    def _get_assumption_conclusion_edges_timestamps(self):
        """Returns the timestamps of top-level edges to assumption, conclusion subgraphs."""
        assumption_timestamp = None
        conclusion_timestamp = None

        if isinstance(self.get_root_node(), ImplicationOperator):
            root_edges = list(self.graph.edges(self.get_root_node(), data=True))
            for e in root_edges:
                edge_data = e[2]
                timestamp = edge_data.get(TIMESTAMP_PROPERTY_NAME, None)
                if edge_data.get(IMPLICATION_PROPERTY_NAME, None) == ASSUMPTION_GRAPH:
                    assumption_timestamp = timestamp
                elif edge_data.get(IMPLICATION_PROPERTY_NAME, None) == CONCLUSION_GRAPH:
                    conclusion_timestamp = timestamp

        return assumption_timestamp, conclusion_timestamp

    def _retain_only_edges_that_starts_with(self, path_prefix):
        """Retains only the edges belonging to a path that starts with given prefix.

        All edges that do not belong to such a prefixed path are removed from the graph,
        along with orphan nodes.

        :param path_prefix: A path prefix represented as a sequence of edges in the form
                of two-tuples.
        """
        final_prefix_node = path_prefix[-1][1]
        leaf_nodes = _get_leaf_nodes(self)
        suffix_paths = all_simple_edge_paths(self.graph, final_prefix_node, leaf_nodes)
        all_edges_to_retain = list(path_prefix)
        for p in suffix_paths:
            all_edges_to_retain.extend(p)

        edges_to_remove = [e for e in self.graph.edges if not _edge_in_set(all_edges_to_retain, e)]
        self.graph.remove_edges_from(edges_to_remove)

        nodes_to_remove = [n for n in self.graph.nodes if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(nodes_to_remove)

        self._fix_orphan_logical_operators()

    def _fix_orphan_logical_operators(self):
        """Cleans the graph from orphan AND and NOT operators."""
        self._remove_orphan_and_operators()
        self._collapse_sequential_not_operators()

    def _remove_orphan_and_operators(self):
        and_nodes_to_remove = []
        for n in self.graph.nodes:
            if isinstance(n, AndOperator) and self.graph.out_degree(n) < 2:
                and_nodes_to_remove.append(n)
        for n in and_nodes_to_remove:
            self._remove_orphan_and_operator(n)

    def _remove_orphan_and_operator(self, and_node):
        predecessors = list(self.graph.predecessors(and_node))
        successor = list(self.graph.successors(and_node))[0]
        successor_data = self.graph.edges[and_node, successor]

        if predecessors:  # AND node is not the root node.
            for predecessor in predecessors:
                predecessor_data = self.graph.edges[predecessor, and_node]
                # Retain data from the edge with the older timestamp.
                if (predecessor_data[TIMESTAMP_PROPERTY_NAME] <
                        successor_data[TIMESTAMP_PROPERTY_NAME]):
                    data_to_retain = successor_data | predecessor_data
                else:
                    data_to_retain = predecessor_data | successor_data
                self._add_edge(predecessor, successor, data_to_retain)
        else:  # AND node is the root node.
            self.root_node = successor
            # Since no new edge is added, propagate data with older timestamp to deeper nodes.
            self._propagate_edge_data_to_deeper_edges(successor, successor_data)

        self.graph.remove_node(and_node)

    def _collapse_sequential_not_operators(self):
        more_not_pairs_to_remove = True
        while more_not_pairs_to_remove:
            more_not_pairs_to_remove = False
            for e in self.graph.edges:
                if isinstance(e[0], NotOperator) and isinstance(e[1], NotOperator):
                    self._collapse_not_operators_pair(e[0], e[1])
                    more_not_pairs_to_remove = True
                    break

    def _collapse_not_operators_pair(self, not1, not2):
        not1_predecessors = list(self.graph.predecessors(not1))
        not2_successor = list(self.graph.successors(not2))[0]

        # Between the data of out edge of second NOT and the data of the edge between the
        # two NOT nodes, keep the ones whose timestamp is older.
        between_data = self.graph.edges[not1, not2]
        not2_out_data = self.graph.edges[not2, not2_successor]
        if between_data[TIMESTAMP_PROPERTY_NAME] < not2_out_data[TIMESTAMP_PROPERTY_NAME]:
            data_to_retain = not2_out_data | between_data
        else:
            data_to_retain = between_data | not2_out_data

        if not1_predecessors:  # First NOT is not the root node.
            for predecessor in not1_predecessors:
                predecessor_data = self.graph.edges[predecessor, not1]
                if (predecessor_data[TIMESTAMP_PROPERTY_NAME] <
                        data_to_retain[TIMESTAMP_PROPERTY_NAME]):
                    new_data = data_to_retain | predecessor_data
                else:
                    new_data = predecessor_data | data_to_retain
                self._add_edge(predecessor, not2_successor, new_data)
        else:  # First NOT is the root node.
            self.root_node = not2_successor
            self._propagate_edge_data_to_deeper_edges(not2_successor, data_to_retain)

        self.graph.remove_nodes_from([not1, not2])

    def _propagate_edge_data_to_deeper_edges(self, source_node, data):
        for deeper_node in self.graph.successors(source_node):
            deeper_node_data = self.graph.edges[source_node, deeper_node]
            if (data[TIMESTAMP_PROPERTY_NAME] < deeper_node_data[TIMESTAMP_PROPERTY_NAME]):
                data_to_retain = deeper_node_data | data
            else:
                data_to_retain = data | deeper_node_data
            self._add_edge(source_node, deeper_node, data_to_retain)  # just for updating data

    def _clean_inverse_parallel_paths(self):
        pass  # TODO: Implement

    # def find_time_matching_paths_from_node_to_root(self, start_node, other_graph, other_start_node):
    #     matched_paths, matching_groups, found = self.find_equivalent_paths_from_node_to_root(
    #         start_node, other_graph, other_start_node)
    #
    #     # Check that for every matched path, there is at least one with matching timestamps.
    #     for i in range(len(matched_paths)):
    #         matched_path = matched_paths[i]
    #         matching_paths = matching_groups[i]
    #         matched_path_timestamp = _find_path_timestamp(matched_path)
    #
    #         # TODO: Implement time matching for timesources different than current one.
    #         # for matching_path in matching_paths:
    #         #     if not _find_path_timestamp(matching_path).matches(matched_path_timestamp):
    #         #         matching_paths.remove(matching_path)
    #
    #         if not matching_paths:
    #             matched_paths.remove(matched_path)
    #
    #     return matched_paths, matching_groups, bool(matched_paths)
    #
    # def find_equivalent_paths_from_node_to_root(self, start_node, other_graph, other_start_node):
    #     # TODO: Reimplement this method in a more elegant way.
    #     paths_to_upper_non_and_other_nodes = _find_path_to_upper_non_and_nodes(
    #         other_graph, other_start_node)
    #     paths_to_upper_non_and_current_nodes = _find_path_to_upper_non_and_nodes(self, start_node)
    #
    #     # If there are still nodes in other graph to be validated, while current graph has
    #     # reached to root, then no matching paths has been found.
    #     if paths_to_upper_non_and_other_nodes and not paths_to_upper_non_and_current_nodes:
    #         return [], [], False
    #     # Also, if other graph has reached to root, while current graph still contains non and
    #     # node to be validated, then no matching paths has been found.
    #     elif not paths_to_upper_non_and_other_nodes and paths_to_upper_non_and_current_nodes:
    #         matching_paths = _find_clean_paths_to_root(self, start_node)
    #         matched_paths = []
    #         if matching_paths:
    #             matched_paths = _find_clean_paths_to_root(other_graph, other_start_node)
    #         return matched_paths, [matching_paths for p in matched_paths], bool(matched_paths)
    #     # If non-and upper paths are empty in both other and current graphs, then the requested
    #     # one has been validated.
    #     elif not paths_to_upper_non_and_other_nodes and not paths_to_upper_non_and_current_nodes:
    #         matched_paths = _find_clean_paths_to_root(other_graph, other_start_node)
    #         matching_paths = _find_clean_paths_to_root(self, start_node)
    #         return matched_paths, [matching_paths for p in matched_paths], True
    #
    #     matched_other_paths = []
    #     matching_current_path_groups = []
    #
    #     for other_upper_path in paths_to_upper_non_and_other_nodes:  # paths to be validated
    #         other_upper_path_matched = False
    #
    #         for current_upper_path in paths_to_upper_non_and_current_nodes:
    #
    #             other_upper_path_non_and_node = other_upper_path[-1][0]
    #             current_upper_path_non_and_node = current_upper_path[-1][0]
    #
    #             if (isinstance(other_upper_path_non_and_node, LogicalOperator) and
    #                 isinstance(current_upper_path_non_and_node, LogicalOperator) and
    #                 current_upper_path_non_and_node.logically_matches(
    #                         other_upper_path_non_and_node)) or (
    #                     other_upper_path_non_and_node == current_upper_path_non_and_node):
    #
    #                 matched_paths, matching_groups, found = \
    #                     self.find_equivalent_paths_from_node_to_root(
    #                         current_upper_path_non_and_node,
    #                         other_graph,
    #                         other_upper_path_non_and_node
    #                     )
    #
    #                 # The matched and matching paths should be prepended with the subpaths up
    #                 # to the node where search started.
    #                 if found:
    #                     other_upper_path_matched = True
    #                     if not matched_paths:
    #                         # Path returned empty, because successfully terminated to root node.
    #                         matched_other_paths.append(other_upper_path)
    #                         matching_current_path_groups.append([current_upper_path])
    #                     else:
    #                         for p in matched_paths:
    #                             matched_other_paths.append([*other_upper_path, *p])
    #                         for matching_group in matching_groups:
    #                             matching_current_paths = []
    #                             for p in matching_group:
    #                                 matching_current_paths.append([*current_upper_path, *p])
    #                             matching_current_path_groups.append(matching_current_paths)
    #         if not other_upper_path_matched:  # Not matching a single path, is enough to fail.
    #             return [], [], False
    #
    #     return matched_other_paths, matching_current_path_groups, bool(matched_other_paths)

    # def replace_subgraph(self, old_subgraph, new_subgraph):
    #     matching_cases, _, original_timestamps, cases_timestamps = \
    #         self.find_equivalent_subgraphs(old_subgraph)
    #     if not matching_cases:
    #         return False
    #
    #     all_matching_paths = matching_cases[0]  # TODO: pass selection to prover
    #     matching_timestamps = cases_timestamps[0]
    #
    #     # Find the upper node where all those paths connect.
    #     if len(all_matching_paths) > 1:
    #         for edge in all_matching_paths[0][::-1]:
    #             for path in all_matching_paths:
    #                 if not _edges_match(path[-1], edge):
    #                     break
    #             else:
    #                 for path in all_matching_paths:
    #                     path.remove(edge)
    #                     continue
    #             break
    #
    #     # Add the new subgraph as an unconnected component.
    #     if new_subgraph:
    #         old_subgraph_timestamp = max(matching_timestamps)  # first moment the structure holds
    #
    #         for edge in new_subgraph.get_graph().edges(data=TIMESTAMP_PROPERTY_NAME):
    #             # TODO: Consider if it is required to assign timestamps according to relative ones.
    #             # Replace relative timestamps with the absolute timestamp when old subgraph
    #             # firstly holds.
    #             timestamp = edge[2]
    #             if not timestamp.is_absolute():
    #                 timestamp = old_subgraph_timestamp
    #             self._add_edge(edge[0], edge[1], {TIMESTAMP_PROPERTY_NAME: timestamp})
    #
    #         # Intervene an AND node between the upper non and node of matching subgraph
    #         # and its predecessors.
    #         upper_common_node = all_matching_paths[0][-1][0]
    #         and_node = AndOperator(upper_common_node, new_subgraph.get_root_node())
    #         and_timestamp = max(
    #             *(e[2] for e in self.graph.out_edges(upper_common_node, data=TIMESTAMP_PROPERTY_NAME)))
    #         predecessors = list(self.graph.predecessors(upper_common_node))
    #         for predecessor in predecessors:
    #             t = self.graph.edges[predecessor, upper_common_node][TIMESTAMP_PROPERTY_NAME]
    #             self.graph.remove_edge(predecessor, upper_common_node)
    #             self._add_edge(predecessor, and_node, {TIMESTAMP_PROPERTY_NAME: t})
    #         self._add_edge(and_node, upper_common_node, {TIMESTAMP_PROPERTY_NAME: and_timestamp})
    #         self._add_edge(and_node, new_subgraph.get_root_node(),
    #                        {TIMESTAMP_PROPERTY_NAME: and_timestamp})
    #
    #     # Remove old edges and nodes that doesn't participate in any other path.
    #     #self._logically_remove_path_set(all_matching_paths)
    #
    #     return True


class PredicateGraph(TimedPropertyGraph):
    # TODO: Name predicate nodes using their children too, to not be treated equal.
    def __init__(self, predicate, *args):
        super().__init__()

        # Build predicate node first, so hash doesn't change. Implement it better later.
        self._predicate_node = PredicateNode(predicate)
        for arg in args:
            self._predicate_node.add_argument(arg)

        self._add_node(self._predicate_node)
        for arg in args:
            self._add_argument(arg)

    def get_copy(self):
        # TODO: Provide a more elegant fix.
        copy_obj = type(self)(self._predicate_node.predicate, *self._predicate_node.arguments)
        copy_obj.graph = self.graph.copy()
        copy_obj.root_node = self.root_node  # Node references remain the same.
        copy_obj.time_source = self.time_source
        copy_obj.property_textual_representation = self.property_textual_representation
        return copy_obj

    def _add_argument(self, argument):
        self._add_edge(self._predicate_node, argument)


class PredicateNode:
    def __init__(self, predicate):
        self.predicate = predicate
        self.arguments = []

    def add_argument(self, argument):
        self.arguments.append(argument)
        # self.arguments.sort()

    def __str__(self):
        str_repr = "{}({})".format(
            self.predicate.__str__(), ",".join([arg.__str__() for arg in self.arguments]))
        str_repr = str_repr.replace(" ", "_")
        return str_repr

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if isinstance(other, PredicateNode):
            return self.__str__() == other.__str__()
        else:
            return None

    def __repr__(self):
        return self.__str__()


class MonitoredVariable:
    # TODO: Make a registrar so monitored variables with the same name, are the same
    # object in memory too, also encasuplating the real variable.
    def __init__(self, monitored_variable):
        self.monitored_variable = monitored_variable

    def __str__(self):
        return self.monitored_variable

    def __hash__(self):
        return hash(self.monitored_variable)

    def __eq__(self, other):
        try:
            return self.monitored_variable == other.monitored_variable
        except AttributeError:
            return False

    def __repr__(self):
        return self.monitored_variable


class TimestampedPath:
    def __init__(self, path, timestamp):
        self.path = path
        self.timestamp = timestamp


def _get_leaf_nodes(property_graph):
    leaf_nodes = list()
    for node, deg in property_graph.get_graph().out_degree():
        if deg == 0:
            leaf_nodes.append(node)
    return leaf_nodes


def _find_path_to_upper_non_and_nodes(property_graph, start_node):
    graph = property_graph.get_graph()
    paths = []
    for in_edge in graph.in_edges(start_node, data=TIMESTAMP_PROPERTY_NAME):
        if not isinstance(in_edge[0], AndOperator):
            paths.append([in_edge])
        else:
            upper_paths = _find_path_to_upper_non_and_nodes(
                property_graph, in_edge[0])
            for p in upper_paths:
                paths.append([in_edge, *p])
    return paths


def _find_clean_paths_to_root(property_graph, start_node):
    graph = property_graph.get_graph()
    paths = []
    for in_edge in graph.in_edges(start_node, data=TIMESTAMP_PROPERTY_NAME):
        if isinstance(in_edge[0], AndOperator):
            upper_paths = _find_clean_paths_to_root(property_graph, in_edge[0])
            for p in upper_paths:
                paths.append([in_edge, *p])
    return paths


def find_path_timestamp(path):
    path_timestamp = path[0][2]
    for edge in path:
        path_timestamp = min(path_timestamp, edge[2])
    return path_timestamp


def _edges_match(e1, e2):
    if e1[0] != e2[0]:
        return False
    if e1[1] != e2[1]:
        return False
    if len(e1) > 2 and len(e2) > 2:
        if isinstance(e1[2], Timestamp) and isinstance(e2[2], Timestamp):
            if not e1[2].matches(e2[2]):
                return False
        else:
            raise Exception("Edge comparison without solely timestamp data, not implemented yet.")
    return True


def _edge_in_set(iterable, edge):
    for e in iterable:
        if _edges_match(e, edge):
            return True
    return False


def _fill_paths_with_timestamps(graph, paths):
    for p in paths:
        yield [(u, v, graph.get_edge_data(u, v)[TIMESTAMP_PROPERTY_NAME]) for u, v in p]


def _paths_logically_match(path1, path2):
    # Make sure that both paths are either positive or negative.
    if (_count_nots_in_path(path1) % 2) != (_count_nots_in_path(path2) % 2):
        return False

    # # Make sure that timestamps in both paths match.
    # if not _find_path_timestamp(path1).matches(_find_path_timestamp(path2)):
    #     return False

    def _is_skipable_node(node):
        return isinstance(node, AndOperator) or isinstance(node, NotOperator) \
               or isinstance(node, ImplicationOperator)

    cur_node_1 = path1[-1][1]  # start checking from the end
    cur_node_2 = path2[-1][1]
    if cur_node_1 != cur_node_2:
        return False

    index_1 = len(path1) - 1
    index_2 = len(path2) - 1
    while index_1 >= 0:
        cur_node_1 = path1[index_1][0]
        index_1 -= 1
        if _is_skipable_node(cur_node_1):
            continue

        cur_node_2 = path2[index_2][0]
        index_2 -= 1
        while index_2 >= 0 and _is_skipable_node(cur_node_2):
            cur_node_2 = path2[index_2][0]
            index_2 -= 1

        if not ((isinstance(cur_node_1, LogicalOperator) and isinstance(cur_node_2, LogicalOperator)
                 and cur_node_1.logically_matches(cur_node_2)) or cur_node_1 == cur_node_2):
            return False

    return True


def _count_nots_in_path(path):
    all_nots = [e[0] for e in path if isinstance(e[0], NotOperator)]
    if isinstance(path[-1][1], NotOperator):
        all_nots.append(path[-1][1])
    return len(all_nots)


def _remove_common_starting_subpath_from_paths(paths):
    """Removes common starting subpath from given sequence of paths.

    Subpath removal is performed in-place into the edge-sequence of each path.

    :param paths: An iterable containing paths represented as lists of edges.
    """
    if len(paths) > 1:
        for edge in paths[0][::-1]:
            for path in paths:
                if not _edges_match(path[-1], edge):
                    break
            else:
                for path in paths:
                    path.remove(edge)
                    continue
            break
