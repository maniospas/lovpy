import networkx


EDGE_COLORIZATION_LABEL = "colorized_edge"
NODE_IN_COLORIZATION_LABEL = "node_in_colorized"
NODE_OUT_COLORIZATION_LABEL = "node_out_colorized"


class ColorizableMultiDiGraph(networkx.MultiDiGraph):

    def build_colorization_scheme(self):
        """Updates the colorization scheme of the graph.

        It should be called after all desired paths have been colorized and before any method
        that depends on colorization scheme is called.
        """
        self.in_colorize_nodes()
        self.out_colorize_nodes()

    def colorize_path(self, path):
        """Colorizes given path.

        :param path: A path represented as a sequence of edges in three-tuple form
                (source, target, key).
        """
        for e in path:
            self.edges[e[0], e[1], e[2]][EDGE_COLORIZATION_LABEL] = True

    def out_colorize_nodes(self):
        node_out_colorized = {}

        for n in reversed(list(networkx.algorithms.dag.topological_sort(self))):
            if self.out_degree(n) == 0:  # leaf nodes are always out-colorized
                node_out_colorized[n] = True
            else:
                is_node_out_colorized = True
                for e in self.edges(n, keys=True):
                    if not self.is_edge_colorized(e[0], e[1], e[2]) or not node_out_colorized[e[1]]:
                        is_node_out_colorized = False
                        break
                node_out_colorized[n] = is_node_out_colorized

        networkx.set_node_attributes(self, node_out_colorized, NODE_OUT_COLORIZATION_LABEL)

    def in_colorize_nodes(self, root=None):
        if not root:
            root = self.get_root_node()
            nodes_in_colorization = {self.get_root_node(): True}  # root node is always in-colorized
            networkx.set_node_attributes(self, nodes_in_colorization, NODE_IN_COLORIZATION_LABEL)

        in_colorized_neighbors = []

        # Using BFS colorize all neighbors whose incoming edges are all colorized.
        node_attrs = networkx.get_node_attributes(self, NODE_IN_COLORIZATION_LABEL)
        for n in self.neighbors(root):
            is_neighbor_in_colorized = True
            for e in self.in_edges(n, keys=True):
                if not self.is_edge_colorized(e[0], e[1], e[2]):
                    is_neighbor_in_colorized = False
                    break
            node_attrs[n] = is_neighbor_in_colorized
            if is_neighbor_in_colorized:
                in_colorized_neighbors.append(n)
        networkx.set_node_attributes(self, node_attrs, NODE_IN_COLORIZATION_LABEL)

        # Keep colorizing the subDAG only when a node is marked as in-colorized.
        for n in in_colorized_neighbors:
            self.in_colorize_nodes(n)

    def is_node_in_colorized(self, node):
        return networkx.get_node_attributes(self, NODE_IN_COLORIZATION_LABEL).get(node, False)

    def is_node_out_colorized(self, node):
        return networkx.get_node_attributes(self, NODE_OUT_COLORIZATION_LABEL).get(node, False)

    def is_edge_colorized(self, u, v, k):
        return self.get_edge_data(u, v, k).get(EDGE_COLORIZATION_LABEL, False)

    def get_root_node(self):
        return [n for n, d in self.in_degree() if d == 0].pop()

    def disconnect_fully_colorized_sub_dag(self, root=None):
        should_clean_nodes = False
        if not root:
            root = self.get_root_node()
            should_clean_nodes = True

        # Cannot remove any edge to successors, if current node has non colorized dependencies.
        if not self.is_node_in_colorized(root):
            return

        # Traverse only colorized edges.
        successors_edges = {e for e in self.edges(root, keys=True)
                            if self.is_edge_colorized(e[0], e[1], e[2])}

        if successors_edges:
            # Remove any colorized edge leading towards a fully colorized subDAG.
            for e in successors_edges:
                if self.is_node_out_colorized(e[1]):
                    self.remove_edge(root, e[1], key=e[2])
                self.disconnect_fully_colorized_sub_dag(root=e[1])

            # Finally, cleanup orphan nodes.
            if should_clean_nodes:
                self._remove_zero_degree_nodes()

    def clear_colorization(self):
        for e in self.edges:
            if self.edges[e[0], e[1], e[2]].get(EDGE_COLORIZATION_LABEL, False):
                self.edges[e[0], e[1], e[2]][EDGE_COLORIZATION_LABEL] = False
        networkx.set_node_attributes(self, {n: False for n in self.nodes},
                                     NODE_IN_COLORIZATION_LABEL)
        networkx.set_node_attributes(self, {n: False for n in self.nodes},
                                     NODE_OUT_COLORIZATION_LABEL)

    def _remove_zero_degree_nodes(self):
        nodes_to_remove = []
        for n, d in self.degree():
            if d == 0:
                nodes_to_remove.append(n)
        self.remove_nodes_from(nodes_to_remove)
