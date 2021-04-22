DEFAULT_THEOREM_SELECTOR = None  # Current default next theorem selector.


class NextTheoremSelector:
    def select_next(self, graph, theorem_applications, goal, previous_applications):
        raise NotImplementedError("Subclass and implement.")


class SimpleNextTheoremSelector(NextTheoremSelector):
    def select_next(self, graph, theorem_applications, goal, previous_applications):
        used_base_theorems = {t.implication_graph for t in previous_applications}
        unused_base_applications = [t for t in theorem_applications
                                    if t.implication_graph not in used_base_theorems]
        if unused_base_applications:
            return unused_base_applications[0]
        else:
            return None


if not DEFAULT_THEOREM_SELECTOR:
    DEFAULT_THEOREM_SELECTOR = SimpleNextTheoremSelector()
