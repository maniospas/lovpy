if 'DEFAULT_THEOREM_SELECTOR' not in globals():
    DEFAULT_THEOREM_SELECTOR = None  # Current default next theorem selector.


class NextTheoremSelector:
    def select_next(self, graph, theorem_applications, goal, previous_applications):
        raise NotImplementedError("Subclass and implement.")


class SimpleNextTheoremSelector(NextTheoremSelector):
    """Theorem selector that applies the first available theorem."""
    def select_next(self, graph, theorem_applications, goal, previous_applications):
        used_base_theorems = {t.implication_graph for t in previous_applications}
        unused_base_applications = [t for t in theorem_applications
                                    if t.implication_graph not in used_base_theorems]
        if unused_base_applications:
            return unused_base_applications[0]
        else:
            return None


class BetterNextTheoremSelector(NextTheoremSelector):
    """Theorem selector that applies theorems chronologically.

    Next theorem is selected to be the one whose assumption depends on the older
    information in graph.

    Also, the same theorem is never applied twice in a row.
    """

    def select_next(self, graph, theorem_applications, goal, previous_applications):
        # Don't use the last applied theorem.
        used_theorems = \
            [previous_applications[-1].implication_graph] if previous_applications else []
        unused_applications = [t for t in theorem_applications
                               if t.implication_graph not in used_theorems]

        if unused_applications:
            unused_applications.sort(key=lambda app: max(app.matching_paths_timestamps))
            return unused_applications[0]
        else:
            return None


def set_default_theorem_selector(theorem_selector):
    global DEFAULT_THEOREM_SELECTOR

    if not isinstance(theorem_selector, NextTheoremSelector):
        raise TypeError("Only subclasses of NextTheoremSelector can be used.")

    DEFAULT_THEOREM_SELECTOR = theorem_selector


def get_default_theorem_selector():
    return DEFAULT_THEOREM_SELECTOR


if not DEFAULT_THEOREM_SELECTOR:
    DEFAULT_THEOREM_SELECTOR = BetterNextTheoremSelector()
