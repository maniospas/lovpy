# Fix for pygraphviz issue not properly releasing resources after calling graphviz binaries.
# Exporting thousands of graphs in a single interpreter session will still crash.
import win32file
win32file._setmaxstdio(8192)

import pickle
import random
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib import image as mpimage
from tensorflow.keras.models import load_model


# Paths about simple NN model.
main_model_path = None
predicates_map_path = None

# Paths about DGCNN model.
graph_selection_model_path = None
graph_termination_model_path = None
graph_encoder_path = None

# Paths about sample visualization.
current_graph_path = None
goal_graph_path = None
next_graph_path = None

# Paths about training samples exporting.
graph_model_train_output_dir_path = None

# Globals about DGCNN theorem selection process exporting.
dgcnn_selection_process_export_path = None
dgcnn_selection_processes_exported = Counter()


def save_gnn_models(selection_model, termination_model, encoder,
                    selection_model_path=None,
                    termination_model_path=None,
                    encoder_path=None):
    """Saves given gnn models along with nodes encoder to disk."""
    if not selection_model_path or not termination_model_path or not encoder_path:
        selection_model_path = graph_selection_model_path
        termination_model_path = graph_termination_model_path
        encoder_path = graph_encoder_path

    selection_model.save(selection_model_path)
    # termination_model.save(termination_model_path)
    with encoder_path.open("wb") as f:
        pickle.dump(encoder, f)


def load_gnn_models():
    """Loads gnn models along with nodes encoder from disk."""
    selection_model = None
    termination_model = None
    encoder = None

    if (graph_selection_model_path.exists()
            # and graph_termination_model_path.exists()
            and graph_encoder_path.exists()):
        selection_model = load_model(graph_selection_model_path)
        # termination_model = load_model(graph_termination_model_path)
        with graph_encoder_path.open("rb") as f:
            encoder = pickle.load(f)

    return selection_model, termination_model, encoder


def export_generated_samples(samples, max_num=None):
    samples_out_dir = graph_model_train_output_dir_path / "samples"

    if samples_out_dir is None:
        raise RuntimeError("Models module was not correctly initialized.")

    if not samples_out_dir.exists():
        samples_out_dir.mkdir(parents=True)

    if not max_num:
        max_num = len(samples)
    samples = random.sample(samples, max_num)
    for i, s in enumerate(samples):
        print(f"\t\tExported {i+1}/{len(samples)}", end="\r")
        is_positive_text = "Positive" if s.is_next_theorem_correct() else "Negative"
        is_terminal_text = "Terminate" if s.should_proving_process_terminate() else "Continue"
        s.visualize(f"Sample #{i+1} ({is_positive_text} | {is_terminal_text})",
                    samples_out_dir/f"sample{i + 1}.png")


def export_theorems_and_properties(theorems, properties):
    theorems_out_path = graph_model_train_output_dir_path / "theorems"
    properties_out_path = graph_model_train_output_dir_path / "properties"

    if not theorems_out_path.exists():
        theorems_out_path.mkdir(parents=True)
    if not properties_out_path.exists():
        properties_out_path.mkdir(parents=True)

    for i, t in enumerate(theorems):
        t.visualize(f"Theorem #{i+1}", export_path=theorems_out_path/f"theorem_{i+1}.png")
    for i, t in enumerate(properties):
        t.visualize(f"Property #{i + 1}", export_path=properties_out_path/f"property_{i + 1}.png")


def export_grouped_instance(current_graph, goal_graph, next_graph, title,
                            property_group, proving_process_group, group_label):
    if not dgcnn_selection_process_export_path:
        raise RuntimeError("Module was not initialized properly.")

    export_dir = dgcnn_selection_process_export_path
    export_dir /= str(property_group)
    export_dir /= f"proving_process_{proving_process_group}"
    export_dir /= f"selection_process_{group_label}"
    if not export_dir.exists():
        export_dir.mkdir(parents=True)

    # Name exports under the same group label with an incrementing numerical value.
    export_file_path = export_dir / (
            str(dgcnn_selection_processes_exported[group_label]+1) + ".png")
    visualize_three_graphs(current_graph, goal_graph, next_graph, title=title,
                           export_path=export_file_path)
    dgcnn_selection_processes_exported[group_label] += 1


def model_file_exists():
    return main_model_path.exists()


def visualize_three_graphs(current_graph, goal_graph, next_graph,
                           title="",
                           export_path=None,
                           current_title="Current Graph",
                           goal_title="Goal Property",
                           next_title="Next Theorem",
                           visualize_next_assumption_in_current=True):
    """Visualizes current, goal, next graphs in a single figure, side by side.

    Figure is consisted of three subplots:
     -The leftmost subplot is the instance of execution graph.
     -The central subplot is the goal property that should be proved.
     -The rightmost subplot is the next theorem to be applied.

    :param current_graph: Current graph.
    :param goal_graph: Goal graph.
    :param next_graph: Next theorem graph.
    :param str title: A supertitle for the whole figure.
    :param Path export_path: If this argument is given, then instead of displaying the
            sample figure on screen, it is exported to pointed location.
    :param current_title: Subtitle of current graph.
    :param goal_title: Subtitle of goal graph.
    :param next_title: Subtitle of next theorem graph.
    :param visualize_next_assumption_in_current: If set to True, assumption part of next
            theorem is marked with a different color in current graph. If assumption is not
            contained in current graph, then nothing happens.
    """
    if visualize_next_assumption_in_current:
        assumption, _ = next_graph.get_top_level_implication_subgraphs()
        current_graph = current_graph.get_copy()
        matching_cases, _, _, _ = current_graph.find_equivalent_subgraphs(assumption)
        if matching_cases:
            for path in matching_cases[0]:
                current_graph.graph.colorize_path(path)

    a_graph1 = current_graph.to_agraph(current_title)
    a_graph2 = goal_graph.to_agraph(goal_title)
    if next_graph:
        a_graph3 = next_graph.to_agraph(next_title)

    # Export to disk temp jpg images of the three graphs.
    a_graph1.layout("dot")
    a_graph1.graph_attr.update(dpi=300.0)
    a_graph1.graph_attr.update(size=4)
    a_graph1.graph_attr.update(bgcolor="transparent")
    a_graph1.draw(current_graph_path)
    a_graph2.layout("dot")
    a_graph2.graph_attr.update(dpi=300.0)
    a_graph2.graph_attr.update(size=4)
    a_graph2.graph_attr.update(bgcolor="transparent")
    a_graph2.draw(goal_graph_path)
    if next_graph:
        a_graph3.layout("dot")
        a_graph3.graph_attr.update(dpi=300.0)
        a_graph3.graph_attr.update(size=4)
        a_graph3.graph_attr.update(bgcolor="transparent")
        a_graph3.draw(next_graph_path)

    # Plot the three graph images side by side.
    f, axarr = plt.subplots(1, 3, num=None, figsize=(12, 4), dpi=300,
                            facecolor='w', edgecolor='w')
    f.tight_layout()
    f.suptitle(title, fontsize=15, fontweight='bold')
    axarr[0].imshow(mpimage.imread(current_graph_path))
    axarr[1].imshow(mpimage.imread(goal_graph_path))
    if next_graph:
        axarr[2].imshow(mpimage.imread(next_graph_path))
    for axes in axarr:
        axes.axis('off')

    if export_path:
        f.savefig(export_path)
        plt.close(f)
    else:
        plt.show()
