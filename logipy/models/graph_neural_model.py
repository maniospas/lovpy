import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import AUC

from logipy.graphs.timed_property_graph import TimedPropertyGraph, TIMESTAMP_PROPERTY_NAME
from .dataset_generator import DatasetGenerator
from .callbacks import ModelEvaluationOnTheoremProvingCallback
from .io import export_generated_samples, export_theorems_and_properties


DATASET_SIZE = 1000
MAX_DEPTH = 12
EPOCHS = 100
BATCH_SIZE = 1


class NextTheoremSamplesGenerator(Sequence):
    """Wrapper sequence for creating samples out of current, goal, next sequences."""

    def __init__(self, current_generator, goal_generator, next_generator,
                 target_data=None, active_indexes=None, batch_size=1):
        # If no active_indexes sequence is applied, then use all data of given generators.
        if not active_indexes:
            active_indexes = list(range(len(current_generator.graphs)))

        self.targets = target_data[active_indexes] if target_data is not None else None

        self.current_data_generator = current_generator.flow(
            active_indexes, targets=self.targets, batch_size=batch_size)
        self.goal_data_generator = goal_generator.flow(active_indexes, batch_size=batch_size)
        self.next_data_generator = next_generator.flow(active_indexes, batch_size=batch_size)

    def __len__(self):
        return self.current_data_generator.__len__()

    def __getitem__(self, item):
        x1 = self.current_data_generator[item]
        x2 = self.goal_data_generator[item]
        x3 = self.next_data_generator[item]
        return [x1[0], x2[0], x3[0]], x1[1]


def train_gnn_theorem_proving_model(properties):
    """Trains an end-to-end GNN-based model for next theorem selection."""
    # Create an one-hot encoder for node labels.
    nodes_encoder = OneHotEncoder(handle_unknown='ignore')
    nodes_labels = list(get_nodes_labels(properties))
    nodes_encoder.fit(np.array(nodes_labels).reshape((-1, 1)))

    print("-" * 80)
    print("Training a DGCNN model.")
    print("-" * 80)
    print(f"\tGenerating {DATASET_SIZE} samples...")

    # Create data generators.
    generator = DatasetGenerator(properties, MAX_DEPTH, DATASET_SIZE,
                                 random_expansion_probability=0.)

    print(f"\tExporting theorems and properties...")
    export_theorems_and_properties(generator.theorems, generator.valid_properties_to_prove)

    graph_samples = []
    for i, s in enumerate(generator):
        if (i % 10) == 0 or i == DATASET_SIZE - 1:
            print(f"\t\tGenerated {i}/{DATASET_SIZE}...", end="\r")
        graph_samples.append(s)
    print(f"\tExporting samples...")
    export_generated_samples(graph_samples, min(DATASET_SIZE, 50))
    y = np.array([int(s.is_positive()) for s in graph_samples]).reshape((-1, 1))
    current_generator, goal_generator, next_generator = \
        create_sample_generators(graph_samples, nodes_encoder)

    # Split train and test data.
    test_size = 0.25
    i_train, i_test = train_test_split(list(range(len(graph_samples))), test_size=test_size)
    train_generator = NextTheoremSamplesGenerator(current_generator, goal_generator, next_generator,
                                                  y, i_train, batch_size=BATCH_SIZE)
    test_generator = NextTheoremSamplesGenerator(current_generator, goal_generator, next_generator,
                                                 y, i_test, batch_size=1)

    print("-" * 80)
    print(f"Training model...")
    print("-" * 80)

    # Train model.
    model = create_gnn_model(current_generator, goal_generator, next_generator)
    print(model.summary())

    actual_evaluation_cb = ModelEvaluationOnTheoremProvingCallback(
        [graph_samples[i] for i in i_train],
        [graph_samples[i] for i in i_test],
        nodes_encoder
    )

    model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1,
        validation_data=test_generator,
        callbacks=[actual_evaluation_cb]
    )

    return model, nodes_encoder


def create_gnn_model(current_generator: PaddedGraphGenerator, goal_generator: PaddedGraphGenerator,
                     next_generator: PaddedGraphGenerator):
    """Creates an end-to-end model for next theorem selection."""
    current_dgcnn_layer_sizes = [32, 32, 32, 1]
    goal_dgcnn_layer_sizes = [32, 32, 32, 1]
    next_dgcnn_layer_sizes = [32, 32, 32, 1]
    k = 20

    # Define the graph embedding branches for the three types of graphs (current, goal, next).
    current_input, current_out = create_graph_embedding_branch(
        current_generator, current_dgcnn_layer_sizes, k
    )
    goal_input, goal_out = create_graph_embedding_branch(
        goal_generator, goal_dgcnn_layer_sizes, k
    )
    next_input, next_out = create_graph_embedding_branch(
        next_generator, next_dgcnn_layer_sizes, k
    )

    # Define the final common branch.
    out = Concatenate()([current_out, goal_out, next_out])
    out = Dense(units=64, activation="relu")(out)
    out = Dense(units=32, activation="relu")(out)
    out = Dense(units=1, activation="sigmoid")(out)

    model = Model(inputs=[current_input, goal_input, next_input], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=["acc", AUC()]
    )
    return model


def create_graph_embedding_branch(generator: PaddedGraphGenerator, dgcnn_layer_sizes: list, k: int):
    dgcnn = DeepGraphCNN(
        layer_sizes=dgcnn_layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        generator=generator,
        k=k,
        bias=False
    )
    x_in, x_out = dgcnn.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(dgcnn_layer_sizes),
                   strides=sum(dgcnn_layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    return x_in, x_out


def create_sample_generators(graph_samples: list, encoder: OneHotEncoder, verbose=False):
    current_graphs = []
    goal_graphs = []
    next_graphs = []
    for s in graph_samples:
        current_graph, norm = convert_timedpropertygraph_to_stellargraph(s.current_graph, encoder)
        goal_graph, _ = convert_timedpropertygraph_to_stellargraph(s.goal, encoder)
        next_graph, _ = convert_timedpropertygraph_to_stellargraph(s.next_theorem, encoder, norm)
        current_graphs.append(current_graph)
        goal_graphs.append(goal_graph)
        next_graphs.append(next_graph)

    # Print statistical info about nodes and edges number in three types of graphs.
    if verbose:
        current_summary = pd.DataFrame(
            [(g.number_of_nodes(), g.number_of_edges) for g in current_graphs],
            columns=["nodes", "edges"]
        )
        print("Summary of current graphs:")
        print(current_summary.describe().round(1))
        goal_summary = pd.DataFrame(
            [(g.number_of_nodes(), g.number_of_edges) for g in goal_graphs],
            columns=["nodes", "edges"]
        )
        print("Summary of goal graphs:")
        print(goal_summary.describe().round(1))
        next_summary = pd.DataFrame(
            [(g.number_of_nodes(), g.number_of_edges) if g else (0, 0) for g in next_graphs],
            columns=["nodes", "edges"]
        )
        print("Summary of next theorem graphs:")
        print(next_summary.describe().round(1))

    return create_three_padded_generators(current_graphs, goal_graphs, next_graphs)


def create_three_padded_generators(current_graphs, goal_graphs, next_graphs):
    current_generator = PaddedGraphGenerator(current_graphs)
    goal_generator = PaddedGraphGenerator(goal_graphs)
    next_generator = PaddedGraphGenerator(next_graphs)
    return current_generator, goal_generator, next_generator


def convert_timedpropertygraph_to_stellargraph(graph: TimedPropertyGraph, encoder: OneHotEncoder,
                                               normalization_value=None):
    nx_graph = graph.graph.copy()

    # Use 1-hot encoded node labels as features of the nodes.
    nodes = list(nx_graph.nodes)
    node_features = [
        encoder.transform(np.array([graph.get_node_label(n)]).reshape(-1, 1)).toarray().flatten()
        for n in nodes
    ]

    # Use normalized time values as weights of the edges.
    edges = list(nx_graph.edges)
    time_values = []
    for e in edges:
        timestamp = nx_graph[e[0]][e[1]][e[2]][TIMESTAMP_PROPERTY_NAME]
        if timestamp.is_absolute():
            value = timestamp.get_absolute_value()
        else:
            value = timestamp.get_relative_value()
        time_values.append(value)
    time_values = np.array(time_values, dtype="float32")
    if not normalization_value:
        normalization_value = max(time_values)
    if normalization_value > 1.:
        time_values = time_values / normalization_value
    for e, v in zip(edges, time_values):
        nx_graph[e[0]][e[1]][e[2]]["time_value"] = v

    sg_graph = StellarGraph.from_networkx(graph.graph, edge_weight_attr="time_value",
                                          node_features=zip(nodes, node_features))

    return sg_graph, normalization_value


def get_nodes_labels(properties):
    """Returns the node labels contained in given property graphs.

    :param properties: An iterable of TimedPropertyGraph objects.

    :return: A set containing all node labels used in given sequence of property graphs.
    """
    labels = set()
    for p in properties:
        for n in p.graph.nodes:
            labels.add(p.get_node_label(n))
    return labels
