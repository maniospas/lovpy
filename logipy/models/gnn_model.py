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
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from logipy.graphs.timed_property_graph import TimedPropertyGraph, TIMESTAMP_PROPERTY_NAME
# from .callbacks import ModelEvaluationOnTheoremProvingCallback
from .train_config import TrainConfiguration
from logipy.evaluation.evaluation import evaluate_theorem_selector_on_samples
from .theorem_proving_model import TheoremProvingModel
from .io import save_gnn_models, load_gnn_models


class GNNModel(TheoremProvingModel):
    """An end-to-end GNN-based model for theorem proving process."""

    def __init__(self, name, path,
                 next_theorem_model=None,
                 proving_termination_model=None,
                 nodes_encoder=None):
        super().__init__(name, path)
        self.next_theorem_model = next_theorem_model
        self.proving_termination_model = proving_termination_model
        self.nodes_encoder = nodes_encoder

    def train_core(self, dataset, properties, config: TrainConfiguration):
        self.nodes_encoder = create_nodes_encoder(properties)

        # Split train and test data.
        i_train, i_test = train_test_split(list(range(len(dataset))),
                                           test_size=config.test_size)

        # print(f"Training next theorem selection model...")
        # print("-" * 80)
        self.next_theorem_model = train_next_theorem_selection_model(dataset, self.nodes_encoder,
                                                                     i_train, i_test, config)

        # print("-" * 80)
        # print(f"Training proving process termination model...")
        # print("-" * 80)
        # self.proving_termination_model = train_proving_termination_model(
        #     graph_samples, nodes_encoder, i_train, i_test, config)
        # proving_termination_model = None

        if config.system_evaluation_after_train:
            _evaluate_model(
                self,
                [dataset[i] for i in i_train],
                [dataset[i] for i in i_test],
                config
            )

    def predict(self,
                current: TimedPropertyGraph,
                theorem_applications: list,
                goal: TimedPropertyGraph):
        current_graph, norm = convert_timedpropertygraph_to_stellargraph(current,
                                                                         self.nodes_encoder)
        goal_graph, _ = convert_timedpropertygraph_to_stellargraph(goal, self.nodes_encoder)

        current_generator, goal_generator, next_generator = create_padded_generators(
            [current_graph] * len(theorem_applications),
            [goal_graph] * len(theorem_applications),
            [convert_timedpropertygraph_to_stellargraph(t_app.actual_implication,
                                                        self.nodes_encoder, norm)[0]
             for t_app in theorem_applications]
        )

        inference_generator = ProvingModelSamplesGenerator(
            current_generator, goal_generator, next_generator)

        return self.next_theorem_model.predict(inference_generator)

    def save(self):
        save_gnn_models(self.next_theorem_model, self.proving_termination_model, self.nodes_encoder)

        # Save GNN model also in scratchdir, for easy retrieval.
        # save_gnn_models(
        #     next_theorem_model,
        #     proving_termination_model,
        #     encoder,
        #     scratch_model_out_base / GRAPH_SELECTION_MODEL_NAME,
        #     scratch_model_out_base / GRAPH_TERMINATION_MODEL_NAME,
        #     scratch_model_out_base / GRAPH_ENCODER_NAME
        # )

    def plot(self, folder):
        plot_model(
            self.next_theorem_model,
            to_file=folder / f"{self.name}.png",
            show_shapes=True,
            show_layer_names=True
        )

        # plot_model(
        #     proving_termination_model,
        #     to_file=scratch_model_out_base / "dgcnn_termination_model.png",
        #     show_shapes=True,
        #     show_layer_names=True
        # )

    @staticmethod
    def load(path=None):
        next_model, term_model, encoder = load_gnn_models()
        return GNNModel("GNN Model", path,
                        next_theorem_model=next_model,
                        proving_termination_model=term_model,
                        nodes_encoder=encoder)


class ProvingModelSamplesGenerator(Sequence):
    """Wrapper sequence for creating samples out of current, goal, next sequences."""

    def __init__(self, current_generator, goal_generator, next_generator=None,
                 target_data=None, active_indexes=None, batch_size=1):
        # If no active_indexes sequence is applied, then use all data of given generators.
        if not active_indexes:
            active_indexes = list(range(len(current_generator.graphs)))

        self.targets = target_data[active_indexes] if target_data is not None else None

        self.current_data_generator = current_generator.flow(
            active_indexes,
            targets=self.targets,
            symmetric_normalization=True,
            weighted=True,
            shuffle=False,
            batch_size=batch_size
        )
        self.goal_data_generator = goal_generator.flow(
            active_indexes,
            symmetric_normalization=True,
            weighted=True,
            shuffle=False,
            batch_size=batch_size
        )
        if next_generator:
            self.next_data_generator = next_generator.flow(
                active_indexes,
                symmetric_normalization=True,
                weighted=True,
                shuffle=False,
                batch_size=batch_size
            )
        else:
            self.next_data_generator = None

    def __len__(self):
        return self.current_data_generator.__len__()

    def __getitem__(self, item):
        x1 = self.current_data_generator[item]
        x2 = self.goal_data_generator[item]
        if self.next_data_generator:
            x3 = self.next_data_generator[item]
            return [x1[0], x2[0], x3[0]], x1[1]
        else:
            return [x1[0], x2[0]], x1[1]


def train_next_theorem_selection_model(graph_samples, nodes_encoder, i_train, i_test,
                                       config: TrainConfiguration):
    # Create input generators to feed model and output data.
    current_generator, goal_generator, next_generator = \
        create_sample_generators(graph_samples, nodes_encoder, verbose=True)
    next_theorem_labels = np.array(
        [int(s.is_next_theorem_correct()) for s in graph_samples]).reshape((-1, 1))

    train_generator = ProvingModelSamplesGenerator(current_generator,
                                                   goal_generator,
                                                   next_generator,
                                                   target_data=next_theorem_labels,
                                                   active_indexes=i_train,
                                                   batch_size=config.batch_size)
    test_generator = ProvingModelSamplesGenerator(current_generator,
                                                  goal_generator,
                                                  next_generator,
                                                  target_data=next_theorem_labels,
                                                  active_indexes=i_test,
                                                  batch_size=1)

    # Train model.
    model = create_gnn_model(current_generator, goal_generator, next_generator)
    print(model.summary())

    model_filename = ("selection_model"
                      + "-epoch_{epoch:02d}"
                      + "-val_acc_{val_acc:.2f}"
                      + "-val_auc_{val_auc:.2f}")
    model_checkpoint_cb = ModelCheckpoint(
        filepath=config.selection_models_dir/model_filename,
    )
    best_model_path = config.selection_models_dir / "_best_selection_model"
    best_model_cb = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True
    )

    model.fit(
        train_generator,
        epochs=config.epochs,
        verbose=1,
        validation_data=test_generator,
        callbacks=[model_checkpoint_cb, best_model_cb]
    )

    best_model = load_model(best_model_path)

    return best_model


# def train_proving_termination_model(graph_samples, nodes_encoder, i_train, i_test,
#                                     config: TrainConfiguration):
#     # Create input generators to feed model and output data.
#     current_generator, goal_generator, _ = \
#         create_sample_generators(graph_samples, nodes_encoder, verbose=True)
#     termination_labels = []
#     for s in graph_samples:
#         should_terminate = s.should_proving_process_terminate()
#         if should_terminate:
#             termination_labels.append([0., 1.])
#         else:
#             termination_labels.append([1., 0])
#     termination_labels = np.array(termination_labels).reshape((-1, 2))
#
#     train_generator = ProvingModelSamplesGenerator(current_generator,
#                                                    goal_generator,
#                                                    target_data=termination_labels,
#                                                    active_indexes=i_train,
#                                                    batch_size=config.batch_size)
#     test_generator = ProvingModelSamplesGenerator(current_generator,
#                                                   goal_generator,
#                                                   target_data=termination_labels,
#                                                   active_indexes=i_test,
#                                                   batch_size=1)
#
#     # Train model.
#     model = create_proving_termination_model(current_generator, goal_generator)
#     print(model.summary())
#
#     model_filename = ("termination_model"
#                       + "-epoch_{epoch:02d}"
#                       + "-val_acc_{val_acc:.2f}"
#                       + "-val_auc_{val_auc_1:.2f}")
#     model_checkpoint_cb = ModelCheckpoint(
#         filepath=config.termination_models_dir/model_filename,
#         monitor="val_auc_1"
#     )
#
#     model.fit(
#         train_generator,
#         epochs=config.epochs,
#         verbose=1,
#         validation_data=test_generator,
#         callbacks=[model_checkpoint_cb]
#     )
#
#     return model


def create_nodes_encoder(properties):
    """Create an one-hot encoder for node labels."""
    nodes_encoder = OneHotEncoder(handle_unknown='ignore')
    nodes_labels = list(get_nodes_labels(properties))
    nodes_encoder.fit(np.array(nodes_labels).reshape((-1, 1)))
    return nodes_encoder


def create_gnn_model(current_generator: PaddedGraphGenerator, goal_generator: PaddedGraphGenerator,
                     next_generator: PaddedGraphGenerator):
    """Creates an end-to-end model for next theorem selection."""
    current_dgcnn_layer_sizes = [32] * 3 + [1]
    current_dgcnn_layer_activations = ["relu"] * 4
    goal_dgcnn_layer_sizes = [32] * 3 + [1]
    goal_dgcnn_layer_activations = ["relu"] * 4
    next_dgcnn_layer_sizes = [32] * 3 + [1]
    next_dgcnn_layer_activations = ["relu"] * 4
    sortpooling_out_nodes = 32

    # Define the graph embedding branches for the three types of graphs (current, goal, next).
    current_input, current_out = create_graph_embedding_branch(
        current_generator, current_dgcnn_layer_sizes,
        current_dgcnn_layer_activations, sortpooling_out_nodes
    )
    goal_input, goal_out = create_graph_embedding_branch(
        goal_generator, goal_dgcnn_layer_sizes,
        goal_dgcnn_layer_activations, sortpooling_out_nodes
    )
    next_input, next_out = create_graph_embedding_branch(
        next_generator, next_dgcnn_layer_sizes,
        next_dgcnn_layer_activations, sortpooling_out_nodes
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
        metrics=["acc", AUC(name="auc")]
    )
    return model


# def create_proving_termination_model(current_generator: PaddedGraphGenerator,
#                                      goal_generator: PaddedGraphGenerator):
#     """Creates an end-to-end model for next theorem selection."""
#     current_dgcnn_layer_sizes = [64] * 6
#     current_dgcnn_layer_activations = ["relu"] * 6
#     goal_dgcnn_layer_sizes = [64] * 6
#     goal_dgcnn_layer_activations = ["relu"] * 6
#     sortpooling_out_nodes = 32
#
#     # Define the graph embedding branches for the three types of graphs (current, goal, next).
#     current_input, current_out = create_graph_embedding_branch(
#         current_generator, current_dgcnn_layer_sizes,
#         current_dgcnn_layer_activations, sortpooling_out_nodes
#     )
#     goal_input, goal_out = create_graph_embedding_branch(
#         goal_generator, goal_dgcnn_layer_sizes,
#         goal_dgcnn_layer_activations, sortpooling_out_nodes
#     )
#
#     # Define the final common branch.
#     out = Concatenate()([current_out, goal_out])
#     out = Dense(units=64, activation="relu")(out)
#     out = Dense(units=32, activation="relu")(out)
#     out = Dense(units=2, activation="softmax")(out)
#
#     model = Model(inputs=[current_input, goal_input], outputs=out)
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss="binary_crossentropy",
#         metrics=["acc", AUC()]


def create_graph_embedding_branch(generator: PaddedGraphGenerator, dgcnn_layer_sizes: list,
                                  dgcnn_layer_activations: list, k: int):
    dgcnn = DeepGraphCNN(
        layer_sizes=dgcnn_layer_sizes,
        activations=dgcnn_layer_activations,
        generator=generator,
        k=k,
        bias=True,
        dropout=0.5,
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

    return create_padded_generators(current_graphs, goal_graphs, next_graphs)


def create_padded_generators(*args):
    generators = [PaddedGraphGenerator(arg) for arg in args]
    return tuple(generators)


def convert_timedpropertygraph_to_stellargraph(graph: TimedPropertyGraph, encoder: OneHotEncoder,
                                               normalization_value=None, shift_to_positive=True):
    """Converts a TimedPropertyGraph to a StellarGraph object.

    :param TimedPropertyGraph graph: Graph to be converted to StellarGraph object.
    :param OneHotEncoder encoder: One-hot encoder for node labels contained in graph.
    :param normalization_value: Value to be used for edge timestamps normalization. If not
            provided, normalization is performed using the maximum timestamp value out of
            all edges in the graph.
    :param shift_to_positive: Currently not used.

    :return: A StellarGraph representation of input graph, suitable for input to a
            PaddedGraphGenerator.
    """
    nx_graph = graph.graph

    if not normalization_value:
        # Find maximum timestamp value to be used for normalization.
        time_values = []
        for e in nx_graph.edges:
            timestamp = nx_graph[e[0]][e[1]][e[2]][TIMESTAMP_PROPERTY_NAME]
            time_values.append(_get_timestamp_numerical_value(timestamp))
        time_values = np.array(time_values, dtype="float32")
        normalization_value = max(time_values)

    # Use 1-hot encoded node labels, along with normalized min and max of outgoing edges as
    # features of the nodes.
    nodes = list(nx_graph.nodes)
    node_features = []
    for n in nodes:
        feature = encoder.transform(
            np.array([graph.get_node_label(n)]).reshape(-1, 1)).toarray().flatten()

        out_timestamps = [_get_timestamp_numerical_value(e[3])
                          for e in nx_graph.edges(n, data=TIMESTAMP_PROPERTY_NAME, keys=True)]
        if out_timestamps:
            out_timestamps = np.array(out_timestamps, dtype="float32")
            if normalization_value > 1.:
                out_timestamps = out_timestamps / normalization_value
            feature = np.append(feature, [min(out_timestamps), max(out_timestamps)])
        else:
            feature = np.append(feature, [0., 0.])

        node_features.append(feature)

    sg_graph = StellarGraph.from_networkx(nx_graph, node_features=zip(nodes, node_features))
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


def _get_timestamp_numerical_value(timestamp):
    if timestamp.is_absolute():
        return timestamp.get_absolute_value()
    else:
        return timestamp.get_relative_value()


def _evaluate_model(model: GNNModel, train_samples, validation_samples, config: TrainConfiguration):
    print("-" * 80)
    print("Evaluating DGCNN proving system on synthetic theorems of samples...")
    print("-" * 80)
    from .graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
    theorem_selector = GraphNeuralNextTheoremSelector(model)
    _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples)

    if config.system_comparison_to_deterministic_after_train:
        print("-" * 80)
        print("Evaluating deterministic proving system on synthetic theorems of samples...")
        print("-" * 80)
        from logipy.logic.next_theorem_selectors import BetterNextTheoremSelector
        theorem_selector = BetterNextTheoremSelector()
        _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples)


def _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples):
    acc, fallout = evaluate_theorem_selector_on_samples(
        theorem_selector, train_samples, verbose=True)
    print("\tTesting dataset:  proving_acc: {} - proving_fallout: {}".format(
        round(acc, 4), round(fallout, 4)))

    val_acc, val_fallout = evaluate_theorem_selector_on_samples(
        theorem_selector, validation_samples, verbose=True)
    print("\tValidation dataset: val_proving_acc: {} - val_proving_fallout: {}".format(
        round(val_acc, 4), round(val_fallout, 4)))
