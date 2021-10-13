# from tensorflow.keras.callbacks import Callback
#
# from .evaluation import evaluate_theorem_selector
# from lovpy.logic.prover import prove_property
#
#
# class ModelEvaluationOnTheoremProvingCallback(Callback):
#     """Callback to evaluate a neural theorem proving model on its ability to prove theorems.
#
#     The ability to prove theorems is evaluated in the sense that a model should be able
#     to produce a sequence of theorem applications in order to finally prove or consider as
#     non-provable the target theorem. This binary output of the prover (proved | not proved)
#     is compared against the ground truth value of each sample (provable | not provable).
#
#     It outputs the following metrics:
#         -Accuracy
#         -Fallout
#         -AUC (not yet implemented)
#     """
#
#     def __init__(self, train_samples, validation_samples, nodes_encoder):
#         super().__init__()
#         self.train_samples = train_samples
#         self.validation_samples = validation_samples
#         self.nodes_encoder = nodes_encoder
#
#     def on_train_end(self, logs=None):
#         acc, fallout = compute_accuracy_fallout_on_samples_proving(
#             self.train_samples, self.model, self.nodes_encoder, verbose=True)
#         val_acc, val_fallout = compute_accuracy_fallout_on_samples_proving(
#             self.validation_samples, self.model, self.nodes_encoder, verbose=True)
#
#         if logs:
#             logs["proving_acc"] = acc
#             logs["proving_fallout"] = fallout
#             logs["val_proving_acc"] = acc
#             logs["val_proving_fallout"] = fallout
#
#         print("\tTesting dataset:  proving_acc: {} - proving_fallout: {}".format(
#                 round(acc, 4), round(fallout, 4)))
#         print("\tValidation dataset: val_proving_acc: {} - val_proving_fallout: {}".format(
#                 round(val_acc, 4), round(val_fallout, 4)))
