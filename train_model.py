from logipy.models.theorem_proving_model import train_theorem_proving_model
# from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.models.graph_neural_model import train_gnn_theorem_proving_model
from logipy.models.io import save_gnn_model
from logipy.importer.gherkin_importer import import_gherkin_path
from logipy.logic.properties import get_global_properties


if __name__ == "__main__":
    import_gherkin_path()
    properties = get_global_properties()
    # train_theorem_proving_model(properties)
    model, encoder = train_gnn_theorem_proving_model(properties)
    save_gnn_model(model, encoder)

