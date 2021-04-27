from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.importer.gherkin_importer import import_gherkin_path
from logipy.logic.properties import get_global_properties


if __name__ == "__main__":
    import_gherkin_path()
    train_theorem_proving_model(get_global_properties())
