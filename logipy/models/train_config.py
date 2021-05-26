class TrainConfiguration:
    def __init__(self, dataset_size, max_depth, epochs, batch_size, test_size,
                 export_samples, export_properties, system_evaluation_after_train,
                 random_expansion_probability, negative_samples_percentage,
                 selection_models_dir, termination_models_dir):
        self.dataset_size = dataset_size
        self.max_depth = max_depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.export_samples = export_samples
        self.export_properties = export_properties
        self.system_evaluation_after_train = system_evaluation_after_train
        self.random_expansion_probability = random_expansion_probability
        self.negative_samples_percentage = negative_samples_percentage
        self.selection_models_dir = selection_models_dir
        self.termination_models_dir = termination_models_dir

    def print(self):
        print(f"\tDATASET_SIZE = {self.dataset_size}")
        print(f"\tMAX_DEPTH = {self.max_depth}")
        print(f"\tEPOCHS = {self.epochs}")
        print(f"\tBATCH_SIZE = {self.batch_size}")
        print(f"\tTEST_SIZE = {self.test_size:.2f}")
        print(f"\tEXPORT_SAMPLES = {self.export_samples}")
        print(f"\tEXPORT_PROPERTIES = {self.export_properties}")
        print(f"\tSYSTEM_EVALUATION_AFTER_TRAIN = {self.system_evaluation_after_train}")
        print(f"\tRANDOM_EXPANSION_PROBABILITY = {self.random_expansion_probability:.2f}")
        print(f"\tNEGATIVE_SAMPLES_PERCENTAGE = {self.negative_samples_percentage:.2f}")
        print(f"\tSELECTION_MODELS_DIR = {self.selection_models_dir}")
        print(f"\tTERMINATION_MODELS_DIR = {self.termination_models_dir}")
