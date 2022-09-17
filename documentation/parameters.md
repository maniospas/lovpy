# Custom run

## Environment setup
- `LOVPY_ENGINE = BASIC | MLP | GNN | HYBRID` : Explicitly enables a specific verification engine.  
- `LOVPY_DISABLE_GPU = 0 | 1` : When set to `1` disables GPU usage by tensorflow.
- `LOVPY_SESSION_NAME = <name>` : Sets a custom name for current session.
- `LOVPY_TEMP_DIR = <dir>` : Directory where lovpy will store all data and reports of a session.
- `LOVPY_MODELS_DIR = <dir>` : Directory which lovpy will use for storing and loading models. 
- `LOVPY_DEV_MODE = 0 | 1` : When set to `1` enables development mode.
