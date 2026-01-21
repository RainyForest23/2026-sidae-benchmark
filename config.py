# Configuration settings

# Evaluation Settings
SAMPLE_SIZE = 200  # Number of samples to evaluate per dataset (None for all)
SEED = 42

# Enabled Models for Evaluation
ENABLED_MODELS = [
    "helpy-pro",        # Helpy Pro Dragon (api.helpy.ai)
    "gpt-oss-20b",      # via mlapi.run
    "gpt-5.2",          # via mlapi.run
    # "helpy-edu",      # Disabled due to 504 Gateway Timeout
]

# Enabled Benchmarks
ENABLED_BENCHMARKS = [
    "kobest",
    "kmmlu",
    "haerae",
    "logickor"
]

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
