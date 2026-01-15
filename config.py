# Configuration settings

# Evaluation Settings
SAMPLE_SIZE = 200  # Number of samples to evaluate per dataset (None for all)
SEED = 42

# Enabled Models for Evaluation
ENABLED_MODELS = [
    "elice-helpy-pro",
    "elice-helpy-edu",
    "gpt-oss-20b",
    "gpt-5.2",
    "gemini-3-pro",
    "gemini-3-flash"
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
