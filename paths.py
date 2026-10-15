import os

HERE = os.path.dirname(os.path.abspath(__file__))

THESIS_DATA = os.environ.get(
    "THESIS_DATA",
    r"C:\Users\hp\Downloads\typybenchdata\typybenchdata",
)

CLEAN_REPOS = os.path.join(THESIS_DATA, "cleaned_repos")
REPORTS_DIR = os.path.join(THESIS_DATA, "ground_truth_validation_mypy_only")
DATASETS_DIR = CLEAN_REPOS

FILTERED_DIR = os.path.join(HERE, "datasets", "filtered")
UNFILTERED_CLEAN_DIR = os.path.join(HERE, "datasets", "unfiltered_from_clean")
UNFILTERED_RAW_DIR = os.path.join(HERE, "datasets", "unfiltered_from_raw")
BRAINS_DIR = os.path.join(HERE, "Brains")
DATASET_CREATION_DIR = os.path.join(HERE, "Dataset creation")
INJECTION_DIR = os.path.join(HERE, "injection")
FILTERING_DIR = os.path.join(HERE, "filtering")
EVALUATION_DIR = os.path.join(HERE, "Evaluation")
TRAINING_DIR = os.path.join(HERE, "training scripts")
TESTING_DIR = os.path.join(HERE, "testing scripts")