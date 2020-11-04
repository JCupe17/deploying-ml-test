import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'survived'

FEATURES  = []

def save_pipeline() -> None:
    """Persist the pipeline."""
    pass


def run_training() -> None:
    """Train the model."""
    print("Training...")


if __name__=="__main__":
    run_training()
