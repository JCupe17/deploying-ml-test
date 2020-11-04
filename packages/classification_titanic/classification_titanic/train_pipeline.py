import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classification_titanic import pipeline
from classification_titanic.processing.data_management import load_dataset, save_pipeline
from classification_titanic.config import config


def run_training() -> None:
    """Train the model."""

    # Read Training Data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # Divide Train and Test sets (only if we read the original data set
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.2, random_state=0
    )

    pipeline.titanic_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=pipeline.titanic_pipe)


if __name__=="__main__":
    run_training()
