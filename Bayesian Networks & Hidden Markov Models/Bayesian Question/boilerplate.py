#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model  # Ensure this module is correctly implemented
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

########################
## Utility Functions ##
########################

def load_data(train_path="train_data.csv", val_path="validation_data.csv"):
    """
    Load training and validation datasets from CSV files.

    Args:
        train_path (str): Path to the training dataset.
        val_path (str): Path to the validation dataset.

    Returns:
        tuple: Cleaned train and validation datasets as Pandas DataFrames.
    """
    logging.info("[+] Loading datasets...")
    try:
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        logging.info(f"[+] Train Data Shape: {train_data.shape}")
        logging.info(f"[+] Validation Data Shape: {val_data.shape}")
    except FileNotFoundError as fnfe:
        logging.critical(f"[-] Dataset file not found: {fnfe}")
        raise fnfe
    except Exception as e:
        logging.error(f"[-] Failed to load data: {e}")
        raise e

    # Validate and clean data
    train_data = validate_and_clean_data(train_data, "Train Data")
    val_data = validate_and_clean_data(val_data, "Validation Data")
    return train_data, val_data


def validate_and_clean_data(df, name):
    """
    Validate and clean the dataset.

    Args:
        df (DataFrame): Input DataFrame to validate and clean.
        name (str): Name of the dataset for logging purposes.

    Returns:
        DataFrame: Cleaned DataFrame.
    """
    logging.info(f"[+] Validating {name}...")
    if df.isnull().sum().sum() > 0:
        logging.warning(f"[-] Missing values detected in {name}. Dropping rows with NaN values.")
        df = df.dropna()

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logging.warning(f"[-] Non-numeric columns detected in {name}: {list(non_numeric_cols)}. Encoding them.")
        df = pd.get_dummies(df, drop_first=True)

    if df.empty:
        logging.critical(f"[-] {name} is empty after cleaning. Check your data source.")
        raise ValueError(f"{name} is empty after cleaning.")

    logging.info(f"[+] {name} cleaned. New Shape: {df.shape}")
    return df


######################
## Bayesian Networks ##
######################

def create_network(df, methodtype, task_name):
    """
    Define and fit a Bayesian Network.

    Args:
        df (DataFrame): Dataset for creating the network.
        methodtype (str): Structure learning method type.
        task_name (str): Descriptive name for the task.

    Returns:
        dict: Bayesian network model.
    """
    logging.info(f"[+] Creating {task_name} Bayesian Network using {methodtype}...")
    start_time = time.time()
    try:
        # Fit the Bayesian network
        DAG = bn.structure_learning.fit(df, methodtype=methodtype)
        model = bn.parameter_learning.fit(DAG, df)
        logging.info(f"[+] {task_name} Bayesian Network created successfully.")
    except Exception as e:
        logging.error(f"[-] Error creating {task_name} Bayesian Network ({methodtype}): {e}")
        raise e

    runtime = time.time() - start_time
    logging.info(f"[+] {task_name} Bayesian Network creation runtime: {runtime:.2f} seconds.")

    # Visualize and save the DAG
    try:
        bn.plot(DAG, params={"title": f"{task_name} Bayesian Network"})
        logging.info(f"[+] {task_name} Network Visualization completed.")
    except Exception as e:
        logging.error(f"[-] Error visualizing the DAG for {task_name}: {e}")

    return model


def save_model(filename, model):
    """
    Save the Bayesian network model to a file.

    Args:
        filename (str): Filepath to save the model.
        model (dict): Bayesian network model.
    """
    logging.info(f"[+] Saving model to {filename}...")
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"[+] Model saved successfully.")
    except Exception as e:
        logging.error(f"[-] Failed to save model: {e}")
        raise e


def evaluate_model(model_name, val_df):
    """
    Load and evaluate a Bayesian network model.

    Args:
        model_name (str): Name of the model file (without extension).
        val_df (DataFrame): Validation dataset.

    Returns:
        None
    """
    logging.info(f"[+] Evaluating {model_name}...")
    try:
        with open(f"{model_name}.pkl", 'rb') as f:
            model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        logging.info(f"[+] {model_name} Evaluation Results:")
        logging.info(f"    Total Test Cases: {total_cases}")
        logging.info(f"    Correct Predictions: {correct_predictions}")
        logging.info(f"    Accuracy: {accuracy:.2f}%")
    except FileNotFoundError as fnfe:
        logging.error(f"[-] Model file not found: {fnfe}")
    except Exception as e:
        logging.error(f"[-] Error evaluating {model_name}: {e}")


#################
## Main Driver ##
#################

def main():
    """
    Main function to execute the entire workflow.
    """
    try:
        # Load datasets
        train_df, val_df = load_data()

        # Task 1: Create Initial Bayesian Network
        logging.info("[+] Starting Task 1: Base Model...")
        base_model = create_network(train_df.copy(), methodtype="hc", task_name="Base Model")
        save_model("models/base_model.pkl", base_model)
        evaluate_model("models/base_model", val_df)
        logging.info("[+] Task 1: Base Model completed successfully.")

        # Task 2: Prune the Bayesian Network
        logging.info("[+] Starting Task 2: Pruned Model...")
        pruned_model = create_network(train_df.copy(), methodtype="tan_hc", task_name="Pruned Model")
        save_model("models/pruned_model.pkl", pruned_model)
        evaluate_model("models/pruned_model", val_df)
        logging.info("[+] Task 2: Pruned Model completed successfully.")

        # Task 3: Optimize the Bayesian Network
        logging.info("[+] Starting Task 3: Optimized Model...")
        optimized_model = create_network(train_df.copy(), methodtype="exhaustive", task_name="Optimized Model")
        save_model("models/optimized_model.pkl", optimized_model)
        evaluate_model("models/optimized_model", val_df)
        logging.info("[+] Task 3: Optimized Model completed successfully.")

        logging.info("[+] All tasks completed successfully.")

    except Exception as e:
        logging.critical(f"[-] Critical failure during execution: {e}")
        raise e


if __name__ == "__main__":
    main()
