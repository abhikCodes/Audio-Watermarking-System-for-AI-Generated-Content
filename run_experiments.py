#!/usr/bin/env python
import os
import sys
import subprocess
import itertools
import argparse
from pathlib import Path
import datetime
import logging

# Add project root to path to allow importing configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from configuration.config import settings
except ImportError:
    print("Error: Could not import settings from configuration.config.")
    print("Ensure configuration/config.py exists and the script is run from the project root.")
    sys.exit(1)

# Configure logging for the runner script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
        # Optionally add a FileHandler here if needed
    ]
)
logger = logging.getLogger("ExperimentRunner")


# --- Define Hyperparameter Grid ---
# Modify these lists to explore different values
LEARNING_RATES = [0.001, 0.0005]
ALPHAS = [0.05, 0.1, 0.2] # Weight for detection loss
PERTURB_SCALES = [0.001, 0.0005]
PERCEPTUAL_LOSS_TYPES = ['mse', 'spec', 'mel', 'psych'] # Add 'psych' if implemented and desired

# --- Test Mode Grid (Runs faster for verification) ---
# Uses fewer combinations for quick testing
TEST_LEARNING_RATES = [0.001]
TEST_ALPHAS = [0.07]
TEST_PERTURB_SCALES = [0.001]
TEST_PERCEPTUAL_LOSS_TYPES = ['mse']

def run_single_experiment(
    data_dir: str,
    test_dir_for_eval: str,
    lr: float,
    alpha: float,
    perturb_scale: float,
    loss_type: str,
    is_test_mode: bool,
    base_output_dir: Path
) -> bool:
    """Runs train.py and test.py for a single hyperparameter combination."""

    # Create a unique directory name for this run
    run_name = f"lr_{lr}_alpha_{alpha}_ps_{perturb_scale}_loss_{loss_type}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory is created
    output_dir_str = str(output_dir.resolve())

    logger.info(f"--- Starting Experiment: {run_name} ---")
    logger.info(f"Output directory: {output_dir_str}")

    # --- Build Training Command ---
    train_script = os.path.join("training", "train.py")
    train_cmd = [
        sys.executable, # Use the same python interpreter that runs this script
        train_script,
        "--train_dir", data_dir,
        "--output_dir", run_name, # Pass relative name, train.py prepends TRAINED_MODELS_DIR
        "--learning_rate", str(lr),
        "--alpha", str(alpha),
        "--perturb_scale", str(perturb_scale),
        "--perceptual_loss_type", loss_type,
        # Add other args like --batch_size if needed
    ]
    if is_test_mode:
        train_cmd.append("--test")
        # train_cmd.extend(["--test_epochs", "1"]) # test_epochs is handled inside train.py now
        # train_cmd.extend(["--test_samples", "50"]) # test_samples is handled inside train.py now

    # --- Run Training --- 
    logger.info(f"Running training command: {' '.join(train_cmd)}")
    train_process = subprocess.run(train_cmd, capture_output=True, text=True)

    # Log stdout regardless of outcome
    train_stdout_log_path = output_dir / "train_output.log"
    try:
        with open(train_stdout_log_path, "w") as f:
            f.write(train_process.stdout)
        logger.info(f"Training stdout saved to {train_stdout_log_path}")
    except Exception as e:
        logger.error(f"Failed to write training stdout log: {e}")

    # Check for errors
    if train_process.stderr:
        print("--- Training Errors --- ")
        print(train_process.stderr)
        logger.error(f"Training failed for {run_name}. Stderr logged.")
        # Log stderr to a file
        with open(output_dir / "train_error.log", "w") as f:
            f.write(train_process.stderr)
        return False # Indicate failure
    elif train_process.returncode != 0:
        logger.error(f"Training failed for {run_name} with exit code {train_process.returncode}. Stdout logged.")
        return False # Indicate failure
    else:
        logger.info(f"Training completed successfully for {run_name}.")

    # --- Build Testing Command ---
    test_script = os.path.join("training", "test.py")
    test_cmd = [
        sys.executable,
        test_script,
        "--model_dir", output_dir_str, # Pass the full path to the specific run directory
        "--test_dir", test_dir_for_eval,
    ]
    if is_test_mode:
        test_cmd.append("--test")
        # test_cmd.extend(["--test_samples", "50"]) # test_samples is handled inside test.py now

    # --- Run Testing --- 
    logger.info(f"Running testing command: {' '.join(test_cmd)}")
    test_process = subprocess.run(test_cmd, capture_output=True, text=True)

    # Log stdout regardless of outcome
    test_stdout_log_path = output_dir / "test_output.log"
    try:
        with open(test_stdout_log_path, "w") as f:
            f.write(test_process.stdout)
        logger.info(f"Testing stdout saved to {test_stdout_log_path}")
    except Exception as e:
        logger.error(f"Failed to write testing stdout log: {e}")

    # Check for errors
    if test_process.stderr:
        print("--- Testing Errors --- ")
        print(test_process.stderr)
        logger.error(f"Testing failed for {run_name}. Stderr logged.")
        # Log stderr to a file
        with open(output_dir / "test_error.log", "w") as f:
            f.write(test_process.stderr)
        return False # Indicate failure
    elif test_process.returncode != 0:
        logger.error(f"Testing failed for {run_name} with exit code {test_process.returncode}. Stdout logged.")
        return False
    else:
        logger.info(f"Testing completed successfully for {run_name}.")

    logger.info(f"--- Finished Experiment: {run_name} ---")
    return True # Indicate success

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning experiments for audio steganography models.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the main directory containing audio data (e.g., 50_speakers_audio_data).')
    # Retain separate test_dir for evaluation for now, until data splitting utils are implemented
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the directory containing TEST audio data for evaluation.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (uses fewer hyperparameters, passes --test to train/test scripts).')
    parser.add_argument('--skip_failed', action='store_true', help='Continue running experiments even if one fails.')

    args = parser.parse_args()

    # ALWAYS use the main hyperparameter grid
    logger.info("Defining FULL hyperparameter grid.")
    param_grid = list(itertools.product(
        LEARNING_RATES,
        ALPHAS,
        PERTURB_SCALES,
        PERCEPTUAL_LOSS_TYPES
    ))

    # The args.test flag ONLY controls whether train/test run in test mode
    if args.test:
        logger.info("Script run with --test: Passing --test flag to train/test scripts.")
    else:
        logger.info("Script run without --test: train/test scripts will run in full mode.")

    logger.info(f"Total experiments to run: {len(param_grid)}")

    base_output_dir = settings.TRAINED_MODELS_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True) # Ensure base directory exists

    successful_runs = 0
    failed_runs = 0

    for i, params in enumerate(param_grid):
        lr, alpha, perturb_scale, loss_type = params
        logger.info(f"\n===== Running Experiment {i+1}/{len(param_grid)} ====")
        success = run_single_experiment(
            data_dir=args.data_dir, # Use data_dir for training data
            test_dir_for_eval=args.test_dir, # Use separate test_dir for evaluation
            lr=lr,
            alpha=alpha,
            perturb_scale=perturb_scale,
            loss_type=loss_type,
            is_test_mode=args.test,
            base_output_dir=base_output_dir
        )
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            if not args.skip_failed:
                logger.error("Experiment failed. Stopping run because --skip_failed is not set.")
                break
            else:
                logger.warning("Experiment failed. Continuing to next experiment...")

    logger.info("\n===== Experiment Run Summary ====")
    logger.info(f"Total Experiments Attempted: {len(param_grid)}")
    logger.info(f"Successful Runs: {successful_runs}")
    logger.info(f"Failed Runs: {failed_runs}")
    logger.info("===============================")

if __name__ == "__main__":
    main() 