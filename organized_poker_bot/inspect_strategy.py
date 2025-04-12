import pickle
import os
import sys

# --- Configuration ---
# Relative path from where you run this script to the models directory
MODEL_DIR_RELATIVE_PATH = "models/6max_test_run"
# Name of the pickle file to inspect
PICKLE_FILENAME = "partial_strategy_iter_0.pkl"
# How many info sets to sample and print
NUM_SAMPLES_TO_PRINT = 5
# --- End Configuration ---

def inspect_pickle_file(filepath):
    """Loads and inspects the contents of a pickle file."""

    print(f"Attempting to load strategy from: {filepath}")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found at '{filepath}'")
        return

    loaded_strategy = None
    try:
        with open(filepath, 'rb') as f:
            # Security Warning: Only load trusted pickle files.
            loaded_strategy = pickle.load(f)
        print("File loaded successfully.")

    except EOFError:
        print("ERROR: Failed to load - End Of File error. File might be empty or corrupt.")
        return
    except pickle.UnpicklingError:
        print("ERROR: Failed to load - Unpickling error. File is likely corrupt or not a valid pickle.")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during loading: {type(e).__name__}: {e}")
        return

    if loaded_strategy is None:
        print("Loading completed, but the result is None.")
        return

    # --- Inspection ---
    print("-" * 50)
    print("Data Inspection:")
    print("-" * 50)

    # 1. Check the type of the loaded data
    data_type = type(loaded_strategy)
    print(f"Loaded data type: {data_type}")

    if isinstance(loaded_strategy, dict):
        # 2. Get the number of information sets (keys)
        num_info_sets = len(loaded_strategy)
        print(f"Number of information sets (keys): {num_info_sets}")

        if num_info_sets == 0:
            print("The strategy dictionary is empty.")
            return

        # 3. Print a sample of the keys and their values
        print(f"\nPrinting details for up to {NUM_SAMPLES_TO_PRINT} sample information sets:")
        count = 0
        for key, value in loaded_strategy.items():
            if count >= NUM_SAMPLES_TO_PRINT:
                break

            print("\n--- Sample ---")
            print(f"Information Set Key:\n  '{key}'")

            # Value should be another dictionary: {action_tuple: probability}
            print(f"Value Type: {type(value)}")
            if isinstance(value, dict):
                print("Action Probabilities:")
                if not value:
                    print("  (Empty dictionary for this key)")
                else:
                    for action_tuple, probability in value.items():
                        # Format for readability
                        action_type = action_tuple[0] if isinstance(action_tuple, tuple) else str(action_tuple)
                        action_amount = action_tuple[1] if isinstance(action_tuple, tuple) and len(action_tuple) > 1 else ''
                        action_str = f"{action_type}{action_amount}" if action_amount != 0 else action_type
                        print(f"  - Action: {action_str:<10} | Probability: {probability:.4f}")
            else:
                print(f"Value Content (truncated): {str(value)[:200]}") # Show a snippet if not a dict

            count += 1

    elif isinstance(loaded_strategy, list):
        print(f"Number of items in list: {len(loaded_strategy)}")
        print("\nPrinting first few items:")
        for i, item in enumerate(loaded_strategy[:NUM_SAMPLES_TO_PRINT]):
            print(f"Item {i}: {str(item)[:200]}") # Truncate long items

    else:
        # Handle other data types if necessary
        print("\nLoaded data content (truncated):")
        print(str(loaded_strategy)[:500]) # Print the first 500 chars

    print("-" * 50)
    print("Inspection Complete.")

# --- Main Execution ---
if __name__ == "__main__":
    # Construct the full path
    # Assumes the script is run from the project root where main.py is
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir # Adjust if inspect_strategy.py is elsewhere
    full_filepath = os.path.join(project_root, MODEL_DIR_RELATIVE_PATH, PICKLE_FILENAME)

    inspect_pickle_file(full_filepath)
