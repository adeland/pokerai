# --- START OF FILE organized_poker_bot/cfr/cfr_strategy.py ---
"""
CFR strategy implementation for poker games.
This module provides a class for using trained CFR strategies.
(Refactored V3: Use shared info_set_util.py for key generation)
"""

import random
import os
import sys
import pickle # For loading/saving
import numpy as np # Added for np.random.choice
import traceback # Added for detailed error logging

# --- Absolute Imports ---
try:
    # Card Abstraction might not be needed directly anymore
    # from organized_poker_bot.cfr.card_abstraction import CardAbstraction
    # Import the NEW Utility for key generation
    from organized_poker_bot.cfr.info_set_util import generate_info_set_key
    # Keep GameState import if _default_strategy or others need it
    from organized_poker_bot.game_engine.game_state import GameState
except ImportError as e:
    print(f"FATAL Import Error in cfr_strategy.py: {e}")
    print("Ensure 'organized_poker_bot' is in PYTHONPATH or run from root.")
    sys.exit(1)
# --- End Absolute Imports ---

class CFRStrategy:
    """
    A class for loading and using trained CFR average strategies.

    Reads a strategy dictionary mapping information set keys (strings) to
    action probability dictionaries {action_tuple: probability}.

    Attributes:
        strategy (dict): The loaded strategy map.
    """

    def __init__(self):
        """
        Initialize the CFR strategy holder.
        """
        self.strategy = {} # Strategy map loaded from file

    def get_action(self, game_state, player_idx):
        """
        Get a stochastic action based on the loaded strategy for the current game state.

        Args:
            game_state (GameState): The current game state object.
            player_idx (int): The index of the player deciding the action.

        Returns:
            tuple: The chosen action tuple (action_type, amount), or a default
                   action (e.g., check/fold) if the state is not found or invalid.
        """
        # --- Generate InfoSet Key using Utility Function ---
        info_set_key = None # Initialize
        try:
            info_set_key = generate_info_set_key(game_state, player_idx)
            if info_set_key is None: # Check if utility function failed
                raise ValueError("InfoSet key generation returned None")
        except Exception as key_err:
            print(f"WARN CFRStrategy: Failed to generate info key P{player_idx}: {key_err}. Using default action.")
            # Pass available actions to default strategy if possible
            available_actions = None
            try:
                available_actions = game_state.get_available_actions()
            except Exception:
                 pass # Ignore failure to get actions here, default will handle it
            return self._default_strategy(game_state, available_actions) # Fallback if key generation fails

        # --- Strategy Lookup ---
        # Use .get() for safer dictionary access
        action_probs = self.strategy.get(info_set_key)

        # If info set not found in strategy map, or invalid, use default
        if action_probs is None or not isinstance(action_probs, dict) or not action_probs:
            # InfoSet not found or invalid format - Use default strategy
            # Logging this might be useful during debugging initial strategy runs
            # print(f"DEBUG CFRStrategy: Key '{info_set_key}' not found or invalid. Defaulting.")
            available_actions = None
            try:
                available_actions = game_state.get_available_actions()
            except Exception:
                 pass
            return self._default_strategy(game_state, available_actions)

        # --- Choose Action based on Loaded Probabilities ---
        # Ensure the chosen action is valid within the current game context
        return self._choose_action(action_probs, game_state)

    # --- REMOVE _create_info_set_key method ---
    # The logic is now handled by the imported generate_info_set_key utility

    # --- REMOVE _determine_round method ---
    # This logic is also embedded within generate_info_set_key

    def _choose_action(self, action_probs, game_state):
        """
        Choose an action stochastically based on the strategy probabilities,
        ensuring the chosen action is currently available.

        Args:
            action_probs (dict): Dictionary {action_tuple: probability} from strategy map.
            game_state (GameState): The current game state (to check available actions).

        Returns:
            tuple: The chosen, available action tuple.
        """
        # Get currently available actions from the game state
        available_actions = []
        try:
            available_actions = game_state.get_available_actions() # Assumes GameState returns list of tuples
        except Exception as e:
            print(f"WARN _choose_action: Failed to get available actions: {e}. Defaulting.")
            return ('fold', 0) # Default safe action if we can't even get actions

        if not available_actions: # Should ideally not happen if get_action is called correctly
            # print("WARN _choose_action: No available actions found in game state.")
            return ('fold', 0) # Default safe action

        # Filter the strategy probabilities to include only available actions
        valid_actions_dict = {}
        total_prob_available = 0.0
        for action_tuple, prob in action_probs.items():
            # Important: Check if the *exact* action tuple exists in available_actions
            # Handle potential type mismatches defensively (e.g., ensure action_tuple is hashable)
            try:
                if action_tuple in available_actions and isinstance(prob, (int, float)) and prob > 1e-9: # Use small threshold, check prob type
                    valid_actions_dict[action_tuple] = prob
                    total_prob_available += prob
            except TypeError:
                # Handle cases where action_tuple might not be hashable (e.g., if format is wrong)
                # print(f"WARN _choose_action: Non-hashable action tuple encountered: {action_tuple}")
                pass


        # If no available actions match the strategy (or sum is zero), use default
        if not valid_actions_dict or total_prob_available <= 1e-9: # Use threshold
            # print(f"WARN _choose_action: No overlap between strategy and available actions, or zero probability sum. Strategy: {action_probs}, Available: {available_actions}. Defaulting.")
            return self._default_strategy(game_state, available_actions) # Pass available for better default

        # Normalize probabilities of available actions
        normalized_probs = []
        action_list = list(valid_actions_dict.keys())
        # Ensure action_list is not empty before proceeding
        if not action_list:
             print(f"ERROR _choose_action: action_list became empty after filtering. Defaulting.")
             return self._default_strategy(game_state, available_actions)

        for action in action_list:
            # Ensure probability is valid before appending
            prob = valid_actions_dict.get(action, 0.0)
            # Handle potential division by zero or invalid probs
            if total_prob_available > 1e-9:
                normalized_probs.append(prob / total_prob_available)
            else:
                 # Should not happen if checked earlier, but as a safeguard
                 normalized_probs.append(0.0)

        # Ensure probabilities sum close to 1 before passing to choice
        prob_sum = sum(normalized_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            # If sum is significantly off, re-normalize or default to uniform/deterministic
            # print(f"WARN _choose_action: Normalized probs sum to {prob_sum}. Re-normalizing/Defaulting.")
            # Option 1: Simple re-normalization (if sum > 0)
            if prob_sum > 1e-9:
                 normalized_probs = [p / prob_sum for p in normalized_probs]
            # Option 2: Default to uniform probability among valid actions
            else:
                 num_valid = len(action_list)
                 normalized_probs = [1.0 / num_valid] * num_valid
                 if not normalized_probs: # If action_list was empty somehow
                      print(f"ERROR _choose_action: No valid actions for uniform fallback. Defaulting.")
                      return self._default_strategy(game_state, available_actions)


        # Choose action stochastically based on normalized probabilities
        try:
            # Use numpy's choice for potentially better handling of floating point sums
            chosen_action_index = np.random.choice(len(action_list), p=normalized_probs)
            chosen_action = action_list[chosen_action_index]
            # Final check: ensure the chosen action is still in available_actions (sanity check)
            if chosen_action not in available_actions:
                 print(f"ERROR _choose_action: Chosen action {chosen_action} not in available {available_actions}. Defaulting.")
                 return self._default_strategy(game_state, available_actions)
            return chosen_action
        except ValueError as ve: # Catch specific error if probs don't sum to 1 or contain negatives
            print(f"ERROR _choose_action: ValueError during np.random.choice: {ve}. Probs={normalized_probs}. Defaulting.")
            return self._default_strategy(game_state, available_actions)
        except Exception as e: # Catch other potential errors
            print(f"ERROR _choose_action: Failed random choice: {e}. Probs={normalized_probs}. Defaulting.")
            traceback.print_exc()
            return self._default_strategy(game_state, available_actions)

    def _default_strategy(self, game_state, available_actions=None):
        """
        Fallback strategy when an info set is not found or no valid action exists.
        Prioritizes check/call over folding.

        Args:
            game_state (GameState): The current game state (might be used if actions not passed).
            available_actions (list, optional): Pre-fetched list of available actions.

        Returns:
            tuple: The chosen default action.
        """
        if available_actions is None:
            # Get actions if not provided (slightly less efficient)
            try:
                available_actions = game_state.get_available_actions()
            except Exception as e:
                 print(f"WARN _default_strategy: Failed to get available actions: {e}. Defaulting to fold.")
                 available_actions = [] # Ensure it's an empty list

        if not available_actions or not isinstance(available_actions, list): # No actions possible or invalid type
            return ('fold', 0) # Or perhaps None if the game state is inconsistent

        # Ensure actions are in the expected tuple format if possible
        valid_formatted_actions = []
        for act in available_actions:
             if isinstance(act, tuple) and len(act) == 2:
                  valid_formatted_actions.append(act)
        available_actions = valid_formatted_actions # Use only validly formatted actions

        if not available_actions: # If filtering removed all actions
             return ('fold', 0)

        # Check if check is available
        check_action = ('check', 0)
        if check_action in available_actions:
            return check_action

        # Check if call is available (find any call action, regardless of amount)
        # Ensure action comparison is robust
        call_actions = [a for a in available_actions if isinstance(a[0], str) and a[0].lower() == 'call']
        if call_actions:
            # Prefer the call action with the correct amount if possible,
            # otherwise just take the first one found. GameState should provide the correct one.
            return call_actions[0] # Return the first available call action

        # If check/call not available, default to folding (safest)
        fold_action = ('fold', 0)
        if fold_action in available_actions:
             return fold_action

        # If somehow fold is also unavailable, return the first action in the list
        # This should only happen in very strange edge cases or forced actions
        # print(f"WARN _default_strategy: No check/call/fold found. Returning first available action: {available_actions[0]}")
        return available_actions[0]

    def save(self, filename):
        """
        Save the strategy dictionary to a file using pickle.

        Args:
            filename (str): Path to the save file.
        """
        try:
            # Ensure directory exists
            dir_name = os.path.dirname(filename)
            if dir_name: # Only create if path includes a directory
                os.makedirs(dir_name, exist_ok=True)
            # Save using highest protocol
            with open(filename, 'wb') as f:
                pickle.dump(self.strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Strategy saved successfully to {filename}")
        except (OSError, pickle.PicklingError) as e: # Catch specific file/pickle errors
            print(f"ERROR saving strategy to {filename}: {e}")
        except Exception as e: # Catch unexpected errors
             print(f"ERROR saving strategy to {filename}: Unexpected error - {e}")
             traceback.print_exc()


    def load(self, filename):
        """
        Load a strategy dictionary from a pickle file.

        Args:
            filename (str): Path to the strategy file.

        Returns:
             bool: True if loading was successful, False otherwise.
        """
        if not os.path.exists(filename):
            print(f"ERROR loading strategy: File not found at {filename}")
            self.strategy = {} # Ensure strategy is empty if load fails
            return False

        try:
            with open(filename, 'rb') as f:
                loaded_strategy = pickle.load(f)
            # Basic validation: Ensure it's a dictionary
            if isinstance(loaded_strategy, dict):
                self.strategy = loaded_strategy
                print(f"Strategy loaded successfully from {filename} ({len(self.strategy):,} info sets)")
                return True
            else:
                print(f"ERROR loading strategy: Loaded object is not a dict (Type: {type(loaded_strategy)})")
                self.strategy = {}
                return False
        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as e: # Catch common pickle errors
            print(f"ERROR loading strategy from {filename}: Invalid pickle format or data - {e}")
            self.strategy = {}
            return False
        except Exception as e: # Catch other unexpected errors
            print(f"ERROR loading strategy from {filename}: Unexpected error - {e}")
            traceback.print_exc()
            self.strategy = {}
            return False

    def __str__(self):
        """ String representation showing number of info sets loaded. """
        # Ensure self.strategy is a dict before calling len
        count = len(self.strategy) if isinstance(self.strategy, dict) else 0
        return f"CFRStrategy({count:,} info sets loaded)"

# --- END OF FILE organized_poker_bot/cfr/cfr_strategy.py ---
