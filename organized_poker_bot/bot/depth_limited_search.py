# --- START OF FILE organized_poker_bot/bot/depth_limited_search.py ---
"""
Depth-limited search implementation for poker bot.
This module provides real-time search to refine the blueprint strategy during gameplay.
(Refactored V2: Pass initial_stacks for utility calculation)
"""

import random
import numpy as np
import time
import os
import sys
import traceback # For potential error printing

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
try:
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    # Need GameState if cloning or checking attributes directly
    from organized_poker_bot.game_engine.game_state import GameState
except ImportError:
    print("ERROR importing DLS dependencies.")
    sys.exit(1)

class DepthLimitedSearch:
    """
    Depth-limited search for real-time strategy refinement using MCTS principles.

    Attributes:
        blueprint_strategy: The pre-trained CFR strategy to use as a blueprint.
        search_depth: How many decision nodes deep to search (approx).
        num_iterations: Number of MCTS simulations per decision.
        exploration_constant: Constant for UCB exploration.
        # Other potential attributes (blueprint_weight, etc.)
    """

    def __init__(self, blueprint_strategy, search_depth=1, num_iterations=100,
                 exploration_constant=1.414, blueprint_weight=0.0): # blueprint_weight unused for now
        """
        Initialize the depth-limited search.

        Args:
            blueprint_strategy (CFRStrategy): Pre-trained strategy object.
            search_depth (int): Lookahead depth (tree depth).
            num_iterations (int): MCTS simulations per action.
            exploration_constant (float): UCB exploration constant (e.g., sqrt(2)).
        """
        if not isinstance(blueprint_strategy, CFRStrategy):
             raise TypeError("blueprint_strategy must be an instance of CFRStrategy")
        if not blueprint_strategy.strategy:
            print("WARN: DLS initialized with an empty blueprint strategy.")

        self.blueprint_strategy = blueprint_strategy
        self.search_depth = search_depth
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.blueprint_weight = blueprint_weight # Not currently used in UCB logic

        # Initialize search statistics (reset per action decision)
        self.node_visits = {}   # N(s): visits to info set key s
        self.action_values = {} # Q(s,a): sum of utilities from simulations after s,a
        self.action_visits = {} # N(s,a): visits to action a from info set s

    # --- MODIFIED get_action ---
    def get_action(self, game_state, player_idx, initial_stacks): # ADD initial_stacks arg
        """
        Get the best action for the current game state using depth-limited search (MCTS-like).

        Args:
            game_state (GameState): The current game state.
            player_idx (int): The index of the player making the decision.
            initial_stacks (list): Stacks at the START of the current hand.

        Returns:
            tuple: The chosen action (action_type, amount).
        """
        if game_state.is_terminal():
             # print("Warning: DLS get_action called on terminal state.") # Reduce noise
             available_actions = game_state.get_available_actions() # May return actions even if terminal sometimes?
             return ('check', 0) if ('check', 0) in available_actions else (('fold',0) if ('fold',0) in available_actions else available_actions[0] if available_actions else None)


        # Reset search statistics for this decision point
        self.node_visits = {}
        self.action_values = {}
        self.action_visits = {}

        # Identify the root node (current infoset)
        root_info_set_key = self.create_info_set_key(game_state, player_idx)
        if not root_info_set_key: # Handle potential key generation error
             print("ERROR DLS: Could not create root info set key. Defaulting.")
             available_actions_err = game_state.get_available_actions()
             return ('check', 0) if ('check', 0) in available_actions_err else ('fold', 0)

        self.node_visits[root_info_set_key] = 1 # Initialize root visit count

        # Get available actions from the root state
        available_actions = game_state.get_available_actions()

        # Handle trivial cases
        if not available_actions:
            # print(f"Warning: DLS found no available actions for player {player_idx}. Returning fold.") # Reduce noise
            return ('fold', 0)
        if len(available_actions) == 1:
            return available_actions[0] # Only one option

        # Run MCTS iterations
        for _ in range(self.num_iterations):
            # IMPORTANT: Clone the game state for simulation!
            sim_state = game_state.clone()
            # Run one simulation from the root, passing initial_stacks
            self._simulate(sim_state, player_idx, self.search_depth, initial_stacks) # PASS initial_stacks

        # Choose the best action based on simulation results (most visited is robust)
        best_action = None
        max_visits = -1
        for action in available_actions:
            # Ensure action key generation is robust
            action_key = self._get_action_key(root_info_set_key, action)
            visits = self.action_visits.get(action_key, 0)
            if visits > max_visits:
                max_visits = visits
                best_action = action

        # Fallback if no action was ever visited (should not happen if simulations run)
        if best_action is None:
            print("Warning: DLS failed to select an action (no visits?). Falling back to blueprint.")
            best_action = self.blueprint_strategy.get_action(game_state, player_idx) # Get from blueprint
            # Ensure fallback action is a valid tuple
            if isinstance(best_action, str): best_action = (best_action, 0)
            if not isinstance(best_action, tuple): best_action = ('fold', 0) # Default to fold

        # Ensure final action format is tuple
        if not isinstance(best_action, tuple):
             # print(f"Warning: DLS selected non-tuple action {best_action}. Defaulting.") # Reduce noise
             best_action = ('fold', 0)

        # Optional debug print:
        # print(f"DLS P{player_idx} selected {best_action} (Visits: {max_visits}) | RootKey: {root_info_set_key}")

        return best_action


    # --- MODIFIED _simulate ---
    def _simulate(self, sim_state, player_idx_perspective, depth, initial_stacks): # ADD initial_stacks arg
        """
        Run one MCTS simulation step (Selection, Expansion, Simulation, Backpropagation).

        Args:
            sim_state (GameState): The current simulation game state (will be modified).
            player_idx_perspective (int): The player whose perspective we maximize utility for.
            depth (int): Remaining search depth for this path.
            initial_stacks (list): Initial stacks for the hand (for get_utility).

        Returns:
            float: The estimated utility from this simulation path for player_idx_perspective.
        """
        # --- Base Cases (Termination) ---
        if sim_state.is_terminal():
            utility = 0.0
            try:
                 # USE initial_stacks for terminal utility calculation
                 utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                 utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            except Exception: pass # Default utility to 0 on error
            return utility

        if depth <= 0:
            # Reached depth limit, use blueprint rollout for estimation
            # PASS initial_stacks to blueprint rollout
            utility = self._blueprint_rollout(sim_state.clone(), player_idx_perspective, initial_stacks) # Rollout on clone
            return float(utility) if isinstance(utility, (int, float)) and not (np.isnan(utility) or np.isinf(utility)) else 0.0

        # --- Selection/Expansion ---
        current_player_idx = sim_state.current_player_idx
        # Handle invalid player index (should ideally not happen if state transitions are correct)
        if not (0 <= current_player_idx < sim_state.num_players):
             # print(f"WARN DLS simulate: Invalid current_player_idx {current_player_idx}. Returning 0.") # Reduce noise
             return 0.0

        # Create InfoSet Key for the current state and acting player
        info_set_key = self.create_info_set_key(sim_state, current_player_idx)
        if not info_set_key: return 0.0 # Return 0 if key fails

        # If node is new (not visited), expand it using rollout
        if info_set_key not in self.node_visits:
            self.node_visits[info_set_key] = 0 # Initialize visits before rollout
            # PASS initial_stacks to rollout
            value = self._blueprint_rollout(sim_state.clone(), player_idx_perspective, initial_stacks)
            value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0
            self.node_visits[info_set_key] = 1 # Mark as visited once after rollout
            return value # Return the rollout value directly for expansion

        # Node exists, select action using UCB1
        available_actions = sim_state.get_available_actions()
        if not available_actions: # Safety check, e.g., player is all-in but state not terminal yet?
             utility = 0.0
             try: utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks); utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
             except Exception: pass
             return utility

        chosen_action = self._select_action_ucb(sim_state, info_set_key, available_actions, current_player_idx)
        # Ensure chosen_action is valid tuple format
        if not isinstance(chosen_action, tuple): chosen_action = ('fold', 0) # Safer default

        # Apply action to get next state
        try:
             next_sim_state = sim_state.apply_action(chosen_action)
        except Exception as e:
             # print(f"ERROR DLS apply_action {chosen_action} for P{current_player_idx}: {e}") # Reduce noise
             return 0.0 # Return neutral value on error

        # --- Simulation (Recursive Call) ---
        # Recursively simulate from the next state, passing initial_stacks down
        value = self._simulate(next_sim_state, player_idx_perspective, depth - 1, initial_stacks) # Recurse
        value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0 # Ensure numeric

        # --- Backpropagation ---
        # Update stats for the *chosen action* from the *current node*
        action_key = self._get_action_key(info_set_key, chosen_action)
        if not action_key: return value # Stop backpropagation if action key fails

        # Initialize stats for action if not seen before from this node
        if action_key not in self.action_visits:
            self.action_visits[action_key] = 0
            self.action_values[action_key] = 0.0

        # Update visits and total value (sum of results)
        self.action_visits[action_key] += 1
        self.action_values[action_key] += value
        self.node_visits[info_set_key] += 1 # Increment parent node visits

        return value # Return the result of the simulation


    def _select_action_ucb(self, game_state, info_set_key, available_actions, current_player_idx):
        """
        Select an action using UCB1 formula.

        Args:
            game_state (GameState): Current game state.
            info_set_key (str): Key for the current information set (node).
            available_actions (list): List of valid actions (tuples).
            current_player_idx (int): Index of the player choosing the action.

        Returns:
            tuple: The chosen action.
        """
        parent_visits = self.node_visits.get(info_set_key, 1) # Get visits N(s)
        # Use log of parent visits, ensure it's at least log(1)=0 to avoid log(0)
        log_parent_visits = np.log(max(1, parent_visits))

        best_action = None
        best_ucb_score = float('-inf')

        # If no actions available (should be caught earlier, but for safety)
        if not available_actions: return ('fold', 0)

        for action in available_actions:
            action_key = self._get_action_key(info_set_key, action)
            action_visit_count = self.action_visits.get(action_key, 0) # N(s,a)
            action_total_value = self.action_values.get(action_key, 0.0) # Q_sum(s,a)

            # If action hasn't been explored, prioritize it (UCB is infinite)
            if action_visit_count == 0:
                return action

            # Calculate UCB score: Q(s,a)/N(s,a) + C * sqrt(log(N(s)) / N(s,a))
            average_value = action_total_value / action_visit_count # Exploitation term
            exploration_term = self.exploration_constant * np.sqrt(log_parent_visits / action_visit_count)
            ucb_score = average_value + exploration_term

            # Track best score and action
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        # Fallback if somehow no best action found (e.g., parent_visits was 0?)
        if best_action is None:
            # print(f"Warning: UCB failed to select best action from {available_actions}, choosing random.") # Reduce noise
            best_action = random.choice(available_actions) if available_actions else ('fold', 0)

        return best_action

    # --- MODIFIED _blueprint_rollout ---
    def _blueprint_rollout(self, sim_state, player_idx_perspective, initial_stacks): # ADD initial_stacks arg
        """
        Perform a simulation rollout using the blueprint strategy.

        Args:
            sim_state (GameState): The game state to start rollout from (use clone).
            player_idx_perspective (int): The player whose utility we want.
            initial_stacks (list): Initial stacks for the hand (for get_utility).

        Returns:
            float: The estimated utility from the rollout.
        """
        rollout_depth = 0
        max_rollout_depth = 30 # Limit rollout steps

        while not sim_state.is_terminal() and rollout_depth < max_rollout_depth:
            current_player_idx = sim_state.current_player_idx

            # Skip inactive/all-in players more robustly
            is_player_valid = (0 <= current_player_idx < sim_state.num_players)
            is_player_in_hand = False
            if is_player_valid:
                 is_folded = sim_state.player_folded[current_player_idx] if current_player_idx < len(sim_state.player_folded) else True
                 is_all_in = sim_state.player_all_in[current_player_idx] if current_player_idx < len(sim_state.player_all_in) else True
                 has_stack = sim_state.player_stacks[current_player_idx] > 0.01 if current_player_idx < len(sim_state.player_stacks) else False
                 is_player_in_hand = not is_folded and has_stack # Active = Not folded & has chips
            else: is_player_in_hand = False

            # If current player is invalid or cannot act, try to move to the next one
            if not is_player_valid or not is_player_in_hand:
                 original_idx = current_player_idx
                 sim_state._move_to_next_player()
                 if sim_state.current_player_idx == original_idx: break # Avoid loop if stuck
                 continue # Try next player in the loop

            # Get action from blueprint strategy
            action = None
            try:
                 action = self.blueprint_strategy.get_action(sim_state.clone(), current_player_idx) # Pass clone to blueprint
                 # Ensure format consistency
                 if isinstance(action, str) and action in ['fold','check']: action = (action, 0)
                 elif not isinstance(action, tuple) or len(action)!=2: action = ('fold', 0) # Default if format wrong
            except Exception as e:
                 # print(f"Warning: Blueprint strategy failed in rollout for P{current_player_idx}: {e}") # Reduce noise
                 available = sim_state.get_available_actions()
                 action = ('check', 0) if ('check', 0) in available else (('fold', 0) if ('fold', 0) in available else None)
                 if action is None and available: action = available[0]
                 elif action is None: break # Cannot continue

            if action is None: # Safety break if no action determined
                # print(f"WARN Rollout: No action determined for P{current_player_idx}.")
                break

            # Apply the action - returns a new state object
            try:
                sim_state = sim_state.apply_action(action)
            except Exception as e:
                 # print(f"ERROR applying blueprint action {action} in rollout: {e}")
                 break # End rollout on error

            rollout_depth += 1

        # Get final utility from the perspective player - USE initial_stacks
        utility = 0.0
        try:
            utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
            utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        except Exception: pass # Default 0 on error
        return utility

    def _get_action_key(self, info_set_key, action):
        """ Creates a unique key for a state-action pair for stat tracking. """
        # Ensure action is consistently formatted tuple
        if isinstance(action, str): action_tuple = (action, 0)
        elif isinstance(action, tuple) and len(action) == 2: action_tuple = action
        else:
            # print(f"WARN DLS: Invalid action format for key: {action}") # Reduce noise
            return None # Cannot create key for invalid action
        try:
            # Use integer amount for discrete key representation
            action_str = f"{action_tuple[0]}_{int(round(action_tuple[1]))}"
            return f"{info_set_key}|A:{action_str}"
        except Exception as e:
             # print(f"WARN DLS: Error creating action key: {e}") # Reduce noise
             return None


    def create_info_set_key(self, game_state, player_idx):
        """
        Create a key for an information set using the blueprint strategy's method.
        Relies on blueprint strategy having a compatible _create_info_set_key method.
        """
        # Prefer using the blueprint's own key generation for consistency
        if hasattr(self.blueprint_strategy, '_create_info_set_key') and \
           callable(self.blueprint_strategy._create_info_set_key):
            try:
                # Assume blueprint method takes (game_state, player_idx)
                key = self.blueprint_strategy._create_info_set_key(game_state, player_idx)
                if isinstance(key, str) and key: return key # Ensure key is a non-empty string
                else: raise ValueError(f"Blueprint key invalid: {key}")
            except Exception as e:
                 print(f"ERROR DLS: Blueprint _create_info_set_key failed: {e}. Traceback below.")
                 traceback.print_exc(limit=2)
                 return f"BP_KEY_ERR_P{player_idx}_R{game_state.betting_round}"

        else:
            print("ERROR DLS: Blueprint strategy missing _create_info_set_key method!")
            # Return a very generic key, signaling the error
            return f"MISSING_KEY_FUNC_P{player_idx}_R{game_state.betting_round}"

# --- END OF FILE organized_poker_bot/bot/depth_limited_search.py ---
