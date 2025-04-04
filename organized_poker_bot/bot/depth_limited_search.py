"""
Depth-limited search implementation for poker bot.
This module provides real-time search to refine the blueprint strategy during gameplay.
"""

import random
import numpy as np
import time
# from tqdm import tqdm # tqdm not used here, can be removed if desired
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
# Need GameState if cloning or checking attributes directly
from organized_poker_bot.game_engine.game_state import GameState

class DepthLimitedSearch:
    """
    Depth-limited search for real-time strategy refinement.

    This class implements the depth-limited search algorithm used in Pluribus,
    which allows the bot to refine its strategy in real-time during gameplay.

    Attributes:
        blueprint_strategy: The pre-trained CFR strategy to use as a blueprint
        search_depth: How many betting rounds to look ahead (approximate)
        num_iterations: Number of simulations for Monte Carlo Tree Search part
        exploration_constant: Constant for UCB exploration
        blueprint_weight: Weight given to the blueprint strategy in action selection (not fully implemented here)
    """

    def __init__(self, blueprint_strategy, search_depth=1, num_iterations=100,
                 exploration_constant=1.5, blueprint_weight=0.5): # Adjusted exploration constant default
        """
        Initialize the depth-limited search.

        Args:
            blueprint_strategy (CFRStrategy): The pre-trained CFR strategy object.
            search_depth (int): How many betting rounds to look ahead (approximate, more like tree depth).
            num_iterations (int): Number of MCTS simulations per action.
            exploration_constant (float): UCB exploration constant (e.g., sqrt(2) or 1.5).
            blueprint_weight (float): Weight for blueprint bias (not currently used in UCB here).
        """
        if not isinstance(blueprint_strategy, CFRStrategy):
             raise TypeError("blueprint_strategy must be an instance of CFRStrategy")

        self.blueprint_strategy = blueprint_strategy
        self.search_depth = search_depth # Note: Depth interpretation might need refinement
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.blueprint_weight = blueprint_weight # Not used in current UCB selection logic

        # Initialize search statistics (reset per action)
        self.node_visits = {} # N(s)
        self.action_values = {} # Q(s,a) - total value from simulations starting with s,a
        self.action_visits = {} # N(s,a) - number of times action a was taken from state s

    def get_action(self, game_state, player_idx):
        """
        Get the best action for the current game state using depth-limited search (MCTS-like).

        Args:
            game_state (GameState): The current game state.
            player_idx (int): The index of the player making the decision.

        Returns:
            tuple: The chosen action (action_type, amount).
        """
        # Ensure game state is not terminal
        if game_state.is_terminal():
             print("Warning: get_action called on terminal state in DLS.")
             # Return a default safe action if possible, or None
             return ('check', 0) if ('check', 0) in game_state.get_available_actions() else ('fold', 0)


        # Reset search statistics for this decision
        self.node_visits = {}
        self.action_values = {}
        self.action_visits = {}

        # Root node represents the current state
        root_info_set_key = self.create_info_set_key(game_state, player_idx)
        self.node_visits[root_info_set_key] = 1 # Root visited once initially

        # Get available actions from the root state
        available_actions = game_state.get_available_actions()

        # Handle trivial cases
        if not available_actions:
            print(f"Warning: DLS found no available actions for player {player_idx}. Returning fold.")
            return ('fold', 0)
        if len(available_actions) == 1:
            return available_actions[0]


        # Run MCTS iterations
        for _ in range(self.num_iterations):
            # Clone the game state for simulation (important!)
            sim_state = game_state.clone()
            # Run one simulation from the root
            self._simulate(sim_state, player_idx, self.search_depth)

        # Choose the best action based on visit counts or average value (robustness: visit counts)
        best_action = None
        # Method 1: Choose action with highest average value (Q(s,a) / N(s,a))
        # best_value = float('-inf')
        # for action in available_actions:
        #     action_key = self._get_action_key(root_info_set_key, action)
        #     if action_key in self.action_visits and self.action_visits[action_key] > 0:
        #         value = self.action_values.get(action_key, 0) / self.action_visits[action_key]
        #         if value > best_value:
        #             best_value = value
        #             best_action = action

        # Method 2: Choose action most visited (more robust in MCTS)
        max_visits = -1
        for action in available_actions:
            action_key = self._get_action_key(root_info_set_key, action)
            visits = self.action_visits.get(action_key, 0)
            # print(f"DEBUG DLS: Action {action} visits: {visits} Value: {self.action_values.get(action_key, 0)}") # Optional detailed debug
            if visits > max_visits:
                max_visits = visits
                best_action = action
            elif visits == max_visits and visits > 0:
                 # Tie-breaking? Could use value, or random. Let's stick to first-found for simplicity.
                 pass


        # Fallback if no action was ever visited (should not happen if simulations run)
        if best_action is None:
            print("Warning: DLS failed to select an action (no visits?). Falling back to blueprint.")
            best_action = self.blueprint_strategy.get_action(game_state, player_idx)
            # Ensure fallback is tuple
            if isinstance(best_action, str): best_action = (best_action, 0)


        # Ensure final action is valid tuple format
        if not isinstance(best_action, tuple):
             print(f"Warning: DLS selected non-tuple action {best_action}. Defaulting.")
             best_action = ('fold', 0) # Safer default

        # print(f"DLS selected action: {best_action} with {max_visits} visits.") # Optional debug
        return best_action


    def _simulate(self, sim_state, player_idx_perspective, depth):
        """
        Run one MCTS simulation (Selection, Expansion, Simulation, Backpropagation).

        Args:
            sim_state (GameState): The current simulation game state (will be modified).
            player_idx_perspective (int): The player whose perspective we maximize utility for.
            depth (int): Remaining search depth for this path.

        Returns:
            float: The estimated utility from this simulation path for player_idx_perspective.
        """
        # --- Base Cases (Termination) ---
        if sim_state.is_terminal():
            # Get utility from the perspective of the player making the initial decision
            # Ensure get_utility returns a numeric value
            utility = sim_state.get_utility(player_idx_perspective)
            return utility if isinstance(utility, (int, float)) else 0.0


        if depth <= 0:
            # Reached depth limit, use blueprint rollout for estimation
            # print("DEBUG DLS: Reached depth limit, using blueprint rollout.") # Optional debug
            utility = self._blueprint_rollout(sim_state.clone(), player_idx_perspective) # Rollout on clone
            return utility if isinstance(utility, (int, float)) else 0.0


        # --- Selection/Expansion ---
        # Identify current player for this state
        # --- !!! FIX: Use current_player_idx !!! ---
        current_player_idx = sim_state.current_player_idx
        # --- !!! END FIX !!! ---

        info_set_key = self.create_info_set_key(sim_state, current_player_idx)

        # If node is new (not visited), expand it
        if info_set_key not in self.node_visits:
            # Expand: Add node, simulate from here using blueprint/rollout
            self.node_visits[info_set_key] = 0 # Initialize visits before rollout
            # print(f"DEBUG DLS: Expanding new node {info_set_key}") # Optional debug
            # Rollout from this new node to estimate its value
            value = self._blueprint_rollout(sim_state.clone(), player_idx_perspective)
            value = value if isinstance(value, (int, float)) else 0.0 # Ensure numeric
            self.node_visits[info_set_key] = 1 # Mark as visited once after rollout
            # We don't store value directly for the node, only for (node, action) pairs
            return value

        # Node exists, select action using UCB
        available_actions = sim_state.get_available_actions()
        if not available_actions: # Safety check
             utility = sim_state.get_utility(player_idx_perspective)
             return utility if isinstance(utility, (int, float)) else 0.0


        chosen_action = self._select_action_ucb(sim_state, info_set_key, available_actions, current_player_idx)
        # Ensure chosen_action is valid tuple format
        if not isinstance(chosen_action, tuple): chosen_action = ('fold', 0) # Default if error


        # Apply action to get next state
        # IMPORTANT: apply_action returns a NEW state, use that for recursion
        try:
             next_sim_state = sim_state.apply_action(chosen_action)
        except Exception as e:
             print(f"ERROR during DLS apply_action {chosen_action}: {e}. State:\n{sim_state}")
             # Cannot continue this path, return 0 or estimate based on current state?
             return 0.0 # Return neutral value on error


        # --- Simulation (Recursive Call) ---
        # Recursively simulate from the next state
        value = self._simulate(next_sim_state, player_idx_perspective, depth - 1) # Recurse on new state
        value = value if isinstance(value, (int, float)) else 0.0 # Ensure numeric


        # --- Backpropagation ---
        # Update stats for the *chosen action* from the *current node*
        action_key = self._get_action_key(info_set_key, chosen_action)

        # Initialize if action not taken before from this node
        if action_key not in self.action_visits:
            self.action_visits[action_key] = 0
            self.action_values[action_key] = 0.0

        self.action_visits[action_key] += 1
        self.action_values[action_key] += value # Add the resulting utility
        self.node_visits[info_set_key] += 1 # Increment parent node visits

        return value


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
        parent_visits = self.node_visits.get(info_set_key, 1) # Get visits for node 's'

        best_action = None
        best_ucb_score = float('-inf')

        # Handle case where node_visits might be 0 if just expanded? Use 1 to avoid log(0).
        log_parent_visits = np.log(max(1, parent_visits))

        # Ensure available_actions is not empty
        if not available_actions:
             return ('fold', 0) # Or check logic - should have actions if turn is valid


        for action in available_actions:
            action_key = self._get_action_key(info_set_key, action)
            action_visit_count = self.action_visits.get(action_key, 0) # N(s,a)
            action_total_value = self.action_values.get(action_key, 0.0) # Sum of values Q(s,a)

            # If action hasn't been explored, prioritize it (infinite UCB)
            if action_visit_count == 0:
                # print(f"DEBUG UCB: Selecting unexplored action {action}") # Optional debug
                return action

            # Calculate UCB score
            average_value = action_total_value / action_visit_count # Exploitation term
            exploration_term = self.exploration_constant * np.sqrt(log_parent_visits / action_visit_count)

            # Add blueprint bias (Optional, not fully implemented here)
            # blueprint_prob = self.blueprint_strategy.get_probability(game_state, current_player_idx, action) # Needs this method
            # ucb_score = average_value + exploration_term + self.blueprint_weight * blueprint_prob

            ucb_score = average_value + exploration_term

            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        # Fallback if somehow no best action found (e.g., all visits were 0 - shouldn't happen after check above)
        if best_action is None:
            print(f"Warning: UCB failed to select best action from {available_actions}, choosing random.") # Debug
            best_action = random.choice(available_actions) if available_actions else ('fold', 0)


        return best_action


    def _blueprint_rollout(self, sim_state, player_idx_perspective):
        """
        Perform a simulation rollout using the blueprint strategy.

        Args:
            sim_state (GameState): The game state from which to start the rollout (clone recommended).
            player_idx_perspective (int): The player whose utility we want.

        Returns:
            float: The estimated utility from the rollout.
        """
        rollout_depth = 0
        max_rollout_depth = 30 # Limit rollout steps further

        while not sim_state.is_terminal() and rollout_depth < max_rollout_depth:
            # --- !!! FIX: Use current_player_idx !!! ---
            current_player_idx = sim_state.current_player_idx
            # --- !!! END FIX !!! ---

            # Skip inactive/all-in players
            if current_player_idx not in sim_state.active_players or sim_state.player_stacks[current_player_idx] <= 0:
                 # Find next state correctly
                 original_idx = current_player_idx
                 sim_state._move_to_next_player()
                 # Check for infinite loop if move doesn't change player
                 if sim_state.current_player_idx == original_idx:
                      # print("DEBUG Rollout: Stuck on inactive player, ending rollout.") # Debug
                      break # End rollout if stuck
                 continue # Skip to next loop iteration


            # Get action from blueprint strategy for the current player
            try:
                 # Ensure state is valid before passing
                 if current_player_idx < sim_state.num_players:
                      action = self.blueprint_strategy.get_action(sim_state, current_player_idx)
                      if not isinstance(action, tuple): # Ensure format
                           action = (action, 0) if action in ['check','fold'] else ('fold', 0) # Safer default
                 else:
                      print(f"ERROR Rollout: Invalid current_player_idx {current_player_idx}. Defaulting fold.")
                      action = ('fold', 0)

            except Exception as e:
                 print(f"Warning: Blueprint strategy failed in rollout for player {current_player_idx}. Defaulting. Error: {e}")
                 available = sim_state.get_available_actions()
                 action = ('check', 0) if ('check', 0) in available else ('fold', 0)
                 if not available: break


            # Apply the action - creates a new state object
            try:
                sim_state = sim_state.apply_action(action)
            except Exception as e:
                 print(f"ERROR applying blueprint action {action} in rollout by {current_player_idx}: {e}")
                 break # End rollout on error

            rollout_depth += 1


        if rollout_depth >= max_rollout_depth:
            print("Warning: Blueprint rollout reached max depth.")

        # Return the final utility from the perspective of the original player
        utility = sim_state.get_utility(player_idx_perspective)
        return utility if isinstance(utility, (int, float)) else 0.0 # Ensure numeric


    def _get_action_key(self, info_set_key, action):
        """ Creates a unique key for a state-action pair. """
        # Ensure action is tuple for consistent key
        if isinstance(action, str): action = (action, 0)
        action_str = f"{action[0]}_{action[1]}"
        return f"{info_set_key}|A:{action_str}"


    def create_info_set_key(self, game_state, player_idx):
        """
        Create a key for an information set using the blueprint strategy's method.
        Fallback to a basic implementation if method doesn't exist.
        """
        # Prefer using the blueprint's own key generation for consistency
        if hasattr(self.blueprint_strategy, '_create_info_set_key') and \
           callable(self.blueprint_strategy._create_info_set_key):
            try:
                key = self.blueprint_strategy._create_info_set_key(game_state, player_idx)
                return key
            except Exception as e:
                print(f"Warning: Blueprint _create_info_set_key failed: {e}. Using basic key.")


        # Basic fallback implementation (less effective than abstraction)
        try:
            hole_cards = game_state.hole_cards[player_idx] if player_idx < len(game_state.hole_cards) else []
            community_cards = game_state.community_cards
            hole_str = "_".join(sorted(str(c) for c in hole_cards))
            comm_str = "_".join(sorted(str(c) for c in community_cards))
            betting_str = game_state.get_betting_history() # Assumes simple history string
            pos = game_state.get_position(player_idx)
            round_num = game_state.betting_round
            return f"R{round_num}|P{pos}|H:{hole_str}|C:{comm_str}|B:{betting_str}"
        except Exception as e:
             print(f"Error generating basic info set key: {e}")
             # Return a very generic key on error
             return f"ErrorState_R{game_state.betting_round}_P{player_idx}"
