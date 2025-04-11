# --- START OF FILE organized_poker_bot/bot/depth_limited_search.py ---
"""
Depth-limited search implementation for poker bot using MCTS principles.
(Refactored V4: Use shared info_set_util.py for key generation)
"""

import random
import numpy as np
import time
import os
import sys
import traceback

# Imports (Absolute)
try:
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.game_engine.game_state import GameState
    # --- Import the NEW Utility for key generation ---
    from organized_poker_bot.cfr.info_set_util import generate_info_set_key
    # Keep Card import if evaluate_hand uses it? Probably not needed here.
    # from organized_poker_bot.game_engine.card import Card
    # Keep HandEvaluator if blueprint rollout uses it explicitly (unlikely)
    # from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
except ImportError as e:
    print(f"ERROR importing DLS dependencies: {e}")
    sys.exit(1)

class DepthLimitedSearch:
    """
    Depth-limited search for real-time strategy refinement using MCTS principles.
    """

    def __init__(self, blueprint_strategy, search_depth=1, num_iterations=100,
                 exploration_constant=1.414, blueprint_weight=0.0): # blueprint_weight currently unused
        """ Initialize the depth-limited search. """
        if not isinstance(blueprint_strategy, CFRStrategy):
             raise TypeError("blueprint_strategy must be an instance of CFRStrategy")
        if not blueprint_strategy.strategy:
            print("WARN DLS: Initialized with an empty blueprint strategy.")

        self.blueprint_strategy = blueprint_strategy
        self.search_depth = max(1, search_depth) # Ensure depth is at least 1
        self.num_iterations = max(10, num_iterations) # Ensure minimum simulations
        self.exploration_constant = exploration_constant
        # self.blueprint_weight = blueprint_weight # For potential future UCB modification

        # Search statistics (reset per decision)
        self.node_visits = {}   # N(s): visits to info set key s
        self.action_values = {} # Q_sum(s,a): sum of utilities from sims starting with s,a
        self.action_visits = {} # N(s,a): visits to action a from info set s

    def get_action(self, game_state, player_idx, initial_stacks):
        """ Get the best action using DLS (MCTS-like). Needs initial stacks. """
        if game_state.is_terminal():
            # Fallback for terminal state - shouldn't usually be called here
            # Try to return a sensible default if somehow actions are available
            available_term = []
            try:
                available_term = game_state.get_available_actions()
            except Exception:
                 pass # Ignore errors if already terminal
            return available_term[0] if available_term else ('fold', 0) # Or None?

        # Reset stats for this decision point
        self.node_visits = {}
        self.action_values = {}
        self.action_visits = {}

        # --- Use Shared Utility for Root Key ---
        root_info_set_key = None
        available_actions_err = [] # Initialize for error case
        try:
            root_info_set_key = generate_info_set_key(game_state, player_idx)
            if not root_info_set_key:
                raise ValueError("Root key generation failed")
        except Exception as key_err:
             print(f"ERROR DLS Root Key Gen P{player_idx}: {key_err}. Defaulting.")
             try:
                 available_actions_err = game_state.get_available_actions()
             except Exception:
                 pass # Use empty list if actions fail too
             # Default to check if possible, else fold
             return ('check', 0) if ('check', 0) in available_actions_err else ('fold', 0)

        # Initialize root node visits conceptually (actual increments happen in backprop)
        self.node_visits[root_info_set_key] = 0 # Start at 0, increments add up to num_iterations

        # Get available actions & handle trivial cases
        available_actions = []
        try:
             available_actions = game_state.get_available_actions()
        except Exception as e:
             print(f"ERROR DLS get_action: Failed to get available actions: {e}. Defaulting to fold.")
             return ('fold', 0)

        if not available_actions:
             # print("WARN DLS get_action: No available actions, but state not terminal? Defaulting fold.")
             return ('fold', 0) # Should not happen if not terminal
        if len(available_actions) == 1:
             return available_actions[0] # No choice needed

        # --- MCTS Loop ---
        for _ in range(self.num_iterations):
            sim_state = game_state.clone() # Work on a clone for simulation
            # Pass initial stacks to simulation function
            self._simulate(sim_state, player_idx, self.search_depth, initial_stacks)

        # --- Choose Best Action (Most Visited) ---
        best_action = None
        max_visits = -1

        # Ensure we only consider actions that are currently available
        for action in available_actions:
            action_key = self._get_action_key(root_info_set_key, action)
            if action_key is None: continue # Skip if key failed for this action

            visits = self.action_visits.get(action_key, 0)
            if visits > max_visits:
                max_visits = visits
                best_action = action
            # Optional Tie-breaking: Could use Q-value (action_values[key]/visits) if visits are equal
            # elif visits == max_visits and visits > 0:
            #     current_q = self.action_values.get(self._get_action_key(root_info_set_key, best_action), 0) / max_visits
            #     new_q = self.action_values.get(action_key, 0) / visits
            #     if new_q > current_q:
            #         best_action = action

        # Fallback if MCTS failed (no actions visited?)
        if best_action is None:
            print("Warning: DLS failed (no visits?). Falling back to blueprint.")
            # Get action directly from blueprint strategy
            try:
                best_action = self.blueprint_strategy.get_action(game_state, player_idx)
                 # Ensure tuple format for consistency
                if isinstance(best_action, str):
                     best_action = (best_action, 0)
                if not isinstance(best_action, tuple) or best_action not in available_actions:
                     # If blueprint action invalid, use default logic
                     best_action = self.blueprint_strategy._default_strategy(game_state, available_actions)
            except Exception as bp_err:
                 print(f"ERROR DLS: Blueprint fallback failed: {bp_err}. Using basic default.")
                 best_action = self.blueprint_strategy._default_strategy(game_state, available_actions)

        # Final safety check for tuple format
        if not isinstance(best_action, tuple):
            best_action = ('fold', 0) # Ultimate fallback

        # Ensure the chosen action is actually available
        if best_action not in available_actions:
             print(f"WARN DLS: Chosen action {best_action} not in available {available_actions}. Defaulting.")
             # Fallback to the safest available action (check > call > fold > first)
             if ('check', 0) in available_actions: return ('check', 0)
             call_actions = [a for a in available_actions if a[0] == 'call']
             if call_actions: return call_actions[0]
             if ('fold', 0) in available_actions: return ('fold', 0)
             return available_actions[0] # Should be guaranteed to exist if we got here

        return best_action


    def _simulate(self, sim_state, player_idx_perspective, depth, initial_stacks):
        """ Run one MCTS simulation step (Select/Expand/Simulate/Backprop). """
        # Base Cases: Terminal State or Max Depth
        if sim_state.is_terminal():
            utility = 0.0
            try:
                utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            except Exception:
                pass
            return utility

        if depth <= 0:
            utility = 0.0
            try:
                # Rollout uses a clone, no need to clone again here if rollout handles it
                utility = self._blueprint_rollout(sim_state, player_idx_perspective, initial_stacks)
                utility = float(utility) if isinstance(utility, (int,float)) and not (np.isnan(utility) or np.isinf(utility)) else 0.0
            except Exception as rollout_err:
                 print(f"ERROR DLS Rollout: {rollout_err}") # Log rollout error
                 # Fallback to getting utility from current state if rollout fails? Risky. Return 0.
                 utility = 0.0
            return utility

        # Identify acting player
        current_player_idx = sim_state.current_player_idx
        if not (0 <= current_player_idx < sim_state.num_players):
            # print(f"WARN DLS Simulate: Invalid current_player_idx {current_player_idx}. Returning 0.")
            return 0.0 # Invalid state

        # Handle inactive players (folded, all-in, etc.) by advancing state
        # Need a loop in case multiple players are inactive
        while True:
            is_player_valid = (0 <= current_player_idx < sim_state.num_players)
            is_player_active = False
            if is_player_valid:
                 is_folded = sim_state.player_folded[current_player_idx] if current_player_idx < len(sim_state.player_folded) else True
                 is_all_in = sim_state.player_all_in[current_player_idx] if current_player_idx < len(sim_state.player_all_in) else True
                 # Consider player active if not folded AND not all-in
                 is_player_active = not is_folded and not is_all_in

            if not is_player_valid or not is_player_active:
                 if sim_state.is_terminal(): # Check again after potential state change
                     utility = 0.0
                     try:
                         utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                         utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                     except Exception: pass
                     return utility

                 # Try to move to the next player safely
                 original_idx = current_player_idx
                 try:
                     sim_state._move_to_next_player() # This should modify sim_state in place
                 except Exception as move_err:
                     # print(f"WARN DLS Simulate: Error moving to next player: {move_err}")
                     return 0.0 # Cannot proceed

                 current_player_idx = sim_state.current_player_idx # Update current player

                 # If moving didn't change the player, we're stuck (e.g., heads-up all-in) or game ended
                 if current_player_idx == original_idx or sim_state.is_terminal():
                      utility = 0.0
                      try:
                         utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                         utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                      except Exception: pass
                      return utility
            else:
                 break # Found an active player


        # --- Use Shared Utility for Key ---
        info_set_key = None
        try:
             info_set_key = generate_info_set_key(sim_state, current_player_idx)
             if not info_set_key:
                 raise ValueError("Key gen failed in simulate")
        except Exception as key_err_sim:
             # print(f"WARN DLS Simulate Key Gen P{current_player_idx} Depth {depth}: {key_err_sim}") # Reduce noise
             return 0.0 # Cannot proceed

        # Selection / Expansion
        chosen_action = None
        value = 0.0 # Initialize value

        # Use .get() for checking node visits, default to -1 to distinguish unvisited from visited 0 times
        node_visit_count = self.node_visits.get(info_set_key, -1)

        if node_visit_count == -1: # Expand new node (haven't visited this info set before)
            self.node_visits[info_set_key] = 0 # Initialize visits *before* rollout
            # Rollout from the newly encountered state
            try:
                value = self._blueprint_rollout(sim_state.clone(), player_idx_perspective, initial_stacks)
                value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0
            except Exception as rollout_err:
                 print(f"ERROR DLS Rollout (Expand): {rollout_err}")
                 value = 0.0 # Assign neutral value on rollout error
            self.node_visits[info_set_key] = 1 # Mark as visited *after* rollout and value obtained
            # No action was taken *from* this node yet, so return the rollout value.
            # Backpropagation happens in the calling frame for the action *leading* here.
            return value
        else: # Node visited before, use UCB to select next action
            available_actions = []
            try:
                available_actions = sim_state.get_available_actions()
            except Exception as action_err:
                 print(f"ERROR DLS Simulate: Failed getting actions for UCB: {action_err}")
                 # Cannot select an action, evaluate terminal state if possible
                 utility = 0.0
                 try:
                      utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                      utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                 except Exception: pass
                 return utility

            # Handle case where no actions are possible (e.g., all-in state reached)
            if not available_actions:
                utility = 0.0
                try:
                    utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                    utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                except Exception: pass
                return utility

            # Select action using UCB1
            chosen_action = self._select_action_ucb(sim_state, info_set_key, available_actions, current_player_idx)
            if not isinstance(chosen_action, tuple) or chosen_action not in available_actions:
                 # If UCB selection failed or returned invalid action, default safely
                 print(f"WARN DLS Simulate: UCB returned invalid action {chosen_action}. Defaulting.")
                 chosen_action = self.blueprint_strategy._default_strategy(sim_state, available_actions)

        # We have chosen an action, apply it
        next_sim_state = None
        try:
            next_sim_state = sim_state.apply_action(chosen_action)
        except Exception as apply_err:
            print(f"ERROR DLS Simulate: Failed to apply action {chosen_action}: {apply_err}")
            # Cannot proceed down this path, return neutral value
            # Don't backpropagate this failed attempt
            return 0.0

        # Simulation (Recursive Call) - Pass initial_stacks down
        # Value is the result returned from the recursive call (or rollout)
        value = self._simulate(next_sim_state, player_idx_perspective, depth - 1, initial_stacks)
        value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0

        # Backpropagation for the chosen_action taken from info_set_key
        action_key = self._get_action_key(info_set_key, chosen_action)
        if action_key: # Only backpropagate if key generation succeeds
            # Initialize stats if first time taking this action from this state
            if action_key not in self.action_visits:
                self.action_visits[action_key] = 0
                self.action_values[action_key] = 0.0
            # Update counts and values
            self.action_visits[action_key] += 1
            self.action_values[action_key] += value # Add simulation result to total value
            # Increment visits for the parent node (info_set_key) as well
            self.node_visits[info_set_key] = self.node_visits.get(info_set_key, 0) + 1

        return value # Return the result up the recursion tree


    def _select_action_ucb(self, game_state, info_set_key, available_actions, current_player_idx):
        """ Select action using UCB1 formula. """
        # N(s), ensure it's at least 1 for log calculation, use .get with default
        parent_visits = self.node_visits.get(info_set_key, 1)
        # Ensure visits >= 1 before log
        log_parent_visits = np.log(max(1, parent_visits))

        best_action = None
        best_ucb_score = float('-inf')

        if not available_actions: # Safety check
             return ('fold', 0)

        unvisited_actions = []
        for action in available_actions:
             action_key = self._get_action_key(info_set_key, action)
             if not action_key: continue # Skip if key invalid

             action_visit_count = self.action_visits.get(action_key, 0) # N(s,a)

             if action_visit_count == 0:
                 # Prioritize exploring unvisited actions immediately
                 unvisited_actions.append(action)

        # If there are unvisited actions, choose one randomly to explore
        if unvisited_actions:
            return random.choice(unvisited_actions)

        # If all actions have been visited at least once, use UCB formula
        for action in available_actions:
            action_key = self._get_action_key(info_set_key, action)
            if not action_key: continue # Skip if key invalid

            action_visit_count = self.action_visits.get(action_key, 1) # N(s,a), default 1 to avoid division by zero
            action_total_value = self.action_values.get(action_key, 0.0) # Sum Q(s,a)

            # UCB Score: AvgValue + C * Sqrt(log(N(s)) / N(s,a))
            average_value = action_total_value / action_visit_count
            exploration_term = self.exploration_constant * np.sqrt(log_parent_visits / action_visit_count)
            ucb_score = average_value + exploration_term

            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        # Fallback if no best action found (e.g., all scores are -inf, though unlikely)
        if best_action is None:
            # print("WARN DLS UCB: No best action found via UCB. Choosing randomly.")
            return random.choice(available_actions)
        return best_action

    def _blueprint_rollout(self, sim_state, player_idx_perspective, initial_stacks):
        """ Perform rollout using blueprint strategy. Needs initial stacks. """
        rollout_depth = 0
        max_rollout_depth = 30 # Limit rollout length to prevent infinite loops

        while not sim_state.is_terminal() and rollout_depth < max_rollout_depth:
            current_player_idx = sim_state.current_player_idx

            # --- Refined Inactive Player Check ---
            is_player_valid = (0 <= current_player_idx < sim_state.num_players)
            is_player_active = False
            if is_player_valid:
                 is_folded = sim_state.player_folded[current_player_idx] if current_player_idx < len(sim_state.player_folded) else True
                 is_all_in = sim_state.player_all_in[current_player_idx] if current_player_idx < len(sim_state.player_all_in) else True
                 # Consider player active if not folded AND not all-in
                 is_player_active = not is_folded and not is_all_in

            if not is_player_valid or not is_player_active:
                # Try to move to the next player safely
                original_idx = current_player_idx
                try:
                    sim_state._move_to_next_player() # Modifies sim_state
                except Exception as move_err:
                    # print(f"WARN DLS Rollout: Error moving to next player: {move_err}. Ending rollout.")
                    break # Cannot proceed

                # If moving didn't change the player or game ended, break the loop
                if sim_state.current_player_idx == original_idx or sim_state.is_terminal():
                    break
                continue # Skip to next iteration with the new player

            # --- Get Action for Active Player ---
            action = None
            available_rollout = []
            try:
                available_rollout = sim_state.get_available_actions()
                if not available_rollout: # No actions means game should be over or player all-in handled above
                     break

                # Clone *only* for get_action if it modifies state, otherwise avoid clone
                action = self.blueprint_strategy.get_action(sim_state, current_player_idx)

                # Validate and format action
                if isinstance(action, str) and action in ['fold','check']:
                    action = (action, 0)
                if not isinstance(action, tuple) or action not in available_rollout:
                     # If blueprint action invalid/unavailable, use default
                     # print(f"WARN DLS Rollout: Blueprint action {action} invalid/unavailable. Using default.")
                     action = self.blueprint_strategy._default_strategy(sim_state, available_rollout)

            except Exception as get_action_err:
                print(f"ERROR DLS Rollout: get_action failed: {get_action_err}. Using default.")
                # Fallback using default strategy logic
                action = self.blueprint_strategy._default_strategy(sim_state, available_rollout)

            # Final check if action is valid before applying
            if action is None or action not in available_rollout:
                print(f"ERROR DLS Rollout: No valid action found/chosen ({action}). Ending rollout.")
                break # Stop if no valid action

            # Apply action
            try:
                sim_state = sim_state.apply_action(action) # Use returned new state
            except Exception as apply_err:
                print(f"ERROR DLS Rollout: Applying action {action} failed: {apply_err}. Ending rollout.")
                break # Stop rollout if action application fails
            rollout_depth += 1

        # --- Return Final Utility ---
        # Return final utility using initial stacks
        utility = 0.0
        try:
            # Check if terminal, otherwise result might be misleading
            if sim_state.is_terminal():
                utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            else:
                 # If rollout ended prematurely (depth limit), result is less certain. Still calculate utility?
                 # For now, calculate utility even if not terminal, represents state value at depth limit.
                 # Consider alternative evaluation (e.g., hand strength) if rollout limit hit often.
                 utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                 utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                 # print(f"WARN DLS Rollout: Reached max depth {max_rollout_depth}. Returning utility {utility} from non-terminal state.")

        except Exception as final_util_err:
            print(f"ERROR DLS Rollout: Failed getting final utility: {final_util_err}")
            utility = 0.0 # Return neutral on error
        return utility

    def _get_action_key(self, info_set_key, action):
        """ Creates unique key for state-action pair for DLS stats. """
        if not info_set_key or not isinstance(info_set_key, str):
            # print("WARN _get_action_key: Invalid info_set_key.")
            return None

        action_tuple = None
        if isinstance(action, str) and action in ['fold', 'check']:
            action_tuple = (action, 0)
        elif isinstance(action, tuple) and len(action) == 2:
            # Basic validation of tuple contents
            if isinstance(action[0], str):
                 try:
                      # Ensure second element is numeric or can be rounded to int
                      amount = int(round(float(action[1])))
                      action_tuple = (action[0], amount)
                 except (ValueError, TypeError):
                      # print(f"WARN _get_action_key: Invalid action amount type: {action[1]}")
                      return None
            else:
                 # print(f"WARN _get_action_key: Invalid action type: {action[0]}")
                 return None
        else:
            # print(f"WARN _get_action_key: Invalid action format: {action}")
            return None

        # If action_tuple is validly created
        if action_tuple:
            try:
                # Create string representation consistently
                action_str = f"{action_tuple[0]}_{action_tuple[1]}"
                return f"{info_set_key}|A:{action_str}"
            except Exception as e: # Catch potential string formatting errors
                 print(f"ERROR _get_action_key: Failed string formatting: {e}")
                 return None
        else: # Should already have returned None if validation failed
             return None

    # --- REMOVED create_info_set_key method ---
    # Logic is now centralized in info_set_util.py

# --- END OF FILE organized_poker_bot/bot/depth_limited_search.py ---
