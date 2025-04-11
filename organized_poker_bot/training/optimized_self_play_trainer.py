# --- START OF FILE organized_poker_bot/training/optimized_self_play_trainer.py ---
"""
Optimized self-play training implementation for poker CFR.
(Refactored V8: Correct worker factory call, includes ActionAbstraction handling)
"""

import os
import sys
import pickle
import random
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import traceback
import collections # For defaultdict

# Ensure imports use absolute paths from the project root
try:
    from organized_poker_bot.game_engine.game_state import GameState
    from organized_poker_bot.cfr.information_set import InformationSet
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.cfr.info_set_util import generate_info_set_key
    from organized_poker_bot.cfr.cfr_trainer import CFRTrainer # For REC_LIMIT
    # Import ActionAbstraction if used in worker traversal
    from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
except ImportError as e:
    print(f"FATAL Import Error in optimized_self_play_trainer.py: {e}")
    sys.exit(1)


# --- Worker Function ---
def worker_train_batch(args):
    """ Worker function to run CFR iterations. """
    # Unpack arguments (ensure order matches worker_args in OptimizedSelfPlayTrainer.train)
    game_state_factory_instance, num_players, batch_size, current_iteration_t, \
        _, _, worker_id = args # Unpack proxies even if not used directly

    local_regret_sum = {}
    local_strategy_sum = {}
    worker_failed_hands = 0

    # Seed RNGs for this worker process
    try:
        seed_data = os.urandom(16) + bytes([worker_id % 256])
        random.seed(seed_data)
        np.random.seed(random.randint(0, 2**32 - 1))
    except Exception as seed_err:
        print(f"WARN Worker {worker_id}: Error setting seed: {seed_err}")

    # Simulate batch_size hands
    for hand_idx in range(batch_size):
        game_state = None
        initial_stacks_hand = []

        # --- Setup Hand using factory instance ---
        try:
            # --- Corrected: Call factory with num_players ---
            # Assumes factory is callable and takes num_players
            game_state = game_state_factory_instance(num_players)
            # --- End Correction ---

            # Retrieve initial stacks *after* state creation
            default_stack = 10000.0 # Fallback, should match factory config
            if hasattr(game_state, 'player_stacks') and game_state.player_stacks:
                 initial_stacks_hand = game_state.player_stacks[:] # Use copy
            else:
                 initial_stacks_hand = [default_stack] * num_players
            # Ensure list has correct length and isn't empty
            if not initial_stacks_hand or len(initial_stacks_hand) != num_players:
                 initial_stacks_hand = [default_stack] * num_players

        except TypeError as te:
            # Catch if factory was called incorrectly (e.g., missing num_players arg)
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: TypeError Calling Factory: {te}")
            worker_failed_hands += 1
            continue
        except Exception as e:
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Error Creating State via Factory: {type(e).__name__}: {e}")
            worker_failed_hands += 1
            continue

        # --- Check Initial State ---
        if game_state is None:
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Factory returned None state.")
            worker_failed_hands += 1
            continue
        if game_state.is_terminal():
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Immediately terminal state from factory.")
            worker_failed_hands += 1
            continue
        if game_state.current_player_idx == -1 and not game_state.is_terminal():
            # Allow if hand isn't terminal (e.g., all-in pre-deal) but log warning
             print(f"!!! WARN Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Invalid current_player_idx (-1) from factory but not terminal.")


        # --- Traverse perspectives ---
        initial_reach_probs = np.ones(num_players, dtype=float)
        perspective_failed = False # Flag if any perspective traversal fails
        for p_idx in range(num_players):
            try:
                # Pass clone to traversal, handle local sums
                _worker_cfr_traverse(
                    game_state=game_state.clone(),
                    reach_probs=initial_reach_probs.copy(),
                    perspective_player_idx=p_idx,
                    initial_stacks=initial_stacks_hand[:],
                    current_iteration_t=float(current_iteration_t),
                    local_regret_sum=local_regret_sum,
                    local_strategy_sum=local_strategy_sum,
                    num_players=num_players
                )
            except RecursionError:
                print(f"!!! FAIL Worker {worker_id}: RECURSION LIMIT hit traverse P{p_idx}")
                perspective_failed = True
                break # Stop processing this hand if one perspective hits recursion limit
            except Exception as traverse_e:
                print(f"!!! FAIL Worker {worker_id}: Error TRAVERSING P{p_idx}: {type(traverse_e).__name__}: {traverse_e}")
                perspective_failed = True
                # Optionally break here too, depending on desired error handling

        # If any perspective failed, count the whole hand simulation as failed
        if perspective_failed:
            worker_failed_hands += 1

    # Return accumulated local results and failure count
    return local_regret_sum, local_strategy_sum, worker_failed_hands


# --- Recursive Worker Logic ---
def _worker_cfr_traverse(game_state, reach_probs, perspective_player_idx,
                         initial_stacks, current_iteration_t,
                         local_regret_sum, local_strategy_sum, num_players,
                         current_depth=0): # Add depth tracking
    """ Recursive CFR logic for worker (matches CFRTrainer closely). """
    # Use recursion limit from CFRTrainer class if available, else default
    WORKER_REC_LIMIT = getattr(CFRTrainer, 'RECURSION_DEPTH_LIMIT', 500) - 50 # Use buffer

    # --- Base Cases ---
    if game_state.is_terminal():
        utility = 0.0
        try:
            utility_val = game_state.get_utility(perspective_player_idx, initial_stacks)
            utility = float(utility_val) if isinstance(utility_val, (int, float)) and not np.isnan(utility_val) and not np.isinf(utility_val) else 0.0
        except Exception: pass # Default 0.0
        return utility

    if current_depth > WORKER_REC_LIMIT:
        raise RecursionError(f"Worker depth limit {WORKER_REC_LIMIT} exceeded")

    # --- Handle Inactive Player ---
    acting_player_idx = game_state.current_player_idx
    if not (0 <= acting_player_idx < num_players): return 0.0 # Invalid index

    # Safely check player state
    is_folded = game_state.player_folded[acting_player_idx] if acting_player_idx < len(game_state.player_folded) else True
    is_all_in = game_state.player_all_in[acting_player_idx] if acting_player_idx < len(game_state.player_all_in) else True

    if is_folded or is_all_in:
        temp_state = game_state.clone()
        original_turn_idx = temp_state.current_player_idx
        try: temp_state._move_to_next_player()
        except Exception: return 0.0 # Fail state

        if temp_state.current_player_idx == original_turn_idx or temp_state.is_terminal():
            utility = 0.0
            try: utility_val = temp_state.get_utility(perspective_player_idx, initial_stacks); utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            except Exception: pass
            return utility
        else:
            # Recursive call, increment depth
            return _worker_cfr_traverse(temp_state, reach_probs, perspective_player_idx, initial_stacks, current_iteration_t, local_regret_sum, local_strategy_sum, num_players, current_depth + 1)

    # --- Active Player: Key, Actions, Strategy ---
    try: # Get InfoSet Key
        info_set_key = generate_info_set_key(game_state, acting_player_idx)
        if not info_set_key or not isinstance(info_set_key, str): raise ValueError("Invalid key")
    except Exception: return 0.0

    try: # Get Actions (Handle Abstraction)
        raw_actions = game_state.get_available_actions()
        # *** IMPORTANT: Worker needs consistent abstraction setting ***
        # This should ideally be passed via args or config if it can vary.
        # Assuming a default value for now.
        USE_ACTION_ABSTRACTION_IN_WORKER = True # Replace with actual config if needed
        if USE_ACTION_ABSTRACTION_IN_WORKER:
            available_actions = ActionAbstraction.abstract_actions(raw_actions, game_state)
        else:
            available_actions = raw_actions
    except Exception: return 0.0 # Fail on action error
    if not isinstance(available_actions, list): available_actions = []

    # Handle no available actions
    if not available_actions:
        utility = 0.0
        try: utility_val = game_state.get_utility(perspective_player_idx, initial_stacks); utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        except Exception: pass
        return utility

    # --- Get Strategy from LOCAL sums ---
    regrets = local_regret_sum.get(info_set_key, {}) # Use .get for safety
    strategy = {}
    positive_regret_sum = 0.0
    action_positive_regrets = {}
    for action in available_actions:
        regret = regrets.get(action, 0.0)
        positive_regret = max(0.0, regret)
        action_positive_regrets[action] = positive_regret
        positive_regret_sum += positive_regret
    # Normalize or use uniform
    if positive_regret_sum > 1e-9:
        strategy = {action: action_positive_regrets[action] / positive_regret_sum for action in available_actions}
    else:
        num_act = len(available_actions)
        prob = 1.0 / num_act if num_act > 0 else 0.0
        strategy = {action: prob for action in available_actions}

    # --- Explore Actions ---
    node_utility_perspective = 0.0
    action_utilities_perspective = {}
    for action in available_actions:
        action_prob = strategy.get(action, 0.0)
        if action_prob < 1e-9:
            action_utilities_perspective[action] = None; continue

        try: next_game_state = game_state.apply_action(action)
        except Exception: action_utilities_perspective[action] = None; continue

        # Update Opponent Reach Prob
        next_reach_probs = reach_probs.copy()
        if acting_player_idx != perspective_player_idx:
            prob_factor = float(action_prob) if isinstance(action_prob,(int,float)) and not(np.isnan(action_prob)or np.isinf(action_prob)) else 0.0
            current_reach = float(next_reach_probs[acting_player_idx]) if acting_player_idx < len(next_reach_probs) and isinstance(next_reach_probs[acting_player_idx],(int,float)) and not(np.isnan(next_reach_probs[acting_player_idx])or np.isinf(next_reach_probs[acting_player_idx])) else 0.0
            updated_reach = np.clip(current_reach * prob_factor, 0.0, 1.0)
            next_reach_probs[acting_player_idx] = updated_reach

        # Recursive Call
        try:
            utility_from_action = _worker_cfr_traverse(
                next_game_state, next_reach_probs, perspective_player_idx,
                initial_stacks, current_iteration_t, local_regret_sum,
                local_strategy_sum, num_players, current_depth + 1 # Increment depth
            )
            action_utilities_perspective[action] = utility_from_action
            # Accumulate node utility safely
            if isinstance(utility_from_action, (int, float)) and not (np.isnan(utility_from_action) or np.isinf(utility_from_action)):
                 node_utility_perspective += action_prob * utility_from_action
        except RecursionError as re_inner: raise re_inner # Propagate recursion error
        except Exception: action_utilities_perspective[action] = None # Mark other errors

    # --- Update LOCAL Sums (If Perspective Player Acting) ---
    if acting_player_idx == perspective_player_idx:
        # Calculate values safely
        safe_reach = np.nan_to_num(reach_probs, nan=0.0, posinf=0.0, neginf=0.0)
        opp_reach_prod = 1.0
        if num_players > 1:
            opp_reaches = [safe_reach[p] for p in range(num_players) if p != perspective_player_idx]
            try: temp_prod = np.prod(opp_reaches) if opp_reaches else 1.0; opp_reach_prod = float(temp_prod) if isinstance(temp_prod,(int,float)) and not (np.isnan(temp_prod) or np.isinf(temp_prod)) else 0.0
            except Exception: opp_reach_prod = 0.0

        player_reach_prob = float(safe_reach[perspective_player_idx]) if perspective_player_idx < len(safe_reach) and isinstance(safe_reach[perspective_player_idx],(int,float)) and not(np.isnan(safe_reach[perspective_player_idx])or np.isinf(safe_reach[perspective_player_idx])) else 0.0
        node_util_val = float(node_utility_perspective) if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)) else 0.0

        # Only update if path reachable by opponents
        if opp_reach_prod > 1e-12:
            # Update Regrets
            current_info_set_regrets = local_regret_sum.setdefault(info_set_key, collections.defaultdict(float))
            for action in available_actions:
                utility_a = action_utilities_perspective.get(action)
                if utility_a is None or not isinstance(utility_a, (int, float)) or np.isnan(utility_a) or np.isinf(utility_a): continue

                instant_regret = utility_a - node_util_val
                if np.isnan(instant_regret) or np.isinf(instant_regret): continue

                current_regret = float(current_info_set_regrets.get(action, 0.0))
                if np.isnan(current_regret) or np.isinf(current_regret): current_regret = 0.0

                regret_inc = opp_reach_prod * instant_regret
                updated_regret = current_regret
                if not (np.isnan(regret_inc) or np.isinf(regret_inc)): updated_regret += regret_inc

                current_info_set_regrets[action] = max(0.0, updated_regret) # Floor at 0

            # Update Strategy Sum
            current_info_set_strategy_sum = local_strategy_sum.setdefault(info_set_key, collections.defaultdict(float))
            strategy_sum_weight = player_reach_prob * current_iteration_t # Linear CFR weight
            if not (np.isnan(strategy_sum_weight) or np.isinf(strategy_sum_weight)):
                for action in available_actions:
                    action_prob = strategy.get(action, 0.0)
                    increment = strategy_sum_weight * action_prob
                    current_sum = float(current_info_set_strategy_sum.get(action, 0.0))
                    if np.isnan(current_sum) or np.isinf(current_sum): current_sum = 0.0

                    if not (np.isnan(increment) or np.isinf(increment)):
                         current_info_set_strategy_sum[action] = current_sum + increment

    # --- Return Node EV ---
    final_utility = float(node_utility_perspective) if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)) else 0.0
    return final_utility


# --- Trainer Class ---
class OptimizedSelfPlayTrainer:
    """ Optimized self-play training using multiprocessing. """

    def __init__(self, game_state_class, num_players=6, num_workers=4):
        """ Initialize the trainer. game_state_class should be a picklable factory. """
        if not callable(game_state_class):
            raise TypeError("game_state_class must be callable (factory or class)")
        self.game_state_factory = game_state_class # Store the factory instance/partial
        self.num_players = num_players
        try:
            self.num_workers = min(max(1, num_workers), mp.cpu_count())
        except NotImplementedError:
            self.num_workers = max(1, num_workers)
            print(f"WARN: mp.cpu_count() failed. Using num_workers={self.num_workers}")
        # Master sums
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration = 0 # Tracks master iterations completed

    def train(self, iterations=1000, checkpoint_freq=100, output_dir="models",
              batch_size_per_worker=10, verbose=False): # Added verbose flag
        """ Train using optimized parallel self-play. """
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"ERROR creating output directory '{output_dir}': {e}")
            return None

        start_iter = self.iteration
        total_hands_simulated = 0
        total_failed_setups = 0
        start_time = time.time()

        print(f"Starting Optimized Training: {iterations} target master iters, {self.num_workers} workers, BatchSize={batch_size_per_worker}...")
        print(f" Output Dir: {os.path.abspath(output_dir)}")

        # Setup progress bar
        pbar = tqdm(range(iterations), desc="Opt CFR", initial=0, total=iterations, disable=(iterations <= 100 and not verbose))

        # Main training loop
        for i in pbar:
            current_master_iteration = start_iter + i + 1

            # --- Prepare worker arguments ---
            # Pass the stored factory instance directly
            worker_args = [
                (self.game_state_factory, self.num_players, batch_size_per_worker,
                 current_master_iteration, None, None, worker_id)
                for worker_id in range(self.num_workers)
            ]
            # --- End Worker Args Prep ---

            results = []
            pool = None
            # --- Execute workers in pool ---
            try:
                # Use spawn context for better compatibility on macOS/Windows
                start_method = 'fork' if sys.platform == 'linux' else 'spawn'
                ctx = mp.get_context(start_method)
                pool = ctx.Pool(processes=self.num_workers)
                results = pool.map(worker_train_batch, worker_args)
            except Exception as pool_err:
                print(f"\nFATAL Multiprocessing Pool Error: {pool_err}")
                traceback.print_exc()
                # Attempt to save checkpoint before breaking
                if output_dir:
                    self._save_checkpoint(output_dir, self.iteration)
                break # Stop training loop on pool error
            finally:
                # Ensure pool resources are released
                if pool:
                    pool.close()
                    pool.join()
            # --- End Pool Execution ---

            # Exit if pool failed and returned no results (unless it's the very first iteration)
            if not results and i > 0:
                print(f"WARN: No results received from worker pool @ iter {current_master_iteration}, stopping.")
                break

            # --- Merge results ---
            hands_this_iter = 0
            fails_this_iter = 0
            for worker_result in results:
                if isinstance(worker_result, tuple) and len(worker_result) == 3:
                    batch_reg, batch_strat, w_fails = worker_result
                    self._merge_results(batch_reg, batch_strat)
                    hands_this_iter += batch_size_per_worker # Count attempted hands
                    fails_this_iter += w_fails
                else:
                    print(f"WARN: Invalid result format received from worker: {worker_result}")
                    fails_this_iter += batch_size_per_worker # Assume all failed

            # Update totals and master iteration count
            total_hands_simulated += hands_this_iter
            total_failed_setups += fails_this_iter
            self.iteration = current_master_iteration # Update master count *after* processing

            # --- Checkpoint and Logging ---
            if self.iteration % checkpoint_freq == 0 or i == iterations - 1: # Checkpoint on freq or last iteration
                elapsed_t = time.time() - start_time
                print(f"\n Checkpoint @ Iter {self.iteration:,}: "
                      f"Elapsed={elapsed_t:.1f}s, InfoSets={len(self.regret_sum):,}, "
                      f"Hands~={total_hands_simulated:,} ({total_failed_setups:,} fails)")
                self._save_checkpoint(output_dir, self.iteration)

        # --- End of Training Loop ---
        pbar.close() # Ensure progress bar is closed
        elapsed_time = time.time() - start_time
        print(f"\nTraining Finished. Final Master Iter: {self.iteration:,}, Total Time: {elapsed_time:.2f}s")
        print(f" Total Hands Simulated (Approx): {total_hands_simulated:,}, Failed setups: {total_failed_setups:,}")

        # Compute and save the final strategy
        final_strategy = self._compute_final_strategy()
        if output_dir: # Save only if output dir is specified
            self._save_final_strategy(output_dir, final_strategy)

        return final_strategy

    def _merge_results(self, batch_regret_sum, batch_strategy_sum):
        """ Merge worker batch results into the main trainer's master sums safely. """
        # Merge regret sums
        for key, regrets in batch_regret_sum.items():
            if not regrets: continue
            master_regrets = self.regret_sum.setdefault(key, collections.defaultdict(float))
            for action, regret in regrets.items():
                if isinstance(regret, (int, float)) and not (np.isnan(regret) or np.isinf(regret)):
                     master_regrets[action] += regret

        # Merge strategy sums
        for key, strategies in batch_strategy_sum.items():
            if not strategies: continue
            master_strategies = self.strategy_sum.setdefault(key, collections.defaultdict(float))
            for action, strategy_sum_inc in strategies.items():
                if isinstance(strategy_sum_inc, (int, float)) and not (np.isnan(strategy_sum_inc) or np.isinf(strategy_sum_inc)):
                     master_strategies[action] += strategy_sum_inc

    def _compute_final_strategy(self):
        """ Computes the final average strategy from accumulated strategy sums. """
        avg_strategy = {}
        num_sets = len(self.strategy_sum)
        if num_sets == 0:
            print("WARN: Cannot compute final strategy, strategy_sum is empty.")
            return {}

        print(f"Computing final average strategy from {num_sets:,} info sets...")
        items_iterable = tqdm(self.strategy_sum.items(), total=num_sets, desc="AvgStrat Calc", disable=(num_sets < 10000))

        for key, action_sums in items_iterable:
            current_set_strategy = {}
            if not isinstance(action_sums, dict): continue # Skip non-dict entries

            valid_vals = [v for v in action_sums.values() if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))]
            norm_sum = sum(valid_vals)
            num_actions = len(action_sums)

            if norm_sum > 1e-9 and num_actions > 0:
                for action, s_sum in action_sums.items():
                    if isinstance(s_sum, (int, float)) and not (np.isnan(s_sum) or np.isinf(s_sum)):
                        current_set_strategy[action] = float(s_sum) / norm_sum
                    else:
                        current_set_strategy[action] = 0.0
                # Re-normalize if necessary due to float precision or initial invalid values
                re_norm_sum = sum(current_set_strategy.values())
                if abs(re_norm_sum - 1.0) > 1e-6 and re_norm_sum > 1e-9:
                    for action in current_set_strategy: current_set_strategy[action] /= re_norm_sum
            elif num_actions > 0: # Default uniform if sum invalid or zero
                prob = 1.0 / num_actions
                current_set_strategy = {action: prob for action in action_sums}
            # Else: num_actions is 0, strategy remains {}

            avg_strategy[key] = current_set_strategy
        return avg_strategy

    def _save_checkpoint(self, output_dir, iteration):
        """ Save a checkpoint of the current training state (master sums). """
        checkpoint_data = {
            "iteration": iteration,
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "num_players": self.num_players
        }
        chk_path = os.path.join(output_dir, f"optimized_checkpoint_{iteration}.pkl")
        try:
            # Ensure directory exists (though train() already does)
            os.makedirs(os.path.dirname(chk_path), exist_ok=True)
            with open(chk_path, "wb") as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as e:
             print(f"\nERROR creating directory for checkpoint {chk_path}: {e}")
        except Exception as e:
            print(f"\nERROR saving optimized checkpoint to {chk_path}: {e}")

    def _save_final_strategy(self, output_dir, strategy_map):
         """ Saves the computed final strategy map. """
         if not strategy_map:
             print("WARN: No final strategy map provided to save.")
             return
         final_save_path = os.path.join(output_dir, "final_strategy_optimized.pkl")
         try:
             with open(final_save_path, "wb") as f:
                 pickle.dump(strategy_map, f, protocol=pickle.HIGHEST_PROTOCOL)
             print(f"Final Optimized Strategy saved: {final_save_path} ({len(strategy_map):,} info sets)")
         except Exception as e:
             print(f"ERROR saving final optimized strategy to {final_save_path}: {e}")

    def load_checkpoint(self, checkpoint_path):
        """ Load state from a checkpoint to resume training. """
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Optimized checkpoint not found: {checkpoint_path}")
            return False
        try:
            print(f"Loading Optimized Checkpoint: {checkpoint_path}...")
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)

            # Load iteration count
            self.iteration = data.get('iteration', 0)

            # Load sums with validation
            loaded_regret = data.get('regret_sum', {})
            loaded_strat = data.get('strategy_sum', {})
            if isinstance(loaded_regret, dict) and isinstance(loaded_strat, dict):
                self.regret_sum = loaded_regret
                self.strategy_sum = loaded_strat
            else:
                print("ERROR: Invalid sum types (not dict) found in checkpoint.")
                # Should we reset sums or keep potentially partially loaded ones? Resetting is safer.
                self.regret_sum = {}
                self.strategy_sum = {}
                return False

            # Load num_players and check for mismatch
            loaded_num_players = data.get('num_players', self.num_players)
            if loaded_num_players != self.num_players:
                print(f"WARN: Checkpoint num_players ({loaded_num_players}) differs from current config ({self.num_players}). Using checkpoint value.")
                self.num_players = loaded_num_players

            print(f"Opt Checkpoint loaded. Resuming training from iteration {self.iteration + 1}.")
            return True
        except Exception as e:
            print(f"ERROR loading optimized checkpoint: {e}")
            traceback.print_exc()
            # Reset state on load failure?
            self.iteration = 0
            self.regret_sum = {}
            self.strategy_sum = {}
            return False

# --- END OF FILE ---
