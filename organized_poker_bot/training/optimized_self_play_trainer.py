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
    """ Worker function to run CFR iterations. (V9 Corrected: Add start_new_hand) """
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

        # --- Setup Hand using factory instance AND START HAND ---
        try:
            # --- Create GameState Object ---
            game_state = game_state_factory_instance(num_players)

            # --- Retrieve Initial Stacks ---
            default_stack = 10000.0 # Fallback, should match factory config
            if hasattr(game_state, 'player_stacks') and game_state.player_stacks:
                 initial_stacks_hand = game_state.player_stacks[:] # Use copy
            else:
                 initial_stacks_hand = [default_stack] * num_players
            # Ensure list has correct length and isn't empty
            if not initial_stacks_hand or len(initial_stacks_hand) != num_players:
                 initial_stacks_hand = [default_stack] * num_players

            # --- **** START THE HAND **** ---
            # Determine dealer position for this specific hand simulation
            # Random dealer selection is suitable for training simulations
            dealer_pos = random.randrange(num_players)
            # Actually start the hand! This deals cards, posts blinds, and sets the first player.
            game_state.start_new_hand(dealer_pos, initial_stacks_hand[:]) # Pass copy of stacks
            # --- **** END START THE HAND **** ---

        except TypeError as te:
            # Catch if factory was called incorrectly (e.g., missing num_players arg)
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: TypeError Calling Factory: {te}")
            worker_failed_hands += 1
            continue
        except Exception as e:
            # Catch errors during GameState creation OR start_new_hand
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Error Creating/Starting State: {type(e).__name__}: {e}")
            # import traceback # Uncomment for full traceback during debugging
            # traceback.print_exc()
            worker_failed_hands += 1
            continue

        # --- Check State AFTER Hand Start ---
        if game_state is None:
            # This check might be redundant now with the try/except above, but keep for safety
            print(f"!!! FAIL Worker {worker_id} Hand {hand_idx+1}/{batch_size}: GameState is None after setup.")
            worker_failed_hands += 1
            continue
        if game_state.is_terminal():
            # Hand might legitimately end during start_new_hand (e.g., insufficient players for blinds)
            # print(f"INFO Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Immediately terminal state after start_new_hand.") # Reduce noise - optional info log
            continue # Skip traversal for this terminal hand, not a failure
        if game_state.current_player_idx == -1 and not game_state.is_terminal():
            # This check is now more critical - indicates an internal failure in start_new_hand if true
             print(f"!!! ERROR Worker {worker_id} Hand {hand_idx+1}/{batch_size}: Invalid current_player_idx (-1) AFTER start_new_hand but not terminal. State likely corrupt.")
             worker_failed_hands += 1 # Count as failure
             continue


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
                    initial_stacks=initial_stacks_hand[:], # Pass the actual initial stacks for this hand
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
                # Optionally print traceback for deeper debug
                # import traceback
                # traceback.print_exc()
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
    """ Recursive CFR logic for worker (matches CFRTrainer closely). (V10 ADD DEBUG LOG)"""
    # Use recursion limit from CFRTrainer class if available, else default
    WORKER_REC_LIMIT = getattr(CFRTrainer, 'RECURSION_DEPTH_LIMIT', 500) - 50 # Use buffer

    # <<< START ADDED DEBUG LOG >>>
    try:
        # Safely format history and board
        hist_str = ";".join(game_state.action_sequence[-5:]) if hasattr(game_state, 'action_sequence') and game_state.action_sequence else ""
        board_str = "".join(map(str, game_state.community_cards)) if hasattr(game_state, 'community_cards') and game_state.community_cards else "-"
        turn_str = str(game_state.current_player_idx) if hasattr(game_state, 'current_player_idx') else "ERR"
        rnd_str = str(game_state.betting_round) if hasattr(game_state, 'betting_round') else "ERR"
        print(f"DEBUG _traverse ENTRY: Depth={current_depth} Persp={perspective_player_idx} Turn={turn_str} Rnd={rnd_str} Board=[{board_str}] Hist=[{hist_str}]")
    except Exception as log_e:
        print(f"DEBUG _traverse ENTRY Log Err: {log_e}")
    # <<< END ADDED DEBUG LOG >>>

    # --- Base Cases ---
    if game_state.is_terminal():
        utility = 0.0
        try:
            utility_val = game_state.get_utility(perspective_player_idx, initial_stacks)
            # Ensure utility is a finite float
            utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        except Exception: pass # Default 0.0 on error
        return utility

    if current_depth > WORKER_REC_LIMIT:
        # <<< ADDED DEBUG LOG BEFORE ERROR >>>
        # Try to get acting player if possible for the log message
        acting_player_debug = getattr(game_state, 'current_player_idx', 'N/A')
        print(f"!!! DEBUG: Hitting Worker Recursion Limit at depth {current_depth} for P{acting_player_debug}")
        # <<< END ADDED DEBUG LOG BEFORE ERROR >>>
        raise RecursionError(f"Worker depth limit {WORKER_REC_LIMIT} exceeded")

    # --- Handle Inactive Player ---
    acting_player_idx = getattr(game_state, 'current_player_idx', -1) # Default to -1 if attribute missing

    if not (0 <= acting_player_idx < num_players):
        print(f"DEBUG _traverse RETURN: Invalid acting_player_idx {acting_player_idx}") # Log invalid index exit
        return 0.0 # Invalid index

    # Safely check player state using getattr with defaults
    is_folded = getattr(game_state, 'player_folded', [True]*num_players)[acting_player_idx]
    is_all_in = getattr(game_state, 'player_all_in', [True]*num_players)[acting_player_idx]

    if is_folded or is_all_in:
        # Create a clone to safely advance the turn
        try:
            temp_state = game_state.clone()
            original_turn_idx = temp_state.current_player_idx
            temp_state._move_to_next_player()
            # Check if moving the turn didn't work or resulted in a terminal state
            if temp_state.current_player_idx == original_turn_idx or temp_state.is_terminal():
                utility = 0.0
                try:
                    utility_val = temp_state.get_utility(perspective_player_idx, initial_stacks)
                    utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                except Exception: pass
                return utility
            else:
                # <<< ADDED DEBUG LOG (Optional) >>>
                # print(f"  Skipping P{acting_player_idx} (Inactive) -> Recursing (Depth {current_depth+1})")
                # <<< END ADDED DEBUG LOG >>>
                # Recursive call, increment depth
                return _worker_cfr_traverse(temp_state, reach_probs, perspective_player_idx, initial_stacks, current_iteration_t, local_regret_sum, local_strategy_sum, num_players, current_depth + 1)
        except Exception as skip_e:
            print(f"DEBUG _traverse RETURN: Error cloning/skipping inactive P{acting_player_idx}: {skip_e}")
            return 0.0 # Return neutral value if skipping fails

    # --- Active Player: Key, Actions, Strategy ---
    try: # Get InfoSet Key
        info_set_key = generate_info_set_key(game_state, acting_player_idx)
        if not info_set_key or not isinstance(info_set_key, str): raise ValueError("Invalid key generated")
    except Exception as key_e:
        print(f"DEBUG _traverse RETURN: KeyGen Error P{acting_player_idx} Depth {current_depth}: {key_e}")
        return 0.0

    try: # Get Actions (Handle Abstraction)
        raw_actions = game_state.get_available_actions()
        # *** IMPORTANT: Configure abstraction consistently ***
        # This should ideally be passed via args or read from a shared config
        USE_ACTION_ABSTRACTION_IN_WORKER = True # Assume default
        if USE_ACTION_ABSTRACTION_IN_WORKER:
            available_actions = ActionAbstraction.abstract_actions(raw_actions, game_state)
        else:
            available_actions = raw_actions
        # Validate result
        if not isinstance(available_actions, list): available_actions = []

    except Exception as action_e:
        print(f"DEBUG _traverse RETURN: GetActions Error P{acting_player_idx} Depth {current_depth}: {action_e}")
        return 0.0 # Fail on action error

    # Handle no available actions (game should be terminal, but check explicitly)
    if not available_actions:
        utility = 0.0
        try:
            utility_val = game_state.get_utility(perspective_player_idx, initial_stacks)
            utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        except Exception: pass
        return utility

    # --- Get Strategy from LOCAL sums ---
    # (Make sure 'collections' is imported at the top of the file)
    regrets = local_regret_sum.get(info_set_key, {}) # Use .get for safety
    strategy = {}
    positive_regret_sum = 0.0
    action_positive_regrets = {}
    for action in available_actions:
        regret = regrets.get(action, 0.0)
        # Ensure regret is a valid number
        regret = regret if isinstance(regret, (int, float)) and not (np.isnan(regret) or np.isinf(regret)) else 0.0
        positive_regret = max(0.0, regret)
        action_positive_regrets[action] = positive_regret
        positive_regret_sum += positive_regret
    # Normalize or use uniform
    if positive_regret_sum > 1e-9: # Use tolerance for float comparison
        try:
            strategy = {action: action_positive_regrets[action] / positive_regret_sum for action in available_actions}
        except ZeroDivisionError: # Should not happen with the > 1e-9 check, but safeguard
             strategy = {action: 1.0 / len(available_actions) for action in available_actions} if available_actions else {}
    else:
        num_act = len(available_actions)
        prob = 1.0 / num_act if num_act > 0 else 0.0
        strategy = {action: prob for action in available_actions}

    # --- Explore Actions ---
    node_utility_perspective = 0.0
    action_utilities_perspective = {} # Store utility received AFTER taking action 'a'
    for action in available_actions:
        action_prob = strategy.get(action, 0.0)
        # Skip actions with negligible probability
        if action_prob < 1e-9:
            action_utilities_perspective[action] = None # Mark as not explored
            continue

        # Apply Action (safely)
        try:
            next_game_state = game_state.apply_action(action)
        except Exception as apply_e:
            # If applying action fails, skip this path
            print(f"DEBUG _traverse SKIP ACTION: ApplyAction Error P{acting_player_idx} Depth {current_depth} Action {action}: {apply_e}")
            action_utilities_perspective[action] = None # Mark as failed
            continue

        # Update Opponent Reach Probability
        next_reach_probs = reach_probs.copy()
        if acting_player_idx != perspective_player_idx: # Update opponent reach
            prob_factor = float(action_prob) if isinstance(action_prob,(int,float)) and not(np.isnan(action_prob)or np.isinf(action_prob)) else 0.0
            current_reach = float(next_reach_probs[acting_player_idx]) if acting_player_idx < len(next_reach_probs) and isinstance(next_reach_probs[acting_player_idx],(int,float)) and not(np.isnan(next_reach_probs[acting_player_idx])or np.isinf(next_reach_probs[acting_player_idx])) else 0.0
            updated_reach = np.clip(current_reach * prob_factor, 0.0, 1.0) # Clip to handle float issues
            next_reach_probs[acting_player_idx] = updated_reach

        # --- Recursive Call ---
        try:
            # <<< START ADDED DEBUG LOGGING for recursion >>>
            # Safely format relevant info from the *next* state before calling
            next_hist_str = ";".join(next_game_state.action_sequence[-5:]) if hasattr(next_game_state, 'action_sequence') and next_game_state.action_sequence else ""
            next_board_str = "".join(map(str, next_game_state.community_cards)) if hasattr(next_game_state, 'community_cards') and next_game_state.community_cards else "-"
            next_turn_str = str(next_game_state.current_player_idx) if hasattr(next_game_state, 'current_player_idx') else "ERR"
            next_rnd_str = str(next_game_state.betting_round) if hasattr(next_game_state, 'betting_round') else "ERR"
            state_info_for_log = f"P{acting_player_idx}-Act={action} -> Depth={current_depth+1} Persp={perspective_player_idx} Turn={next_turn_str} Rnd={next_rnd_str} Board=[{next_board_str}] Hist=[{next_hist_str}]"
            print(f"  Recursing: {state_info_for_log}") # Print info BEFORE the call
            # <<< END ADDED DEBUG LOGGING for recursion >>>

            utility_from_action = _worker_cfr_traverse(
                next_game_state, next_reach_probs, perspective_player_idx,
                initial_stacks, current_iteration_t, local_regret_sum,
                local_strategy_sum, num_players, current_depth + 1 # Increment depth
            )
            # Validate utility from recursion
            utility_from_action = float(utility_from_action) if isinstance(utility_from_action, (int, float)) and not (np.isnan(utility_from_action) or np.isinf(utility_from_action)) else 0.0
            action_utilities_perspective[action] = utility_from_action

            # Accumulate node utility weighted by action probability
            node_utility_perspective += action_prob * utility_from_action

        except RecursionError as re_inner:
            # Propagate recursion error up immediately
            raise re_inner
        except Exception as rec_e:
            print(f"DEBUG _traverse Error during recursive call for action {action}: {rec_e}")
            action_utilities_perspective[action] = None # Mark recursive call as failed

        # --- End Recursive Call Section ---


    # --- Update LOCAL Sums (If Perspective Player Acting) ---
    if acting_player_idx == perspective_player_idx:
        # Calculate opponent reach product safely
        safe_reach = np.nan_to_num(reach_probs, nan=0.0, posinf=0.0, neginf=0.0)
        opp_reach_prod = 1.0
        if num_players > 1:
            opp_reaches = [safe_reach[p] for p in range(num_players) if p != perspective_player_idx]
            try:
                # Ensure values are valid before prod
                valid_opp_reaches = [r for r in opp_reaches if isinstance(r, (int, float)) and not (np.isnan(r) or np.isinf(r))]
                temp_prod = np.prod(valid_opp_reaches) if valid_opp_reaches else 1.0
                opp_reach_prod = float(temp_prod) if isinstance(temp_prod,(int,float)) and not (np.isnan(temp_prod) or np.isinf(temp_prod)) else 0.0
            except Exception: opp_reach_prod = 0.0 # Default to 0 on error

        # Get player reach probability safely
        player_reach_prob = float(safe_reach[perspective_player_idx]) if perspective_player_idx < len(safe_reach) and isinstance(safe_reach[perspective_player_idx],(int,float)) and not(np.isnan(safe_reach[perspective_player_idx])or np.isinf(safe_reach[perspective_player_idx])) else 0.0
        # Validate node utility
        node_util_val = float(node_utility_perspective) if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)) else 0.0

        # Only update if path reachable by opponents (avoid unnecessary computation/potential NaNs)
        if opp_reach_prod > 1e-12: # Use tolerance
            # Ensure local sums are initialized correctly (should use setdefault)
            current_info_set_regrets = local_regret_sum.setdefault(info_set_key, collections.defaultdict(float))
            current_info_set_strategy_sum = local_strategy_sum.setdefault(info_set_key, collections.defaultdict(float))

            # Update Regrets
            for action in available_actions:
                utility_a = action_utilities_perspective.get(action)
                # Skip if action wasn't explored or recursive call failed
                if utility_a is None or not isinstance(utility_a, (int, float)) or np.isnan(utility_a) or np.isinf(utility_a):
                    continue

                instant_regret = utility_a - node_util_val
                # Skip if calculation results in invalid number
                if np.isnan(instant_regret) or np.isinf(instant_regret):
                    continue

                # Get current regret, ensure it's a valid float
                current_regret = float(current_info_set_regrets.get(action, 0.0))
                if np.isnan(current_regret) or np.isinf(current_regret): current_regret = 0.0

                # Calculate and apply increment safely
                regret_inc = opp_reach_prod * instant_regret
                updated_regret = current_regret
                if not (np.isnan(regret_inc) or np.isinf(regret_inc)):
                    updated_regret += regret_inc

                # Update regret, flooring at 0
                current_info_set_regrets[action] = max(0.0, updated_regret)

            # Update Strategy Sum (Linear CFR weight: player_reach * iteration_number)
            strategy_sum_weight = player_reach_prob * float(current_iteration_t) # Make sure weight is float
            # Skip update if weight is invalid
            if not (np.isnan(strategy_sum_weight) or np.isinf(strategy_sum_weight)):
                for action in available_actions:
                    action_prob = strategy.get(action, 0.0)
                    increment = strategy_sum_weight * action_prob
                    # Get current sum safely
                    current_sum = float(current_info_set_strategy_sum.get(action, 0.0))
                    if np.isnan(current_sum) or np.isinf(current_sum): current_sum = 0.0

                    # Apply increment safely
                    if not (np.isnan(increment) or np.isinf(increment)):
                         current_info_set_strategy_sum[action] = current_sum + increment

    # --- Return Node EV for the perspective player ---
    final_utility = float(node_utility_perspective) if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)) else 0.0
    #print(f"DEBUG _traverse RETURN: Depth={current_depth} Persp={perspective_player_idx} Util={final_utility}") # Log final utility
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
