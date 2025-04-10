# --- START OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
"""
Implementation of Counterfactual Regret Minimization (CFR) for poker.
Utilizes External Sampling style updates and Linear CFR weighting.
(Refactored to accept optional custom action function and with enhanced logging)
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import sys
import traceback
import time # For timing

# Path setup (keep if necessary for your structure, might be handled by __init__.py)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports (ensure these are correct for your structure)
try:
    # Using relative imports assuming standard package structure
    from .information_set import InformationSet
    # Assuming card/action abstractions are in the same cfr directory
    from .card_abstraction import CardAbstraction
    from .action_abstraction import ActionAbstraction
    # Assuming game_engine is one level up
    from ..game_engine.game_state import GameState
except ImportError as e:
    print(f"FATAL Import Error in cfr_trainer.py: {e}")
    # Attempt absolute import as fallback if relative fails
    try:
        from organized_poker_bot.cfr.information_set import InformationSet
        from organized_poker_bot.cfr.card_abstraction import CardAbstraction
        from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
        from organized_poker_bot.game_engine.game_state import GameState
    except ImportError:
         print("Could not resolve imports. Check package structure and PYTHONPATH.")
         sys.exit(1)


# Recursion limit setup
try:
    current_limit = sys.getrecursionlimit()
    target_limit = 3000 # Or a suitable value based on expected tree depth
    if current_limit < target_limit:
        print(f"INFO: Increasing Python recursion limit from {current_limit} to {target_limit}")
        sys.setrecursionlimit(target_limit)
        current_limit = sys.getrecursionlimit() # Get the potentially updated limit
        print(f"INFO: Python recursion limit set to {current_limit}")
    else:
        print(f"INFO: Current Python recursion limit ({current_limit}) is sufficient.")

    # Set a practical limit for CFR recursion slightly below the system limit
    CFRTrainer_REC_LIMIT = max(100, current_limit - 100) # Ensure at least 100, take margin

except Exception as e:
    print(f"WARN: Failed to adjust recursion limit: {e}. Using default: 1000")
    CFRTrainer_REC_LIMIT = 1000 # Fallback default


class CFRTrainer:
    RECURSION_DEPTH_LIMIT = CFRTrainer_REC_LIMIT

    # --- MODIFIED INIT ---
    def __init__(self, game_state_class, num_players=2,
                 use_action_abstraction=True, use_card_abstraction=True,
                 # Add optional custom action function
                 custom_get_actions_func=None):
        if not callable(game_state_class): raise TypeError("GS class !callable")
        self.game_state_class = game_state_class
        self.num_players = num_players
        self.information_sets = {}
        self.iterations = 0
        self.use_action_abstraction = use_action_abstraction
        self.use_card_abstraction = use_card_abstraction
        self.training_start_time = None
        # Store the custom function if provided
        self.get_actions_override = custom_get_actions_func
        if custom_get_actions_func:
             print("INFO: CFRTrainer initialized with custom_get_actions_func.")
             # When using an override, usually disable standard action abstraction
             # as the override defines the exact desired actions.
             if self.use_action_abstraction:
                  print("WARN: custom_get_actions_func provided, but use_action_abstraction is still True. Disabling standard abstraction.")
                  self.use_action_abstraction = False
    # --- END MODIFIED INIT ---


    def train(self, iterations=1000, checkpoint_freq=100, output_dir=None, verbose=False):
        """ Trains the CFR model using External Sampling and Linear Weighting. """
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        if self.training_start_time is None: # Initialize start time only once
             self.training_start_time = time.time()

        start_iter = self.iterations # Resume from last completed iteration
        end_iter = self.iterations + iterations
        num_iterations_this_run = iterations

        print(f"Starting Linear CFR Training: Iterations {start_iter + 1} to {end_iter}...")
        print(f"Using Recursion Limit: {self.RECURSION_DEPTH_LIMIT}")
        if output_dir: print(f"Checkpoints saved to: {os.path.abspath(output_dir)}")

        # Configure tqdm: Disable only if 0 iterations or maybe very few (< 10?)
        pbar_disable = (num_iterations_this_run <= 10 and not verbose)
        pbar = tqdm(range(start_iter, end_iter), desc="CFR Training", initial=start_iter, total=end_iter, disable=pbar_disable, unit="iter")

        total_train_time_sec_this_run = 0.0
        log_print_frequency = max(1, min(50, num_iterations_this_run // 20)) # Print roughly 20 times

        for i in pbar: # Main Training Loop
            iter_start_time = time.time()
            current_iter_num = i + 1 # Current iteration number (1-based)

            if not pbar_disable: pbar.set_description(f"CFR Iter {current_iter_num}")

            if verbose: print(f"\n===== Iteration {current_iter_num} =====")

            game_state = None
            # Get initial stacks from the factory, assume it handles default
            initial_stacks_hand = []
            try:
                temp_gs = self.game_state_class(self.num_players)
                initial_stacks_hand = temp_gs.player_stacks[:] # Get stacks
            except Exception as e:
                print(f"WARN: Could not get initial stacks from game_state_class. Using default. Err: {e}")
                initial_stacks_hand = [10000.0] * self.num_players

            try: # Hand Setup
                game_state = self.game_state_class(self.num_players) # Assume init handles blinds/stacks if needed
                # Ensure we pass a copy of stacks for start_new_hand
                stacks_for_hand = initial_stacks_hand[:]
                dealer_pos = current_iter_num % self.num_players # Simple dealer rotation
                game_state.start_new_hand(dealer_pos=dealer_pos, player_stacks=stacks_for_hand)

                if game_state.is_terminal() or game_state.current_player_idx == -1:
                     if verbose: print(f"INFO Iter {current_iter_num}: Hand ended prematurely, skipping CFR calc.")
                     continue

            except Exception as e:
                print(f"\nERROR starting hand for Iter {current_iter_num}: {e}")
                traceback.print_exc()
                continue # Skip iteration on setup error


            reach_probs = np.ones(self.num_players, dtype=float)
            iter_utilities_perspectives = []
            failed_perspectives_count = 0

            # Loop through each player's perspective for External Sampling
            for p_idx in range(self.num_players):
                perspective_utility = 0.0
                try:
                    perspective_utility = self._calculate_cfr(
                         game_state.clone(), # Pass a fresh clone for each perspective
                         reach_probs.copy(), # Pass reach probability array
                         p_idx, # The player whose perspective we are evaluating from
                         initial_stacks_hand[:], # Pass copy of original initial stacks
                         float(current_iter_num), # Weight T (Linear CFR)
                         0.0, # Pruning threshold (0 = off)
                         0, # Initial depth
                         verbose # Verbose flag for _calculate_cfr internal prints
                    )
                    iter_utilities_perspectives.append(perspective_utility)
                except RecursionError as re:
                     print(f"\nFATAL: Recursion limit reached in perspective P{p_idx} at Iter {current_iter_num}.")
                     pbar.close()
                     if output_dir: self._save_final_strategy(output_dir, self.get_strategy())
                     return self.get_strategy() # Return whatever strategy exists
                except Exception as e:
                    print(f"\nERROR in CFR calculation for P{p_idx}, Iter {current_iter_num}: {e}")
                    traceback.print_exc()
                    iter_utilities_perspectives.append(None)
                    failed_perspectives_count += 1


            # Update counters and save checkpoint if needed
            iter_duration_sec = time.time() - iter_start_time
            total_train_time_sec_this_run += iter_duration_sec

            if failed_perspectives_count < self.num_players:
                self.iterations = current_iter_num

                if output_dir and (self.iterations % checkpoint_freq == 0):
                    if log_print_frequency > checkpoint_freq or self.iterations % log_print_frequency != 0:
                         print(f"\nCheckpointing at iteration {self.iterations}...")
                    self._save_checkpoint(output_dir, self.iterations)

                valid_utils = [u for u in iter_utilities_perspectives if u is not None]
                avg_util_iter = f"{np.mean(valid_utils):.3f}" if valid_utils else "N/A"
                if not pbar_disable:
                    pbar.set_postfix({"Sets": len(self.information_sets), "AvgUtil": avg_util_iter, "LastT": f"{iter_duration_sec:.2f}s"}, refresh=True)

                if self.iterations % log_print_frequency == 0 or self.iterations == end_iter:
                     time_elapsed = time.time() - self.training_start_time
                     avg_iter_time = time_elapsed / self.iterations if self.iterations > 0 else 0
                     print(f"  Iter {self.iterations}/{end_iter} | InfoSets: {len(self.information_sets):,} | AvgIterUtil: {avg_util_iter} | AvgIterTime: {avg_iter_time:.3f}s")
                     if output_dir and (self.iterations % checkpoint_freq == 0):
                           print(f"      -> Checkpoint saved.")

            else:
                print(f"WARN: Skipping iteration {current_iter_num} update - all perspectives failed.")


        pbar.close()
        avg_time_per_iter_run = total_train_time_sec_this_run / num_iterations_this_run if num_iterations_this_run > 0 else 0
        total_elapsed_time = time.time() - self.training_start_time

        print(f"\nTraining loop finished ({num_iterations_this_run} iterations run).")
        print(f"  Avg time per iteration (this run): {avg_time_per_iter_run:.4f} seconds")
        print(f"  Total training time (all runs): {total_elapsed_time:.2f} seconds")

        final_strat = self.get_strategy()
        if output_dir:
            self._save_final_strategy(output_dir, final_strat)

        return final_strat

    # --- MODIFIED _calculate_cfr ---
# --- Replace the existing _calculate_cfr function in organized_poker_bot/cfr/cfr_trainer.py ---

# --- Replace the existing _calculate_cfr function in organized_poker_bot/cfr/cfr_trainer.py ---

# --- Replace the existing _calculate_cfr function in organized_poker_bot/cfr/cfr_trainer.py ---

# --- Replace the existing _calculate_cfr function in organized_poker_bot/cfr/cfr_trainer.py ---
# --- VERSION WITH UNWEIGHTED REGRET UPDATE FOR DEBUGGING P/F CONVERGENCE ---

    def _calculate_cfr(self, game_state, reach_probs, player_idx, initial_stacks, weight, prune_threshold, depth, verbose):
        """ Recursive CFR function (External Sampling Style, Linear CFR weighting FOR STRAT SUM ONLY) """
        # --- Debug prints removed for performance ---
        # MAX_DEBUG_DEPTH = 10
        # if depth <= MAX_DEBUG_DEPTH:
        #      try: history_str = game_state.get_betting_history()
        #      except: history_str = "(HistErr)"
        #      display_hist = history_str if len(history_str) < 60 else history_str[:57] + "..."
        #      print(f"{'  ' * depth}D{depth} Enter CFR | P{player_idx} Persp | Turn: P{game_state.current_player_idx} | Rnd: {game_state.betting_round} | Pot: {game_state.pot:.0f} | Hist: {display_hist}")
        # elif depth == MAX_DEBUG_DEPTH + 1:
        #      print(f"{'  ' * depth}D{depth} ... Suppressing deeper logs ...")

        indent = "  " * depth # Keep for potential future prints

        # Base Case 1: Terminal Node
        if game_state.is_terminal():
            utility = 0.0
            try:
                 utility_val = game_state.get_utility(player_idx, initial_stacks)
                 utility = utility_val if isinstance(utility_val, (int, float)) else 0.0
                 if np.isnan(utility) or np.isinf(utility): utility = 0.0
            except Exception as e: pass # Suppress utility errors silently now
            return utility

        # Base Case 2: Recursion Depth Limit
        if depth > self.RECURSION_DEPTH_LIMIT:
            # Keep this critical print
            print(f"{'  ' * depth}D{depth} CRITICAL: Recursion depth limit ({self.RECURSION_DEPTH_LIMIT}) hit! Returning 0.")
            return 0.0

        # Identify Acting Player
        acting_player_idx = game_state.current_player_idx
        if not (0 <= acting_player_idx < self.num_players): return 0.0

        # Check if acting player can act
        is_folded = game_state.player_folded[acting_player_idx] if acting_player_idx < len(game_state.player_folded) else True
        is_all_in = game_state.player_all_in[acting_player_idx] if acting_player_idx < len(game_state.player_all_in) else True
        is_active_in_hand = acting_player_idx in game_state.active_players

        # Skip turn and recurse if player cannot act
        if is_folded or is_all_in or not is_active_in_hand:
             temp_state = game_state.clone()
             original_turn_idx = temp_state.current_player_idx
             temp_state._move_to_next_player()
             if temp_state.current_player_idx == original_turn_idx or temp_state.is_terminal():
                   utility = 0.0
                   try:
                       utility_val = temp_state.get_utility(player_idx, initial_stacks)
                       utility = utility_val if isinstance(utility_val, (int,float)) else 0.0
                       if np.isnan(utility) or np.isinf(utility): utility = 0.0
                   except: pass
                   return utility
             else:
                  # Recurse
                  return self._calculate_cfr(temp_state, reach_probs, player_idx, initial_stacks, weight, prune_threshold, depth + 1, verbose)

        # --- Active player's turn ---
        info_set_key = self._create_info_set_key(game_state, acting_player_idx)

        # Get Available Actions (using override or standard method)
        available_actions = []
        try:
            if self.get_actions_override and callable(self.get_actions_override):
                 available_actions = self.get_actions_override(game_state)
            else:
                 raw_actions = game_state.get_available_actions()
                 if self.use_action_abstraction:
                     try:
                         abstracted = ActionAbstraction.abstract_actions(raw_actions, game_state)
                         available_actions = abstracted if abstracted and isinstance(abstracted, list) else raw_actions
                     except Exception:
                          available_actions = raw_actions
                 else:
                      available_actions = raw_actions
        except Exception: return 0.0

        if not available_actions:
            try: return game_state.get_utility(player_idx, initial_stacks)
            except: return 0.0

        info_set = self._get_or_create_info_set(info_set_key, available_actions)
        if info_set is None: return 0.0

        strategy = info_set.get_strategy()
        node_utility_perspective = 0.0
        action_utilities_perspective = {}

        # --- Explore Actions Loop ---
        for action in available_actions:
            action_prob = strategy.get(action, 0.0)
            if action_prob <= 0.0:
                 action_utilities_perspective[action] = None; continue

            try:
                 next_game_state = game_state.apply_action(action)
            except Exception:
                 action_utilities_perspective[action] = None; continue

            next_reach_probs = reach_probs.copy()
            if acting_player_idx != player_idx:
                 next_reach_probs[acting_player_idx] *= action_prob

            try:
                 # Recursive call
                 utility_from_action = self._calculate_cfr(
                     next_game_state, next_reach_probs, player_idx, initial_stacks,
                     weight, prune_threshold, depth + 1, verbose
                 )
                 action_utilities_perspective[action] = utility_from_action
                 # Accumulate node value safely
                 if isinstance(utility_from_action, (int, float)) and not (np.isnan(utility_from_action) or np.isinf(utility_from_action)):
                      node_utility_perspective += action_prob * utility_from_action
            except RecursionError as re_inner: raise re_inner
            except Exception:
                 action_utilities_perspective[action] = None

        # --- Update Regrets/Strategy Sums (Only if acting_player == perspective_player) ---
        if acting_player_idx == player_idx:
            # Calculate weights safely
            safe_reach = np.nan_to_num(reach_probs, nan=0.0, posinf=0.0, neginf=0.0)
            opp_reach_prod = np.prod([safe_reach[p] for p in range(self.num_players) if p != player_idx]) if self.num_players > 1 else 1.0
            if np.isinf(opp_reach_prod) or np.isnan(opp_reach_prod): opp_reach_prod = 0.0
            player_reach = safe_reach[player_idx] if player_idx < len(safe_reach) else 0.0
            if np.isinf(player_reach) or np.isnan(player_reach): player_reach = 0.0

            # regret_weight = opp_reach_prod * weight # Original Linear CFR Weight
            strategy_weight = player_reach * weight   # Linear CFR Weight for Strat Sum

            node_util_val = node_utility_perspective if isinstance(node_utility_perspective, (int, float)) else 0.0
            if np.isnan(node_util_val) or np.isinf(node_util_val): node_util_val = 0.0

            is_valid_for_strat_update = not (np.isnan(strategy_weight) or np.isinf(strategy_weight))

            # Perform Regret Updates (Using Unweighted Regret for Testing)
            for action in available_actions:
                 utility_a = action_utilities_perspective.get(action)
                 if utility_a is None or np.isnan(utility_a) or np.isinf(utility_a): continue

                 # --- *** USE UNWEIGHTED INSTANT REGRET *** ---
                 instant_regret = utility_a - node_util_val
                 # ------------------------------------------

                 if not np.isnan(instant_regret) and not np.isinf(instant_regret):
                     current_regret = np.nan_to_num(info_set.regret_sum.get(action, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
                     # Accumulate UNWEIGHTED regret
                     new_regret = current_regret + instant_regret
                     info_set.regret_sum[action] = new_regret


            # Perform Strategy Sum Update (Still use Linear CFR weighting)
            if is_valid_for_strat_update:
                 info_set.update_strategy_sum(strategy, strategy_weight)


        # --- Return Value ---
        final_utility = node_utility_perspective if isinstance(node_utility_perspective, (int, float)) else 0.0
        if np.isnan(final_utility) or np.isinf(final_utility): final_utility = 0.0

        # --- Removed Exit Debug Print ---
        # if depth <= MAX_DEBUG_DEPTH:
        #      util_str = f"{final_utility:.2f}"
        #      print(f"{indent}D{depth} Exit CFR  | P{player_idx} Persp | FINAL RETURN NodeUtil: {util_str}")

        return final_utility



# --- Paste remaining methods (_create_info_set_key, _get_or_create_info_set, get_strategy, _save_*, load_*) here ---
    def _create_info_set_key(self, game_state, player_idx):
        """ Creates info set key using abstractions and betting history. """
        cards_part = "NOCARDS"
        pos_part = "POS_ERR"
        hist_part = "BH_ERR" # Betting History Error default

        # 1. Card Abstraction Part
        try:
            # Safely access hole cards
            hole = []
            if game_state.hole_cards and 0 <= player_idx < len(game_state.hole_cards):
                 hole = game_state.hole_cards[player_idx]

            comm = game_state.community_cards if hasattr(game_state, 'community_cards') else []
            num_comm = len(comm)

            if self.use_card_abstraction and hole and len(hole) == 2:
                if num_comm == 0: # Preflop
                     preflop_bucket = CardAbstraction.get_preflop_abstraction(hole)
                     cards_part = f"PRE{preflop_bucket}"
                else: # Postflop
                    postflop_abs_tuple = CardAbstraction.get_postflop_abstraction(hole, comm)
                    s_buck, b_pair, b_flush = postflop_abs_tuple
                    round_names = {3: "FLOP", 4: "TURN", 5: "RIVER"}
                    round_prefix = round_names.get(num_comm, f"POST{num_comm}")
                    cards_part = f"{round_prefix}B{s_buck}P{b_pair}F{b_flush}"

            elif hole: # Fallback to raw cards
                 hole_str = "_".join(sorted(str(c) for c in hole))
                 comm_str = "_".join(sorted(str(c) for c in comm))
                 cards_part = f"RAW_{hole_str}_{comm_str}"

        except Exception as e:
            cards_part = f"CARDS_ERR_{type(e).__name__}"


        # 2. Position Part
        try:
            position_relative = game_state.get_position(player_idx)
            pos_part = f"POS{position_relative}"
        except Exception as e:
            pos_part = f"POS_ERR_{type(e).__name__}"


        # 3. Betting History Part
        try:
            hist_part = game_state.get_betting_history()
            if not hist_part: hist_part = "start"
        except Exception as e:
             hist_part = f"BH_ERR_{type(e).__name__}"

        return f"{cards_part}|{pos_part}|{hist_part}"


    def _get_or_create_info_set(self, key, actions):
        """ Safely gets or creates an InformationSet object. """
        if key not in self.information_sets:
            valid_actions = []
            seen_action_repr = set()
            if not isinstance(actions, list): actions = []

            for action in actions:
                 action_tuple = None
                 if isinstance(action, tuple) and len(action) == 2:
                     try: action_tuple = (str(action[0]), int(round(float(action[1]))))
                     except: continue
                 elif isinstance(action, str) and action in ['fold', 'check']: action_tuple = (action, 0)
                 else: continue

                 if action_tuple:
                      action_repr = f"{action_tuple[0]}{action_tuple[1]}"
                      if action_repr not in seen_action_repr:
                           valid_actions.append(action_tuple)
                           seen_action_repr.add(action_repr)

            if valid_actions:
                 try: self.information_sets[key] = InformationSet(valid_actions)
                 except Exception as e: print(f"ERROR creating InfoSet for key {key}: {e}"); return None
            else: return None

        return self.information_sets.get(key)


    def get_strategy(self):
        """ Calculates the average strategy across all tracked information sets. """
        average_strategy_map = {}
        num_total_sets = len(self.information_sets)
        num_invalid_sets = 0

        if num_total_sets == 0:
             print("WARN: Cannot get strategy - No information sets found.")
             return {}

        print(f"\nCalculating average strategy from {num_total_sets:,} information sets...")
        use_tqdm_local = num_total_sets > 5000
        items_iterable = tqdm(self.information_sets.items(), desc="AvgStrat Calc", total=num_total_sets, disable=not use_tqdm_local, unit="set")

        for key, info_set_obj in items_iterable:
            if not isinstance(info_set_obj, InformationSet):
                num_invalid_sets += 1; continue

            try:
                avg_strat_for_set = info_set_obj.get_average_strategy()
                if isinstance(avg_strat_for_set, dict) and avg_strat_for_set:
                    prob_sum = sum(avg_strat_for_set.values())
                    if abs(prob_sum - 1.0) < 0.01:
                         valid_keys = all(isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) and isinstance(k[1], int) for k in avg_strat_for_set.keys())
                         if valid_keys: average_strategy_map[key] = avg_strat_for_set
                         else: num_invalid_sets += 1
                    elif abs(prob_sum) < 0.01: average_strategy_map[key] = avg_strat_for_set
                    else: num_invalid_sets += 1
                elif isinstance(avg_strat_for_set, dict) and not avg_strat_for_set: num_invalid_sets += 1
                else: num_invalid_sets += 1; print(f"WARN: Invalid type returned by get_avg_strat for key '{key}'")
            except Exception as e: print(f"ERROR calc avg strat for '{key}': {e}"); num_invalid_sets += 1

        if num_invalid_sets > 0: print(f"WARN: Skipped {num_invalid_sets:,}/{num_total_sets:,} invalid/unreached sets.")
        num_valid_strats = len(average_strategy_map)
        print(f"Final average strategy contains {num_valid_strats:,} valid information sets.")
        return average_strategy_map


    def _save_final_strategy(self, output_directory, strategy_map):
        """ Saves the final calculated average strategy dictionary. """
        if not output_directory: return
        final_path = os.path.join(output_directory, "final_strategy.pkl")
        try:
            with open(final_path, 'wb') as f:
                pickle.dump(strategy_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\nFinal average strategy saved successfully: {final_path} ({len(strategy_map):,} sets)")
        except Exception as e:
            print(f"\nERROR saving final strategy to {final_path}: {e}")


    def _save_checkpoint(self, output_directory, current_iteration):
        """ Saves the full trainer state (iterations, infosets, config). """
        if not output_directory: return
        checkpoint_data = {
            'iterations': current_iteration,
            'information_sets': self.information_sets,
            'num_players': self.num_players,
            'use_card_abstraction': self.use_card_abstraction,
            'use_action_abstraction': self.use_action_abstraction,
            'training_start_time': self.training_start_time,
            # Custom action func is NOT picklable, cannot save/load easily
            #'get_actions_override_name': getattr(self.get_actions_override, '__name__', None)
        }
        checkpoint_path = os.path.join(output_directory, f"cfr_checkpoint_{current_iteration}.pkl")
        try:
             with open(checkpoint_path, 'wb') as f:
                 pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"\nERROR saving checkpoint to {checkpoint_path}: {e}")


    def load_checkpoint(self, checkpoint_path):
        """ Loads trainer state from a checkpoint file to resume training. """
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}. Cannot load.")
            return False

        try:
            print(f"Loading checkpoint from: {checkpoint_path}...")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            self.iterations = data.get('iterations', 0)
            loaded_sets = data.get('information_sets', {})
            if isinstance(loaded_sets, dict):
                self.information_sets = loaded_sets
                if self.information_sets:
                    try:
                        first_val = next(iter(self.information_sets.values()))
                        if not isinstance(first_val, InformationSet):
                             raise TypeError("Invalid information_sets format")
                    except StopIteration: pass
            else:
                print("ERROR: Checkpoint 'information_sets' is not dict.")
                self.information_sets = {}; self.iterations = 0; return False

            self.num_players = data.get('num_players', self.num_players)
            self.use_card_abstraction = data.get('use_card_abstraction', self.use_card_abstraction)
            # Need to decide how to handle use_action_abstraction if loading ckpt
            # If a custom func was used before, it won't be restored.
            # Reset use_action_abstraction based on saved value?
            self.use_action_abstraction = data.get('use_action_abstraction', self.use_action_abstraction)
            self.training_start_time = data.get('training_start_time', time.time())
            self.get_actions_override = None # Override cannot be loaded

            print(f"Checkpoint loaded successfully. Resuming from iteration {self.iterations + 1}.")
            print(f"  Loaded {len(self.information_sets):,} information sets.")
            print("  NOTE: custom_get_actions_func override is NOT restored.")
            return True

        except (pickle.UnpicklingError, EOFError, TypeError, KeyError, AttributeError) as e:
             print(f"ERROR loading checkpoint: Invalid format or data - {e}")
             traceback.print_exc()
             self.iterations = 0; self.information_sets = {}; self.training_start_time = None; return False
        except Exception as e:
             print(f"ERROR loading checkpoint: Unexpected error - {e}")
             traceback.print_exc()
             self.iterations = 0; self.information_sets = {}; self.training_start_time = None; return False

# --- END OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
