# --- START OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
"""
Implementation of Counterfactual Regret Minimization (CFR) for poker.
(Refactored V15: Added timing logs to _calculate_cfr)
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import sys
import traceback
import time # <<< Import time for logging

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__)); parent_dir = os.path.dirname(script_dir); grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path: sys.path.append(grandparent_dir)

# Imports
try:
    from organized_poker_bot.cfr.information_set import InformationSet
    from organized_poker_bot.cfr.card_abstraction import CardAbstraction
    from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
    from organized_poker_bot.game_engine.game_state import GameState
    from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction
except ImportError as e: print(f"Error importing modules in cfr_trainer.py: {e}"); sys.exit(1)

# Recursion limit
try:
    current_limit = sys.getrecursionlimit(); target_limit = 3000 # Default target
    if current_limit < target_limit:
        try:
            sys.setrecursionlimit(target_limit)
            print(f"Set Recursion Limit -> {target_limit}")
        except Exception as e_rec:
             print(f"WARN: Could not set recursion limit to {target_limit}: {e_rec}. Using current {current_limit}.")
    else:
        print(f"Current Recursion Limit ({current_limit}) sufficient.")
    # Update the internal limit based on the *actual* limit Python is using
    CFRTrainer_REC_LIMIT = sys.getrecursionlimit() - 50 # Use a slightly smaller margin
    if CFRTrainer_REC_LIMIT < 50: CFRTrainer_REC_LIMIT = sys.getrecursionlimit() // 2 # Ensure a positive value
except Exception as e: print(f"Could not check/set recursion limit: {e}"); CFRTrainer_REC_LIMIT=1000 # Default failsafe

class CFRTrainer:
    # Use the limit determined above
    RECURSION_DEPTH_LIMIT = CFRTrainer_REC_LIMIT if 'CFRTrainer_REC_LIMIT' in globals() else 1000 # Failsafe

    def __init__(self, game_state_class, num_players=2, use_action_abstraction=True, use_card_abstraction=True):
        if not callable(game_state_class): raise TypeError("game_state_class must be callable")
        self.game_state_class = game_state_class; self.num_players = num_players
        self.information_sets = {}; self.iterations = 0
        self.use_action_abstraction = use_action_abstraction
        self.use_card_abstraction = use_card_abstraction


    def train(self, iterations=1000, checkpoint_freq=100, output_dir=None, verbose=False): # Ensure verbose has default False
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        start_iter = self.iterations
        end_iter = self.iterations + iterations
        print(f"Starting CFR training from iter {start_iter + 1} to {end_iter}.")
        # Reduce tqdm noise if not verbose
        pbar_disable = not (verbose or iterations > 50) # Only show progress bar if verbose or many iterations
        pbar = tqdm(range(start_iter, end_iter), desc="CFR Training", initial=start_iter, total=end_iter, disable=pbar_disable)

        for i in pbar:
            iter_num = self.iterations + 1 # Use running total
            if not pbar_disable: pbar.set_description(f"CFR Training (Iter {iter_num})")
            if verbose:
                print(f"\n===== Iteration {iter_num} =====")
            game_state = None
            try:
                game_state = self.game_state_class(self.num_players)
                initial_stacks = [10000] * self.num_players # Simplification for now
                dealer_pos = (iter_num - 1) % self.num_players
                game_state.start_new_hand(dealer_pos=dealer_pos, player_stacks=initial_stacks)
                if verbose:
                    print(f" Start Hand {iter_num} - Dealer: P{dealer_pos}\n{game_state}")
            except Exception as e:
                print(f"\nERROR start hand iter {iter_num}: {e}")
                traceback.print_exc()
                continue # Skip to next iteration

            reach_probs = np.ones(self.num_players)
            expected_values = []
            for player_idx in range(self.num_players):
                if verbose:
                    print(f"\n--- Iter {iter_num} | Perspective: Player {player_idx} ---")
                try:
                    # *** Ensure verbose=verbose is passed to the recursive call ***
                    ev = self._calculate_cfr(game_state.clone(), reach_probs.copy(), player_idx, 1.0, 0.0, 0, verbose=verbose)
                    expected_values.append(ev)
                except RecursionError as re:
                    print(f"\nFATAL RecursionError P{player_idx} iter {iter_num}. Limit {self.RECURSION_DEPTH_LIMIT}? {re}\nState:\n{game_state}")
                    pbar.close()
                    # Try to return the strategy calculated so far
                    return self.get_strategy()
                except Exception as e:
                    print(f"\nERROR CFR calc P{player_idx} iter {iter_num}: {e}\nState:\n{game_state}")
                    traceback.print_exc()
                    expected_values.append(None) # Mark error for this player's traversal

            # Only update iteration count and save if the iteration didn't crash badly for ALL players
            if any(ev is not None for ev in expected_values): # If at least one player traversal succeeded
                 self.iterations = iter_num # Update iteration count *after* some success
                 if output_dir and (self.iterations % checkpoint_freq == 0) and self.iterations > 0:
                     print(f"\nSaving checkpoint at iteration {self.iterations}...")
                     self._save_checkpoint(output_dir, self.iterations)
                 if not pbar_disable and (iter_num % 10 == 0 or verbose or iter_num==end_iter):
                      pbar.set_postfix({"InfoSets": len(self.information_sets)}, refresh=True)
            else:
                 print(f"Skipping iteration {iter_num} due to errors in all player perspectives.")
                 continue # Don't save checkpoint or count this iteration if all failed


        pbar.close()
        print("\nTraining loop finished.")
        final_strategy = self.get_strategy()
        if output_dir:
            final_strategy_path = os.path.join(output_dir, "final_strategy.pkl")
            try:
                with open(final_strategy_path, 'wb') as f:
                    pickle.dump(final_strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Final strategy saved to {final_strategy_path}")
            except Exception as e:
                print(f"ERROR saving final strategy: {e}")
        return final_strategy

    # *** MODIFIED _calculate_cfr with Logging ***
    def _calculate_cfr(self, game_state, reach_probs, player_idx, weight=1.0, prune_threshold=0.0, depth=0, verbose=False):
        indent = "  " * depth
        start_time_node = time.time() # <<< ADD: Time node entry
        current_round = game_state.betting_round if hasattr(game_state, 'betting_round') else -1
        current_pot = game_state.pot if hasattr(game_state, 'pot') else -1

        if verbose:
            print(f"\n{indent}D{depth}| Enter CFR | PerspP{player_idx} | Rnd:{current_round} Pot:{current_pot:.0f} Reach:{reach_probs[player_idx]:.3e} (W:{weight:.1f})")

        if depth > self.RECURSION_DEPTH_LIMIT:
            raise RecursionError(f"Manual depth limit ({self.RECURSION_DEPTH_LIMIT}) hit at D{depth}")

        if game_state.is_terminal():
            utility = game_state.get_utility(player_idx)
            utility = utility if isinstance(utility, (int, float)) else 0.0
            if verbose:
                 node_duration = time.time() - start_time_node
                 print(f"{indent}D{depth}| Terminal. Util P{player_idx}: {utility:.2f} (Node took {node_duration:.4f}s)")
            return utility

        # Ensure valid current player
        if not hasattr(game_state, 'current_player_idx') or not (0 <= game_state.current_player_idx < self.num_players):
             if verbose:
                 node_duration = time.time() - start_time_node
                 print(f"{indent}D{depth}| WARN: Invalid player idx {getattr(game_state, 'current_player_idx', 'N/A')}. Terminalizing. (Node took {node_duration:.4f}s)")
             return game_state.get_utility(player_idx) # Evaluate current state

        acting_player_idx = game_state.current_player_idx

        # Check player status bounds
        if acting_player_idx >= len(game_state.player_folded) or \
           acting_player_idx >= len(game_state.player_all_in) or \
           acting_player_idx >= len(game_state.player_stacks):
           if verbose:
               node_duration = time.time() - start_time_node
               print(f"{indent}D{depth}| WARN: P idx {acting_player_idx} OOB lists. Terminalizing. (Node took {node_duration:.4f}s)")
           return game_state.get_utility(player_idx)

        is_folded = game_state.player_folded[acting_player_idx]
        is_all_in = game_state.player_all_in[acting_player_idx]

        # Skip inactive players (handle potential state loops if skipping doesn't change player)
        if is_folded or is_all_in:
             if verbose: print(f"{indent}D{depth}| Skip inactive P{acting_player_idx} (Folded:{is_folded}, AllIn:{is_all_in}). Trying move.")
             # IMPORTANT: Use clone to attempt move, original state unchanged for caller
             temp_state = game_state.clone()
             temp_state._move_to_next_player()

             # Check if the state actually changed or became terminal
             if temp_state.current_player_idx == acting_player_idx or temp_state.current_player_idx == -1 or temp_state.is_terminal():
                  if verbose:
                      node_duration = time.time() - start_time_node
                      print(f"{indent}D{depth}| Skip resulted in loop or terminal. P{temp_state.current_player_idx}, IsTerm={temp_state.is_terminal()}. Evaluating. (Node took {node_duration:.4f}s)")
                  utility = temp_state.get_utility(player_idx)
                  return utility if isinstance(utility, (int, float)) else 0.0
             else:
                  if verbose: print(f"{indent}D{depth}| Recursing after skip & move to P{temp_state.current_player_idx}")
                  # Recurse on the *temporary* state after move
                  # *** Crucially, use original game_state's reach_probs ***
                  recursive_call_start_time = time.time()
                  result_ev = self._calculate_cfr(temp_state, reach_probs, player_idx, weight, prune_threshold, depth + 1, verbose)
                  recursive_call_duration = time.time() - recursive_call_start_time
                  node_duration = time.time() - start_time_node
                  if verbose: print(f"{indent}D{depth}| <- Returned from skipped player recursion. EV:{result_ev:.3f}. (NodeTook:{node_duration:.4f}s, RecCall:{recursive_call_duration:.4f}s)")
                  return result_ev # Return the value obtained by the actual next actor

        # If reached here, it's the acting_player_idx's turn to make a choice
        if verbose: print(f"{indent}D{depth}| Active P{acting_player_idx} turn.")
        state_hash_approx = hash(str(game_state)) # Basic state hash for debug

        key_start_time = time.time()
        info_set_key = self._create_info_set_key(game_state, acting_player_idx)
        key_duration = time.time() - key_start_time

        avail_start_time = time.time()
        available_actions = game_state.get_available_actions()
        avail_duration = time.time() - avail_start_time

        if verbose:
             print(f"{indent}D{depth}| P{acting_player_idx} Acting. StateHash:{state_hash_approx}") # Approx state hash
             print(f"{indent}D{depth}| Key='{info_set_key}' (Took {key_duration:.4f}s)")
             print(f"{indent}D{depth}| Raw Actions ({len(available_actions)}): {available_actions} (Took {avail_duration:.4f}s)")

        abs_duration = 0.0
        if self.use_action_abstraction and available_actions:
             try:
                 abs_start_time = time.time()
                 abstracted = ActionAbstraction.abstract_actions(available_actions, game_state)
                 abs_duration = time.time() - abs_start_time
                 if verbose: print(f"{indent}D{depth}| Abstracted Actions ({len(abstracted)}): {abstracted} (Took {abs_duration:.4f}s)")
                 available_actions = abstracted or available_actions
             except Exception as e:
                 print(f"{indent}ERROR action abstraction: {e}. Key={info_set_key}")


        if not available_actions:
             # Should ideally not happen if player isn't folded/all-in and round not over.
             if verbose:
                 node_duration = time.time() - start_time_node
                 print(f"{indent}WARN D{depth}: No actions for active P{acting_player_idx}. Evaluating state. (Node took {node_duration:.4f}s)")
             return game_state.get_utility(player_idx)

        info_set = self._get_or_create_info_set(info_set_key, available_actions)
        if info_set is None:
             if verbose:
                 node_duration = time.time() - start_time_node
                 print(f"{indent}WARN D{depth}: Failed get/create info set {info_set_key}. Evaluating state. (Node took {node_duration:.4f}s)")
             return game_state.get_utility(player_idx) # Should we error instead?

        strategy = info_set.get_strategy()

        if verbose:
             strat_str = '{' + ', '.join(f"'{a[0]}{a[1] if a[0] not in ('fold','check') else ''}':{p:.2f}" for a,p in strategy.items()) + '}'
             print(f"{indent}D{depth}| PerspP{player_idx}|ActP{acting_player_idx}|Strat={strat_str}")

        expected_value = 0.0
        action_values = {} # Stores EV for perspective player IF action is taken

        # --- Action loop ---
        for action_idx, action in enumerate(available_actions):
            action_prob = strategy.get(action, 0.0)
            action_str = f"{action[0]}{action[1] if action[0] not in ('fold','check') else ''}"
            action_branch_start_time = time.time() # <<< Time this branch
            if verbose: print(f"{indent}D{depth}| -> Exploring action {action_idx+1}/{len(available_actions)}: {action_str} (Prob: {action_prob:.3f})")

            # Pruning logic can be added here if needed
            if action_prob <= prune_threshold and len(available_actions) > 1 and prune_threshold > 0:
                 if verbose: print(f"{indent}D{depth}|    Pruning action {action_str}")
                 continue # Skip this action

            # Calculate reach probabilities for the next state
            new_reach_probs = reach_probs.copy()
            new_reach_probs[acting_player_idx] *= action_prob

            # Apply action to get next state
            try:
                 apply_start_time = time.time()
                 # Pass clone to apply_action if it modifies state, or clone result if needed
                 next_game_state = game_state.apply_action(action) # Assume apply_action returns new state
                 apply_duration = time.time() - apply_start_time
                 if verbose and apply_duration > 0.01: # Only log if takes noticeable time
                      print(f"{indent}D{depth}|    apply_action took {apply_duration:.4f}s")
            except Exception as e:
                 print(f"{indent}ERROR apply action {action} by P{acting_player_idx} D{depth}: {e}")
                 # traceback.print_exc() # Can be very noisy, use if needed
                 # Cannot proceed with this action, maybe assign very negative value? Or just skip?
                 # Skipping might bias regrets. Assigning a value is tricky.
                 # Let's skip for now, might need refinement.
                 continue

            # Recursive Call
            if verbose: print(f"{indent}D{depth}|    Recursing... (Depth {depth+1})")
            recursive_call_start_time = time.time() # <<< Time recursive call
            # Recurse: Use the *next* game state and updated reach probs
            action_ev = self._calculate_cfr(next_game_state, new_reach_probs, player_idx, weight, prune_threshold, depth + 1, verbose)
            recursive_call_duration = time.time() - recursive_call_start_time # <<< ADD

            action_values[action] = action_ev # Store result

            # Update overall expected value for the node (from perspective player's POV)
            expected_value += action_prob * action_ev

            # Log action branch completion and timing
            action_branch_duration = time.time() - action_branch_start_time # <<< ADD Timing
            if verbose:
                print(f"{indent}D{depth}|    <- Returned from Act {action_str}. EV P{player_idx}={action_ev:.4f} (Rec Call took {recursive_call_duration:.4f}s)")
                if action_branch_duration > 0.1: # Log only if branch took noticeable time
                     print(f"{indent}D{depth}|    Action Branch {action_str} took {action_branch_duration:.4f}s total.")


        # --- Update Regrets and Strategy Sum --- (Only if it was acting player's perspective)
        if acting_player_idx == player_idx:
             # Calculate reach probability of opponents for weighting regret/strategy updates
             opp_reach = np.prod(np.concatenate((reach_probs[:player_idx], reach_probs[player_idx+1:]))) if self.num_players > 1 else 1.0
             # Include iteration weight if using variant like CFR+ or Linear CFR (weight is often iteration number)
             cfr_reach = opp_reach * weight # weight incorporates iteration T

             if verbose: print(f"{indent}D{depth}| **Update P{player_idx}** | NodeEV:{expected_value:.3f} | OppReach:{opp_reach:.3e} CfrReach:{cfr_reach:.3f}")

             # Update regrets
             for action in available_actions:
                  action_str_upd = f"{action[0]}{action[1] if action[0] not in ('fold','check') else ''}"
                  # Regret is the value of taking the action minus the expected value of the node
                  # Ensure action exists in action_values, default to 0 if pruned/error
                  instant_regret = action_values.get(action, 0.0) - expected_value
                  # Update cumulative regret sum, weighted by opponent reach
                  info_set.regret_sum[action] += cfr_reach * instant_regret # Directly modify defaultdict value
                  if verbose and abs(instant_regret) > 0.01 : # Only print significant regrets
                       print(f"{indent}D{depth}|   RegUp: Act={action_str_upd} | InstR:{instant_regret:.3f} | NewSum:{info_set.regret_sum[action]:.3f}")

             # Update strategy sum (used for average strategy calculation)
             player_reach = reach_probs[player_idx]
             # Update sum weighted by player's reach probability * iteration weight
             info_set.update_strategy_sum(strategy, player_reach * weight)
             if verbose: print(f"{indent}D{depth}|   StratSumUp w/ PReach: {player_reach:.3e} * W:{weight:.1f}")


        # --- Return Node Value ---
        node_duration = time.time() - start_time_node # <<< ADD: Log node duration
        if verbose and node_duration > 0.2: # Log only if node took noticeable time
            print(f"{indent}D{depth}| Exit CFR Node | Return EV:{expected_value:.4f} for P{player_idx} (Node took {node_duration:.4f}s)")

        return expected_value


    # --- _create_info_set_key (Unchanged, use previous corrected version) ---
    def _create_info_set_key(self, game_state, player_idx):
        cards_part = "NOCARDS"
        try:
            hole = game_state.hole_cards[player_idx] if player_idx < len(game_state.hole_cards) and game_state.hole_cards else []
            comm = game_state.community_cards if hasattr(game_state, 'community_cards') else []
            if self.use_card_abstraction and hole:
                 # Determine Round Name for Abstraction Key
                 rnd_map={0:"PRE", 3:"FLOP", 4:"TURN", 5:"RIVER"} # Use len(comm) for postflop stages
                 round_len = 0 if not comm else len(comm)
                 rnd_name = rnd_map.get(round_len, f"POST{round_len}")

                 if rnd_name == "PRE":
                      cards_part = f"{rnd_name}_{CardAbstraction.get_preflop_abstraction(hole)}"
                 else: # Postflop
                      # Get (strength_bucket, board_paired, board_flush_suit)
                      postflop_key_tuple = CardAbstraction.get_postflop_abstraction(hole, comm)
                      s_b, b_p, b_f = postflop_key_tuple
                      cards_part = f"{rnd_name}_{s_b}_P{b_p}_F{b_f}"
                      # Example Key: FLOP_7_P0_Fn (Flop, Bucket 7, Board Unpaired, No Flush Suit)
                      # Example Key: TURN_3_P1_Fs (Turn, Bucket 3, Board Paired, Spades Flush Suit)

            elif hole: # If not using abstraction, use raw cards (potentially huge state space)
                 cards_part=f"RAW|{'_'.join(sorted(str(c) for c in hole))}|{'_'.join(sorted(str(c) for c in comm))}"
        except Exception as e:
             # traceback.print_exc() # Optionally print detailed error
             cards_part = f"CARDS_ERR_{e.__class__.__name__}"

        # Position part
        pos_part = "ERR_POS"
        try: pos_part = f"POS_{game_state.get_position(player_idx)}"
        except Exception as e: pos_part = f"POS_ERR_{e.__class__.__name__}"

        # Betting history part (use the existing simplified version for now)
        hist_part = "BH_ERR"
        try: hist_part = game_state.get_betting_history()
        except Exception as e: hist_part = f"BH_ERR_{e.__class__.__name__}"

        return f"{cards_part}|{pos_part}|{hist_part}"

    # --- _get_or_create_info_set (Unchanged) ---
    def _get_or_create_info_set(self, info_set_key, available_actions):
         if info_set_key not in self.information_sets:
             if not isinstance(available_actions, list):
                 available_actions = []
             # Ensure actions are valid tuples
             action_list = []
             seen_repr = set() # Prevent duplicates with same basic form (e.g., raise 100, raise 100.0)
             for a in available_actions:
                  act_tuple = None
                  if isinstance(a, tuple) and len(a) == 2:
                       act_type, amt = a
                       try: # Ensure amount is reasonable numeric, round for consistency key?
                           amt_int = int(round(float(amt))) if amt is not None else 0
                           act_tuple = (str(act_type), amt_int)
                       except (ValueError, TypeError): continue # Skip malformed action
                  elif isinstance(a, str): # Treat simple string as type with amount 0
                      act_tuple = (a, 0)
                  else: continue # Skip other invalid formats

                  action_repr = f"{act_tuple[0]}_{act_tuple[1]}"
                  if act_tuple is not None and action_repr not in seen_repr:
                       action_list.append(act_tuple)
                       seen_repr.add(action_repr)

             if action_list: # Only create if valid actions exist
                 self.information_sets[info_set_key] = InformationSet(action_list)
             else:
                 #print(f"WARN: No valid actions provided to create infoset: {info_set_key} from {available_actions}")
                 return None # Cannot create info set without actions
         return self.information_sets[info_set_key]

    # --- get_strategy (Unchanged) ---
    def get_strategy(self):
        average_strategy = {}
        num_sets = len(self.information_sets)
        count_invalid = 0
        use_tqdm_avg = num_sets > 5000 # Use tqdm only for large numbers
        print(f"Calculating average strategy from {num_sets} info sets...")

        items = tqdm(self.information_sets.items(), desc="Avg Strat", total=num_sets, disable=not use_tqdm_avg)

        for key, info_set_obj in items:
            if not isinstance(info_set_obj, InformationSet):
                count_invalid += 1
                continue
            try:
                avg_strat = info_set_obj.get_average_strategy()
                # Validate average strategy before adding
                if isinstance(avg_strat, dict) and avg_strat: # Check not empty
                    prob_sum = sum(avg_strat.values())
                    if abs(prob_sum - 1.0) < 0.01: # Check normalization
                         # Ensure keys are simple hashable tuples
                         clean_strat = {k:v for k, v in avg_strat.items() if isinstance(k, tuple)}
                         if len(clean_strat) == len(avg_strat): # Check all keys were valid tuples
                            average_strategy[key] = clean_strat
                         else: count_invalid += 1 # Contained invalid keys
                    else:
                         # If sum near zero, maybe unreached? Log if significantly non-normalized.
                         if abs(prob_sum)>0.01 : print(f"WARN: Invalid prob sum {prob_sum:.4f} for key {key}")
                         count_invalid += 1
                else:
                    count_invalid += 1 # Info set obj failed to return valid dict or was empty
            except Exception as e:
                print(f"ERROR getting avg strategy for key {key}: {e}")
                traceback.print_exc()
                count_invalid += 1
        if count_invalid > 0:
            print(f"WARNING: Skipped {count_invalid}/{num_sets} invalid info sets during averaging.")
        print(f"Final strategy contains {len(average_strategy)} valid information sets.")
        return average_strategy

    # --- _save_checkpoint / load_checkpoint (Unchanged) ---
    def _save_checkpoint(self, output_dir, iteration):
        data={'iterations':self.iterations,'information_sets':self.information_sets,'num_players':self.num_players,'use_card_abstraction':self.use_card_abstraction,'use_action_abstraction':self.use_action_abstraction}
        path = os.path.join(output_dir, f"cfr_checkpoint_{iteration}.pkl")
        try:
             with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
             print(f"Chkpt saved: {path}")
        except Exception as e: print(f"ERROR saving checkpoint {path}: {e}")

    def load_checkpoint(self, checkpoint_path):
        try:
            print(f"Loading chkpt: {checkpoint_path}...");
            with open(checkpoint_path,'rb') as f: data=pickle.load(f);
            self.iterations=data.get('iterations',0);
            loaded_sets=data.get('information_sets',{});
            if isinstance(loaded_sets,dict):
                 # Quick validation of loaded structure
                 valid = True; count = 0; max_check=min(5, len(loaded_sets))
                 for value in loaded_sets.values():
                     if not isinstance(value, InformationSet): print(f"ERROR: Checkpoint value type {type(value)} invalid."); valid=False; break;
                     count+=1;
                     if count>=max_check: break
                 if valid: self.information_sets = loaded_sets
                 else: print("ERROR: Invalid checkpoint structure (values). Starting fresh."); self.information_sets={}; self.iterations=0
            else:
                print("ERROR: Loaded info sets not a dictionary. Starting fresh."); self.information_sets={}; self.iterations=0
            # Load other config safely
            self.num_players=data.get('num_players', self.num_players)
            self.use_card_abstraction=data.get('use_card_abstraction', self.use_card_abstraction)
            self.use_action_abstraction=data.get('use_action_abstraction', self.use_action_abstraction)
            print(f"Load complete. Resuming from iteration {self.iterations + 1}. Info Sets loaded: {len(self.information_sets)}")
        except FileNotFoundError: print(f"ERROR: Checkpoint not found: {checkpoint_path}. Starting fresh.")
        except ModuleNotFoundError as e: print(f"ERROR checkpoint module {e} not found (pickle error?). Starting fresh."); self.iterations=0; self.information_sets={}
        except Exception as e: print(f"ERROR loading checkpoint: {e}. Starting fresh."); traceback.print_exc(); self.iterations=0; self.information_sets={}
# --- END OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
