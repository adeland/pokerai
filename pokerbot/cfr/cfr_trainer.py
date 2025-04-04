# --- START OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
"""
Implementation of Counterfactual Regret Minimization (CFR) for poker.
(Refactored V14: FINAL fix for all semicolon-based SyntaxErrors)
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import sys
import traceback

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
    current_limit = sys.getrecursionlimit(); target_limit = 3000
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)
        print(f"Set Recursion Limit -> {target_limit}")
    else:
        print(f"Current Recursion Limit ({current_limit}) sufficient.")
except Exception as e: print(f"Could not set recursion limit: {e}")

class CFRTrainer:
    RECURSION_DEPTH_LIMIT = sys.getrecursionlimit() - 500

    def __init__(self, game_state_class, num_players=2, use_action_abstraction=True, use_card_abstraction=True):
        if not callable(game_state_class): raise TypeError("game_state_class must be callable")
        self.game_state_class = game_state_class; self.num_players = num_players
        self.information_sets = {}; self.iterations = 0
        self.use_action_abstraction = use_action_abstraction
        self.use_card_abstraction = use_card_abstraction


    def train(self, iterations=1000, checkpoint_freq=100, output_dir=None, verbose=False):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        start_iter = self.iterations
        end_iter = self.iterations + iterations
        print(f"Starting CFR training from iter {start_iter + 1} to {end_iter}.")
        pbar = tqdm(range(start_iter, end_iter), desc="CFR Training", initial=start_iter, total=end_iter)

        for i in pbar:
            iter_num = self.iterations + 1 # Use running total
            pbar.set_description(f"CFR Training (Iter {iter_num})")
            if verbose:
                print(f"\n===== Iteration {iter_num} =====")
            game_state = None
            try:
                game_state = self.game_state_class(self.num_players)
                initial_stacks = [10000] * self.num_players # Simplification
                dealer_pos = (iter_num - 1) % self.num_players
                game_state.start_new_hand(dealer_pos=dealer_pos, player_stacks=initial_stacks)
                if verbose:
                    print(f" Start Hand {iter_num} - Dealer: P{dealer_pos}\n{game_state}")
            except Exception as e:
                print(f"\nERROR start hand iter {iter_num}: {e}")
                traceback.print_exc()
                continue

            reach_probs = np.ones(self.num_players)
            expected_values = []
            for player_idx in range(self.num_players):
                if verbose:
                    print(f"\n--- Iter {iter_num} | Perspective: Player {player_idx} ---")
                try:
                    ev = self._calculate_cfr(game_state.clone(), reach_probs.copy(), player_idx, 1.0, 0.0, 0, verbose)
                    expected_values.append(ev)
                except RecursionError as re:
                    print(f"\nFATAL RecursionError P{player_idx} iter {iter_num}. Limit {self.RECURSION_DEPTH_LIMIT}? {re}\nState:\n{game_state}")
                    pbar.close()
                    return self.get_strategy()
                except Exception as e:
                    print(f"\nERROR CFR calc P{player_idx} iter {iter_num}: {e}\nState:\n{game_state}")
                    traceback.print_exc()
                    expected_values.append(None)

            if any(ev is None for ev in expected_values):
                print(f"Skipping iteration {iter_num} errors.")
                continue

            self.iterations = iter_num # Update iteration count *after* success
            if output_dir and (self.iterations % checkpoint_freq == 0) and self.iterations > 0:
                 print(f"\nSaving checkpoint at iteration {self.iterations}...")
                 self._save_checkpoint(output_dir, self.iterations)
            if iter_num % 10 == 0 or verbose:
                pbar.set_postfix({"InfoSets": len(self.information_sets)}, refresh=True)

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


    def _calculate_cfr(self, game_state, reach_probs, player_idx, weight=1.0, prune_threshold=0.0, depth=0, verbose=False):
        indent = "  " * depth
        current_round = game_state.betting_round if hasattr(game_state, 'betting_round') else -1
        current_pot = game_state.pot if hasattr(game_state, 'pot') else -1

        if verbose:
            print(f"\n{indent}D{depth}| Enter CFR | PerspP{player_idx} | Rnd:{current_round} Pot:{current_pot:.0f}")

        if depth > self.RECURSION_DEPTH_LIMIT:
            raise RecursionError(f"Manual depth limit ({self.RECURSION_DEPTH_LIMIT}) at D{depth}")

        if game_state.is_terminal():
            utility = game_state.get_utility(player_idx)
            utility = utility if isinstance(utility, (int, float)) else 0.0
            if verbose:
                print(f"{indent}D{depth}| Terminal. Util P{player_idx}: {utility:.2f}")
            return utility

        if not hasattr(game_state, 'current_player_idx') or not (0 <= game_state.current_player_idx < self.num_players):
             if verbose:
                 print(f"{indent}D{depth}| WARN: Invalid player idx {getattr(game_state, 'current_player_idx', 'N/A')}. Terminal.")
             return game_state.get_utility(player_idx) # Evaluate current state

        acting_player_idx = game_state.current_player_idx

        if acting_player_idx >= len(game_state.player_folded) or acting_player_idx >= len(game_state.player_all_in):
           if verbose:
               print(f"{indent}D{depth}| WARN: P idx {acting_player_idx} OOB. Terminal.")
           return game_state.get_utility(player_idx)

        is_folded = game_state.player_folded[acting_player_idx]
        is_all_in = game_state.player_all_in[acting_player_idx]

        if is_folded or is_all_in:
             if verbose:
                 print(f"{indent}D{depth}| Skip inactive P{acting_player_idx}. Trying move.")
             temp_state = game_state.clone()
             temp_state._move_to_next_player()
             if temp_state.current_player_idx == acting_player_idx or temp_state.current_player_idx == -1 or temp_state.is_terminal():
                 if verbose:
                     print(f"{indent}D{depth}| Skip move result P{temp_state.current_player_idx}, IsTerm={temp_state.is_terminal()}. Eval.")
                 utility = temp_state.get_utility(player_idx)
                 return utility if isinstance(utility, (int, float)) else 0.0
             else:
                 if verbose:
                     print(f"{indent}D{depth}| Recursing after skip & move to P{temp_state.current_player_idx}")
                 return self._calculate_cfr(temp_state, reach_probs, player_idx, weight, prune_threshold, depth + 1, verbose)

        if verbose:
            print(f"{indent}D{depth}| Active P{acting_player_idx} turn.")

        info_set_key = self._create_info_set_key(game_state, acting_player_idx)
        available_actions = game_state.get_available_actions()

        if verbose:
            print(f"{indent}D{depth}| Raw Actions: {available_actions}")

        if self.use_action_abstraction and available_actions:
             try:
                 abstracted = ActionAbstraction.abstract_actions(available_actions, game_state)
                 if verbose:
                     print(f"{indent}D{depth}| Abstracted Actions: {abstracted}")
                 available_actions = abstracted or available_actions
             except Exception as e:
                 print(f"ERROR action abstraction: {e}. Key={info_set_key}")

        if not available_actions:
            if verbose:
                print(f"{indent}WARN D{depth}: No actions for active P{acting_player_idx}. Eval state.")
            return game_state.get_utility(player_idx)

        info_set = self._get_or_create_info_set(info_set_key, available_actions)
        if info_set is None:
             if verbose:
                 print(f"{indent}WARN D{depth}: Failed get/create info set {info_set_key}. Eval state.")
             return game_state.get_utility(player_idx)

        strategy = info_set.get_strategy()

        if verbose:
             strat_str = '{' + ', '.join(f"'{a[0]}{a[1] if a[0] not in ('fold','check') else ''}':{p:.2f}" for a,p in strategy.items()) + '}'
             print(f"{indent}D{depth}| PerspP{player_idx}|ActP{acting_player_idx}|Key={info_set_key}|Strat={strat_str}")

        expected_value = 0.0
        action_values = {}

        for action_idx, action in enumerate(available_actions):
            action_prob = strategy.get(action, 0.0)
            action_str = f"{action[0]}{action[1] if action[0] not in ('fold','check') else ''}"
            if verbose:
                print(f"{indent}D{depth}| -> Exploring action {action_idx+1}/{len(available_actions)}: {action_str} (Prob: {action_prob:.3f})")

            if action_prob <= prune_threshold and len(available_actions) > 1 and prune_threshold > 0:
                if verbose:
                    print(f"{indent}D{depth}|    Pruning action {action_str}")
                continue

            new_reach_probs = reach_probs.copy()
            new_reach_probs[acting_player_idx] *= action_prob

            try:
                next_game_state = game_state.apply_action(action)
            except Exception as e:
                print(f"{indent}ERROR apply action {action} by P{acting_player_idx} D{depth}: {e}")
                traceback.print_exc()
                continue

            if verbose:
                print(f"{indent}D{depth}|    Recursing... (Depth {depth+1})")

            action_ev = self._calculate_cfr(next_game_state, new_reach_probs, player_idx, weight, prune_threshold, depth + 1, verbose)
            action_values[action] = action_ev

            if verbose:
                print(f"{indent}D{depth}|    <- Returned from Act {action_str}. EV P{player_idx}={action_ev:.4f}")

            expected_value += action_prob * action_ev

        if acting_player_idx == player_idx:
            opp_reach = np.prod(np.concatenate((reach_probs[:player_idx], reach_probs[player_idx+1:]))) if self.num_players > 1 else 1.0
            cfr_reach = opp_reach * weight
            if verbose:
                print(f"{indent}D{depth}| **Update P{player_idx}** | NodeEV:{expected_value:.3f} | CFRReach:{cfr_reach:.3f}")

            for action in available_actions:
                 action_str_upd = f"{action[0]}{action[1] if action[0] not in ('fold','check') else ''}"
                 regret = action_values.get(action, 0.0) - expected_value
                 if action in info_set.regret_sum:
                    info_set.regret_sum[action] += cfr_reach * regret
                 else:
                    info_set.regret_sum[action] = cfr_reach * regret
                 if verbose:
                    print(f"{indent}D{depth}|   RegUp: Act={action_str_upd} | InstR:{regret:.3f} | NewSum:{info_set.regret_sum[action]:.3f}")

            player_reach = reach_probs[player_idx]
            info_set.update_strategy_sum(strategy, player_reach * weight)
            if verbose:
                print(f"{indent}D{depth}|   StratSumUp w/ Reach: {player_reach:.3f}")

        if verbose:
            print(f"{indent}D{depth}| Exit CFR | Return EV:{expected_value:.4f} for P{player_idx}")

        return expected_value


    def _create_info_set_key(self, game_state, player_idx):
        cards_part = "NOCARDS" # CORRECTED: No semicolon
        try:
            hole = game_state.hole_cards[player_idx] if player_idx < len(game_state.hole_cards) and game_state.hole_cards else []
            comm = game_state.community_cards if hasattr(game_state, 'community_cards') else [] # CORRECTED: No semicolon
            if self.use_card_abstraction and hole:
                if not comm:
                    cards_part = f"PRE_{CardAbstraction.get_preflop_abstraction(hole)}"
                elif len(comm) >= 3:
                     rnd_map={3:"FLOP", 4:"TURN", 5:"RIVER"}
                     rnd_name = rnd_map.get(len(comm), f"POST{len(comm)}")
                     s_b, b_p, b_f = CardAbstraction.get_postflop_abstraction(hole, comm)
                     cards_part = f"{rnd_name}_{s_b}_P{b_p}_F{b_f}"
                else:
                    cards_part = f"PRE_{CardAbstraction.get_preflop_abstraction(hole)}"
            elif hole:
                cards_part=f"RAW|{'_'.join(sorted(str(c) for c in hole))}|{'_'.join(sorted(str(c) for c in comm))}"
        except Exception as e:
            cards_part = f"CARDS_ERR_{e.__class__.__name__}"

        pos_part = f"POS_{game_state.get_position(player_idx)}" if hasattr(game_state, 'get_position') else f"IDX_{player_idx}"
        hist_part = game_state.get_betting_history() if hasattr(game_state, 'get_betting_history') else "BH_ERR"
        return f"{cards_part}|{pos_part}|{hist_part}"


    def _get_or_create_info_set(self, info_set_key, available_actions):
         if info_set_key not in self.information_sets:
             if not isinstance(available_actions, list):
                 available_actions = []
             action_list = [a for a in available_actions if isinstance(a, tuple) and len(a) == 2]
             if action_list:
                self.information_sets[info_set_key] = InformationSet(action_list)
             else:
                return None # Cannot create info set without actions
         return self.information_sets[info_set_key]

    # --- CORRECTED get_strategy ---
    def get_strategy(self):
        average_strategy = {}
        num_sets = len(self.information_sets)
        count_invalid = 0
        print(f"Calculating average strategy from {num_sets} info sets...")

        use_tqdm = num_sets > 5000
        items = tqdm(self.information_sets.items(), desc="Avg Strat", total=num_sets, disable=not use_tqdm)

        for key, info_set_obj in items:
            if not isinstance(info_set_obj, InformationSet):
                count_invalid += 1
                continue
            try:
                avg_strat = info_set_obj.get_average_strategy()
                if isinstance(avg_strat, dict) and avg_strat:
                    prob_sum = sum(avg_strat.values())
                    if abs(prob_sum - 1.0) < 0.01:
                        average_strategy[key] = avg_strat
                    else:
                        count_invalid += 1
                else:
                    count_invalid += 1
            except Exception as e:
                print(f"ERROR getting avg strategy for key {key}: {e}")
                traceback.print_exc()
                count_invalid += 1
        if count_invalid > 0:
            print(f"WARNING: Skipped {count_invalid}/{num_sets} invalid info sets during averaging.")
        print(f"Final strategy contains {len(average_strategy)} valid information sets.")
        return average_strategy

    # --- _save_checkpoint and load_checkpoint remain correct ---
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
            print(f"Loading chkpt: {checkpoint_path}..."); f=open(checkpoint_path,'rb'); data=pickle.load(f); f.close()
            self.iterations=data.get('iterations',0); sets=data.get('information_sets',{})
            if isinstance(sets,dict):
                valid=True; count=0; max_check=5;
                for value in sets.values():
                    if not isinstance(value, InformationSet): print(f"ERROR: Chkpt type {type(value)} invalid."); valid=False; break;
                    count+=1;
                    if count>=max_check: break
                if valid: self.information_sets = sets
                else: print("ERROR: Invalid chkpt structure. Start fresh."); self.information_sets={}; self.iterations=0
            else: print("ERROR: Info sets not dict. Start fresh."); self.information_sets={}; self.iterations=0
            self.num_players=data.get('num_players', self.num_players)
            self.use_card_abstraction=data.get('use_card_abstraction', self.use_card_abstraction)
            self.use_action_abstraction=data.get('use_action_abstraction', self.use_action_abstraction)
            print(f"Loaded. Resume iter {self.iterations + 1}. Sets: {len(self.information_sets)}")
        except FileNotFoundError: print(f"ERROR: Chkpt not found: {checkpoint_path}. Start fresh.")
        except ModuleNotFoundError as e: print(f"ERROR chkpt module {e} not found. Start fresh."); self.iterations=0; self.information_sets={}
        except Exception as e: print(f"ERROR loading chkpt: {e}. Start fresh."); traceback.print_exc(); self.iterations=0; self.information_sets={}
# --- END OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
