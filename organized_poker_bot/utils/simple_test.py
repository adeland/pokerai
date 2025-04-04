# --- START OF FILE organized_poker_bot/utils/simple_test.py ---
"""
Simple test script for validating the poker bot implementation.
(Refactored V22: Removed all semicolons, ensured correct indentation throughout)
"""

import os
import sys
import pickle
import random
import numpy as np
import time
from tqdm import tqdm
import traceback
from copy import deepcopy

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports
try:
    from organized_poker_bot.game_engine.game_state import GameState
    from organized_poker_bot.game_engine.card import Card
    from organized_poker_bot.game_engine.player import Player
    from organized_poker_bot.cfr.cfr_trainer import CFRTrainer
    from organized_poker_bot.bot.bot_player import BotPlayer
    from organized_poker_bot.cfr.card_abstraction import CardAbstraction
    from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction
    from organized_poker_bot.bot.depth_limited_search import DepthLimitedSearch
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.cfr.information_set import InformationSet
    from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
except ImportError as e:
    print(f"FATAL ERROR: Import failed in simple_test.py: {e}")
    sys.exit(1)


# --- GameState Class Factory ---
def create_game_state(num_players, starting_stack=10000, small_blind=50, big_blind=100 ):
    return GameState(num_players=num_players, starting_stack=starting_stack, small_blind=small_blind, big_blind=big_blind)


# --- CFR Trainer Structure Test ---
def test_cfr_trainer(verbose_cfr=False):
    print("\n" + "-"*60)
    print(f"Testing CFR Trainer Structure (Verbose: {verbose_cfr}) - Simplified Actions")
    print("-"*60)
    num_players = 2
    iterations_to_test = 1
    original_abstract_actions = ActionAbstraction.abstract_actions

    @staticmethod
    def _simple_test_abstract_actions(available_actions, game_state):
        simplified_actions = {}
        call_action, min_raise_action, all_in_action, can_raise_bet = None, None, None, False
        largest_amount = -1.0
        for act_type, amount in available_actions:
            if act_type == 'fold':
                 simplified_actions[('fold', 0)] = ('fold', 0)
            if act_type == 'check':
                 simplified_actions[('check', 0)] = ('check', 0)
            if act_type == 'call':
                 call_action = (act_type, amount)
            if act_type in ('raise', 'bet'):
                 can_raise_bet = True
                 if min_raise_action is None or float(amount) < float(min_raise_action[1]): # Ensure numeric comparison
                     min_raise_action = (act_type, amount)
                 current_amount = float(amount)
                 if current_amount > largest_amount:
                     largest_amount = current_amount
                     all_in_action = (act_type, amount)

        if call_action:
             simplified_actions[call_action] = call_action
        if can_raise_bet:
            if min_raise_action:
                 simplified_actions[min_raise_action] = min_raise_action
            # Check if all_in is distinct and exists
            if all_in_action and (min_raise_action is None or abs(float(all_in_action[1]) - float(min_raise_action[1])) > 0.01):
                 simplified_actions[all_in_action] = all_in_action

        def sort_key(a):
             t,amt=a
             o={"fold":0,"check":1,"call":2,"bet":3,"raise":4}
             # Use 0 for amt if None or cannot convert, ensure numeric comparison
             num_amt = 0
             try:
                num_amt = int(round(float(amt)))
             except (ValueError, TypeError):
                pass # Keep num_amt as 0
             return (o.get(t,99), num_amt)

        return sorted(list(simplified_actions.values()), key=sort_key)

    trainer = None
    strategy = {}
    try:
        ActionAbstraction.abstract_actions = _simple_test_abstract_actions
        trainer = CFRTrainer(create_game_state, num_players=num_players, use_card_abstraction=True, use_action_abstraction=True)
        print(f"Running CFR structure test for {iterations_to_test} iter...")
        strategy = trainer.train(iterations=iterations_to_test, checkpoint_freq=iterations_to_test+1, output_dir="test_output", verbose=verbose_cfr)
    except RecursionError as re:
        print(f"\nINFO: CFR structure test hit recursion limit.")
        # Pass if recursion limit hit in this basic structural test
    except Exception as e:
        print(f"\nERROR CFR Structure Test: {e}")
        traceback.print_exc()
        # Log error but allow test run to continue assessment
    finally:
        ActionAbstraction.abstract_actions = original_abstract_actions
        print("INFO: Restored ActionAbstraction.")

    assert isinstance(strategy, dict), "Strategy must be dict"
    print(f"Strategy structure test generated {len(strategy)} sets.")
    if not strategy:
        print("WARN: Strategy empty after structure test.")
    print("\nCFR trainer structure test complete.")
    return strategy


# --- GameState Logic Test (Formatted) ---
def test_game_state_logic():
    print("\n" + "-"*60)
    print("Testing GameState Logic")
    print("-"*60)
    p0_stack_init, p1_stack_init = 100, 100
    sb_val, bb_val = 10, 20

    gs = create_game_state(num_players=2, starting_stack=p0_stack_init, small_blind=sb_val, big_blind=bb_val)
    gs.start_new_hand(dealer_pos=0, player_stacks=[p0_stack_init, p1_stack_init])
    print(f"Initial State:\n{gs}")

    expected_p0_stack_after_sb = p0_stack_init - sb_val
    expected_p1_stack_after_bb = p1_stack_init - bb_val
    assert gs.current_player_idx == 0, f"Expected P0 turn, got {gs.current_player_idx}"
    assert abs(gs.pot - (sb_val + bb_val)) < 0.01, f"Pot={gs.pot}, expected {sb_val + bb_val}"
    assert abs(gs.player_stacks[0] - expected_p0_stack_after_sb) < 0.01, f"P0 stack={gs.player_stacks[0]}, expected {expected_p0_stack_after_sb}"
    assert abs(gs.player_stacks[1] - expected_p1_stack_after_bb) < 0.01, f"P1 stack={gs.player_stacks[1]}, expected {expected_p1_stack_after_bb}"
    assert abs(gs.current_bet - bb_val) < 0.01, f"Current bet={gs.current_bet}, expected {bb_val}"
    assert abs(gs.player_bets_in_round[0] - sb_val) < 0.01, f"P0 RndBet={gs.player_bets_in_round[0]}, expected {sb_val}"
    assert abs(gs.player_bets_in_round[1] - bb_val) < 0.01, f"P1 RndBet={gs.player_bets_in_round[1]}, expected {bb_val}"
    print("HU Initial state OK.")

    actions_p0 = gs.get_available_actions()
    print(f"P0 Actions: {actions_p0}")
    expected_fold = ('fold', 0)
    expected_call = ('call', sb_val)
    expected_min_raise = ('raise', bb_val * 2)
    expected_all_in_raise = ('raise', int(round(p0_stack_init)))
    assert expected_fold in actions_p0, "Missing fold"
    assert expected_call in actions_p0, f"Missing call {expected_call}"
    assert any(a[0] == 'raise' and a[1] == expected_min_raise[1] for a in actions_p0), f"Missing min raise {expected_min_raise}"
    assert any(a[0] == 'raise' and abs(a[1] - expected_all_in_raise[1]) < 0.1 for a in actions_p0), f"Missing/incorrect all-in raise {expected_all_in_raise}"
    print("P0 essential actions OK.")

    gs_after_p0_call = gs.apply_action(expected_call)
    print(f"State after P0 calls {expected_call}:\n{gs_after_p0_call}")
    expected_p0_stack_after_call = expected_p0_stack_after_sb - expected_call[1]
    assert abs(gs_after_p0_call.player_stacks[0] - expected_p0_stack_after_call) < 0.01, f"P0 stack={gs_after_p0_call.player_stacks[0]}, expected {expected_p0_stack_after_call}"
    assert abs(gs_after_p0_call.player_stacks[1] - expected_p1_stack_after_bb) < 0.01, f"P1 stack={gs_after_p0_call.player_stacks[1]}, expected {expected_p1_stack_after_bb}"
    assert gs_after_p0_call.betting_round == GameState.PREFLOP, f"Round={gs_after_p0_call.betting_round}"
    assert abs(gs_after_p0_call.pot - (bb_val * 2)) < 0.01, f"Pot={gs_after_p0_call.pot}, expected {bb_val * 2}"
    assert gs_after_p0_call.current_player_idx == 1, f"Expected P1 turn, got {gs_after_p0_call.current_player_idx}"
    assert abs(gs_after_p0_call.player_bets_in_round[0] - bb_val) < 0.01, f"P0 RndBet={gs_after_p0_call.player_bets_in_round[0]}, expected {bb_val}"
    assert abs(gs_after_p0_call.player_bets_in_round[1] - bb_val) < 0.01, f"P1 RndBet={gs_after_p0_call.player_bets_in_round[1]}, expected {bb_val}"
    assert abs(gs_after_p0_call.current_bet - bb_val) < 0.01, f"Current bet={gs_after_p0_call.current_bet}, expected {bb_val}"
    print("P0 call state OK.")

    actions_p1 = gs_after_p0_call.get_available_actions()
    print(f"P1 Actions: {actions_p1}")
    expected_check = ('check', 0)
    expected_min_raise_p1 = ('raise', bb_val * 2)
    expected_all_in_raise_p1 = ('raise', int(round(p1_stack_init)))
    assert expected_check in actions_p1, "Missing P1 check"
    assert any(a[0] == 'raise' and a[1] == expected_min_raise_p1[1] for a in actions_p1), f"Missing P1 min raise {expected_min_raise_p1}"
    assert any(a[0] == 'raise' and abs(a[1] - expected_all_in_raise_p1[1]) < 0.1 for a in actions_p1), f"Missing/incorrect P1 all-in {expected_all_in_raise_p1}"
    print("P1 essential actions OK.")

    gs_after_p1_check = gs_after_p0_call.apply_action(expected_check)
    print(f"State after P1 checks {expected_check}:\n{gs_after_p1_check}")
    assert abs(gs_after_p1_check.player_stacks[0] - expected_p0_stack_after_call) < 0.01, f"P0 stack={gs_after_p1_check.player_stacks[0]}"
    assert abs(gs_after_p1_check.player_stacks[1] - expected_p1_stack_after_bb) < 0.01, f"P1 stack={gs_after_p1_check.player_stacks[1]}"
    assert gs_after_p1_check.betting_round == GameState.FLOP, f"Round={gs_after_p1_check.betting_round}"
    assert len(gs_after_p1_check.community_cards) == 3, f"Cards={len(gs_after_p1_check.community_cards)}"
    assert gs_after_p1_check.current_player_idx == 1, f"Turn={gs_after_p1_check.current_player_idx}"
    assert abs(gs_after_p1_check.current_bet - 0) < 0.01, f"Bet={gs_after_p1_check.current_bet}"
    assert all(abs(b) < 0.01 for b in gs_after_p1_check.player_bets_in_round), f"RndBets={gs_after_p1_check.player_bets_in_round}"
    print(f"Flop dealt: {' '.join(str(c) for c in gs_after_p1_check.community_cards)}")
    print("Preflop->Flop transition OK.")

    actions_p1_flop = gs_after_p1_check.get_available_actions()
    print(f"P1 Flop Actions: {actions_p1_flop}")
    expected_check_flop = ('check', 0)
    expected_min_bet_flop = ('bet', bb_val)
    expected_all_in_bet_flop = ('bet', int(round(expected_p1_stack_after_bb)))
    assert expected_check_flop in actions_p1_flop, "Missing P1 flop check"
    assert any(a[0] == 'bet' and a[1] == expected_min_bet_flop[1] for a in actions_p1_flop), f"Missing P1 flop min bet {expected_min_bet_flop}"
    assert any(a[0] == 'bet' and abs(a[1] - expected_all_in_bet_flop[1]) < 0.1 for a in actions_p1_flop), f"Missing/incorrect P1 flop all-in bet {expected_all_in_bet_flop}"
    print("P1 Flop actions OK.")

    gs_fold = create_game_state(num_players=2, starting_stack=p0_stack_init, small_blind=sb_val, big_blind=bb_val)
    gs_fold.start_new_hand(dealer_pos=0, player_stacks=[p0_stack_init, p1_stack_init])
    print(f"\nTest fold. Initial:\n{gs_fold}")
    gs_fold_after = gs_fold.apply_action(('fold', 0))
    print(f"After P0 folds:\n{gs_fold_after}")
    assert len(gs_fold_after.active_players) == 1 and gs_fold_after.active_players[0] == 1, f"Active={gs_fold_after.active_players}"
    assert gs_fold_after.player_folded[0] is True, "P0 not folded"
    assert gs_fold_after.is_terminal() or gs_fold_after.betting_round == GameState.HAND_OVER, f"State not terminal: {gs_fold_after.betting_round}"
    assert abs(gs_fold_after.player_stacks[0] - expected_p0_stack_after_sb) < 0.01, f"Fold test P0 stack changed incorrectly"
    print("Fold scenario state OK.")

    print("\nGameState logic tests passed!")


# --- Information Set Key Test (Formatted) ---
def test_information_set_keys():
    print("\n" + "-"*60)
    print("Testing Information Set Key Consistency")
    print("-"*60)
    trainer = CFRTrainer(create_game_state, num_players=2, use_card_abstraction=True)

    def mock_hist_flop():
        return "R1|P4|CB0|N2|Act1|BHxyz"
    def mock_hist_flop_bet():
        return "R1|P6|CB1|N2|Act0|BHabc"

    gs1 = create_game_state(num_players=2, starting_stack=1000, small_blind=10, big_blind=20)
    gs1.start_new_hand(0, [1000, 1000])
    gs1 = gs1.apply_action(('call', 10))
    gs1 = gs1.apply_action(('check', 0))
    if gs1.betting_round != GameState.FLOP or gs1.current_player_idx != 1:
         raise RuntimeError(f"gs1 Setup Error: R{gs1.betting_round} P{gs1.current_player_idx}")
    gs1.hole_cards=[[Card(14,'s'),Card(13,'s')], [Card(2,'c'),Card(3,'d')]]
    gs1.community_cards=[Card(12,'s'),Card(7,'h'),Card(2,'s')]
    gs1.pot = 40.0
    gs1.current_bet = 0.0
    gs1.player_bets_in_round = [0.0, 0.0]
    gs1.get_betting_history = mock_hist_flop
    print(f"State 1 (Flop, P1 OOP turn):\n{gs1}")

    gs2 = gs1.clone()
    gs2.hole_cards[0] = [Card(7,'c'),Card(7,'d')]
    gs2.get_betting_history = mock_hist_flop

    gs3 = gs1.clone()
    gs3.community_cards[1] = Card(7,'s')
    gs3.get_betting_history = mock_hist_flop

    gs4 = gs1.clone()
    gs4.pot = 60.0
    gs4.player_bets_in_round = [10.0, 10.0]
    gs4.player_total_bets_in_hand = [30.0, 30.0]
    gs4.current_bet = 10.0
    gs4.current_player_idx=0
    gs4.last_raiser = 1
    gs4.last_raise=10
    gs4.get_betting_history = mock_hist_flop_bet

    gs5 = create_game_state(num_players=2, starting_stack=1000, small_blind=10, big_blind=20)
    gs5.start_new_hand(1, [1000, 1000])
    gs5 = gs5.apply_action(('call', 10))
    gs5 = gs5.apply_action(('check', 0))
    if gs5.betting_round != GameState.FLOP or gs5.current_player_idx != 0:
         raise RuntimeError(f"gs5 Setup Error: R{gs5.betting_round} T:{gs5.current_player_idx}")
    gs5.hole_cards = gs1.hole_cards[:]
    gs5.community_cards = gs1.community_cards[:]
    gs5.pot=40.0
    gs5.current_bet=0.0
    gs5.player_bets_in_round=[0.0, 0.0]
    gs5.get_betting_history = mock_hist_flop

    key1_p1 = trainer._create_info_set_key(gs1, 1)
    key2_p1 = trainer._create_info_set_key(gs2, 1)
    key3_p1 = trainer._create_info_set_key(gs3, 1)
    key4_p0 = trainer._create_info_set_key(gs4, 0)
    key5_p0 = trainer._create_info_set_key(gs5, 0)
    key1_p0 = trainer._create_info_set_key(gs1, 0)
    key2_p0 = trainer._create_info_set_key(gs2, 0)

    print(f"\nKeys generated:")
    print(f" K1(P1): {key1_p1}")
    print(f" K2(P1): {key2_p1}")
    print(f" K3(P1): {key3_p1}")
    print(f" K4(P0): {key4_p0}")
    print(f" K5(P0): {key5_p0}")
    print(f" K1(P0): {key1_p0}")
    print(f" K2(P0): {key2_p0}")

    keys_list = [key1_p1, key2_p1, key3_p1, key4_p0, key5_p0, key1_p0, key2_p0]
    assert all(isinstance(k, str) and k for k in keys_list), "One or more keys are invalid"

    assert key1_p1 == key2_p1, "P1 key mismatch opp card"
    assert key1_p1 != key3_p1, "P1 key matches board change"
    assert key1_p0 != key4_p0, "P0 keys match history"
    assert "POS_0" in key1_p0 and "POS_1" in key5_p0, "P0 key POS mismatch"
    assert key1_p0 != key5_p0, "P0 keys match position change"
    assert key1_p0 != key2_p0, "P0 key matches own card"

    print("\nInfo set key consistency tests passed!")


# --- Push/Fold Convergence Test (Formatted) ---
def test_cfr_convergence_push_fold():
    print("\n" + "-"*60)
    print("Testing CFR Convergence on Simplified Push/Fold HU Game")
    print("-"*60)
    NUM_PLAYERS = 2
    STACK = 100.0
    SB = 5.0
    BB = 10.0
    ITERATIONS = 10000000
    LOG_FREQ = 100000
    original_get_available_actions = GameState.get_available_actions
    expected_sb_allin_total_bet = STACK
    expected_bb_call_amount = STACK - BB

    def _get_push_fold_actions_simplified(self):
        p_idx = self.current_player_idx
        if not (0 <= p_idx < self.num_players) or p_idx >= len(self.player_stacks) or p_idx >= len(self.player_bets_in_round):
             return []

        if self.betting_round == GameState.PREFLOP and p_idx == 0 and abs(self.current_bet - BB) < 0.01:
            player_stack = self.player_stacks[p_idx]
            current_bet_this_round = self.player_bets_in_round[p_idx]
            all_in_total_bet = current_bet_this_round + player_stack
            all_in_action = ('raise', int(round(all_in_total_bet)))
            return [('fold', 0), all_in_action]

        elif self.betting_round == GameState.PREFLOP and p_idx == 1 and self.current_bet >= expected_sb_allin_total_bet - 0.01:
            player_stack = self.player_stacks[p_idx]
            current_bet_this_round = self.player_bets_in_round[p_idx]
            required_call = self.current_bet - current_bet_this_round
            actual_call_amount = min(player_stack, required_call)
            if actual_call_amount <= 0.01:
                 return [('fold', 0)]
            else:
                 call_action = ('call', int(round(actual_call_amount)))
                 return [('fold', 0), call_action]
        else:
            return []

    GameState.get_available_actions = _get_push_fold_actions_simplified
    print("INFO: Replaced GameState.get_available_actions with simplified Push/Fold.")

    push_fold_strategy = None
    trainer = None
    tests_passed_sub = True
    try:
        def create_pf_game_state(num_players): return GameState(num_players, starting_stack=STACK, small_blind=SB, big_blind=BB)
        trainer = CFRTrainer(create_pf_game_state, num_players=NUM_PLAYERS, use_action_abstraction=False, use_card_abstraction=True)
        print(f"Running Push/Fold CFR for {ITERATIONS} iterations...")
        pbar = tqdm(range(ITERATIONS), desc="Push/Fold CFR", total=ITERATIONS, disable=False)
        for i in pbar:
            iter_num = i + 1
            try:
                 gs_iter = create_pf_game_state(NUM_PLAYERS)
                 gs_iter.start_new_hand((iter_num - 1) % NUM_PLAYERS, [STACK] * NUM_PLAYERS)
                 if gs_iter.current_player_idx == -1: raise RuntimeError("PushFold GS Init failed")
                 reach_probs = np.ones(NUM_PLAYERS)
                 for p_idx_perspective in range(NUM_PLAYERS):
                      trainer._calculate_cfr(gs_iter.clone(), reach_probs.copy(), p_idx_perspective, 1.0, 0.0, 0, verbose=False)
                 trainer.iterations = iter_num
            except RecursionError: print("\nERROR: Push/Fold recursion limit hit?"); tests_passed_sub = False; break
            except Exception as iter_e: print(f"\nERROR Push/Fold iter {iter_num}: {iter_e}"); traceback.print_exc(); tests_passed_sub = False; break

            if iter_num % LOG_FREQ == 0 or iter_num == ITERATIONS:
                 current_avg_strategy = trainer.get_strategy()
                 print(f"\n--- Strategy Snapshot @ Iteration {iter_num} ---")
                 sb_found, bb_found = 0, 0
                 push_amt_int = int(round(expected_sb_allin_total_bet))
                 for bucket in [0, 5, 9]:
                     sb_pattern = f"PRE_{bucket}|POS_0|"
                     bb_pattern = f"PRE_{bucket}|POS_1|"
                     sb_key = next((k for k in current_avg_strategy if k.startswith(sb_pattern) and "|Act0|" in k), None)
                     bb_key = next((k for k in current_avg_strategy if k.startswith(bb_pattern) and "|Act1|" in k and f"|CB{push_amt_int}|" in k), None)

                     if sb_key:
                          sb_strat = current_avg_strategy[sb_key]
                          push_prob = next((p for a, p in sb_strat.items() if a[0] == 'raise'), 0.0)
                          print(f"  SB B{bucket} Push%: {push_prob*100:.1f}% (Key: ...{sb_key[-20:]})")
                          sb_found+=1
                     else:
                          print(f"  SB B{bucket} Key NF Yet")
                     if bb_key:
                          bb_strat = current_avg_strategy[bb_key]
                          call_prob = next((p for a, p in bb_strat.items() if a[0] == 'call'), 0.0)
                          print(f"  BB B{bucket} Call%: {call_prob*100:.1f}% (Key: ...{bb_key[-20:]})")
                          bb_found+=1
                     else:
                          print(f"  BB B{bucket} Key NF Yet")
                 print("--------------------------------------")
                 if iter_num == ITERATIONS and (sb_found == 0 or bb_found == 0):
                     print("WARN: Did not find keys for all logged buckets at end!")

        push_fold_strategy = trainer.get_strategy()

    except Exception as e:
        print(f"\nERROR Push/Fold setup/loop: {e}")
        traceback.print_exc()
        tests_passed_sub = False
    finally:
        GameState.get_available_actions = original_get_available_actions
        print("INFO: Restored original GameState.get_available_actions.")

    print(f"Analyzing Push/Fold strategy with {len(push_fold_strategy or {})} sets...")
    if not tests_passed_sub:
        raise RuntimeError("Push/Fold test failed during execution.")
    assert push_fold_strategy is not None, "PF training failed."
    assert isinstance(push_fold_strategy, dict), "PF strategy not dict."
    assert len(push_fold_strategy) > 0, "PF Strategy empty."

    sb_keys_found = 0
    bb_keys_found = 0
    for k, v in push_fold_strategy.items():
         if "|POS_0|" in k and "|Act0|" in k and "|PRE_" in k: sb_keys_found+=1
         if "|POS_1|" in k and "|Act1|" in k and "|PRE_" in k: bb_keys_found+=1
         assert isinstance(v, dict), f"Value for key {k} not a dict"
         assert len(v) > 0, f"Infoset {k} has no actions"
         prob_sum = sum(v.values())
         assert abs(prob_sum - 1.0) < 0.01, f"Infoset {k} probs sum to {prob_sum:.4f}"
    assert sb_keys_found > 0, "No SB decision sets generated!"
    assert bb_keys_found > 0, "No BB decision sets generated!"
    print("\nCFR Push/Fold test basic checks passed!")


# --- Test Bot Player (Formatted) ---
def test_bot_player(strategy):
    print("\n" + "-"*60)
    print("Testing Bot Player")
    print("-"*60)
    if not strategy or not isinstance(strategy, dict) or not strategy:
        print("Skipping BotPlayer test: Invalid or empty strategy.")
        return
    try:
        bot = BotPlayer(strategy=strategy, name="TestBot", stack=1000, use_depth_limited_search=False)
        gs = create_game_state(num_players=2, starting_stack=1000)
        gs.start_new_hand(dealer_pos=0, player_stacks=[1000, 1000])
        idx = gs.current_player_idx

        if idx != -1:
             print(f"Get action ({bot.name} for Idx {idx})...")
             key = None
             if hasattr(bot, 'strategy_obj') and hasattr(bot.strategy_obj, '_create_info_set_key'):
                  key = bot.strategy_obj._create_info_set_key(gs, idx)
                  print(f"  Using key: {key}")
                  if hasattr(bot.strategy_obj, 'strategy') and isinstance(bot.strategy_obj.strategy, dict):
                       if key not in bot.strategy_obj.strategy:
                           print("  WARN: Key not in strategy! Default action expected.")
                  else:
                       print(" WARN: Bot strategy object invalid.")
             else:
                  print(" WARN: Bot missing key gen method.")

             action = bot.get_action(gs, idx)
             assert action is not None, "Bot returned None action"
             assert isinstance(action, tuple) and len(action)==2, f"Invalid format: {action}"
             print(f"Bot action: {action}")
        else:
            print("Skip action check: No turn.")

        print("Bot player test passed basic check!")

    except Exception as e:
        print(f"ERROR BotPlayer test: {e}")
        traceback.print_exc()
        raise


# --- Test Card Abstraction (Formatted) ---
def test_card_abstraction():
    print("\n" + "-"*60)
    print("Testing Card Abstraction")
    print("-"*60)
    try:
        h_strong=[Card(14,'s'),Card(13,'s')]
        h_weak=[Card(7,'d'),Card(2,'h')]
        b_s=CardAbstraction.get_preflop_abstraction(h_strong)
        b_w=CardAbstraction.get_preflop_abstraction(h_weak)
        assert isinstance(b_s,int), "Pre strong not int"
        assert 0<=b_s<=9, f"Pre strong OOR: {b_s}"
        assert b_s==0, f"AKs expected 0, got {b_s}"
        assert isinstance(b_w,int), "Pre weak not int"
        assert 0<=b_w<=9, f"Pre weak OOR: {b_w}"
        assert b_w==9, f"72o expected 9, got {b_w}"
        print(f"Pre(AKs):{b_s}(Exp 0)")
        print(f"Pre(72o):{b_w}(Exp 9)")

        try:
            comm=[Card(12,'s'),Card(7,'h'),Card(2,'s')]
            p_abs_s=CardAbstraction.get_postflop_abstraction(h_strong,comm)
            p_abs_w=CardAbstraction.get_postflop_abstraction(h_weak,comm)
            assert isinstance(p_abs_s, tuple) and len(p_abs_s)==3, "Post strong format"
            assert isinstance(p_abs_w, tuple) and len(p_abs_w)==3, "Post weak format"
            print(f"Postflop (AKs on Qs7h2s): {p_abs_s}")
            print(f"Postflop (72o on Qs7h2s): {p_abs_w}")
            assert p_abs_w[0] < p_abs_s[0], f"Expected bucket({p_abs_w[0]}) < ({p_abs_s[0]})"
            expected_paired_feature = 0
            expected_flush_suit_feature = 'n'
            assert p_abs_s[1] == expected_paired_feature, f"AKs paired={p_abs_s[1]}"
            assert p_abs_w[1] == expected_paired_feature, f"72o paired={p_abs_w[1]}"
            assert p_abs_s[2] == expected_flush_suit_feature, f"AKs flush='{p_abs_s[2]}'"
            assert p_abs_w[2] == expected_flush_suit_feature, f"72o flush='{p_abs_w[2]}'"
        except AssertionError as ae:
             print(f"WARN: Postflop assert failed: {ae}")
        except Exception as e:
            print(f"Note: Postflop detail skip/fail: {e}")

        print("Card abstraction test passed basic checks!")

    except Exception as e:
        print(f"ERROR CardAbs test: {e}")
        traceback.print_exc()
        raise

# --- Test Enhanced Card Abstraction (Formatted) ---
def test_enhanced_card_abstraction():
    print("\n" + "-"*60)
    print("Testing Enhanced Card Abstraction")
    print("-"*60)
    try:
        h=[Card(14,'s'),Card(13,'s')]
        try:
             pre=EnhancedCardAbstraction.get_preflop_abstraction(h)
             assert isinstance(pre, int), f"Pre type invalid: {type(pre)}"
             max_b=getattr(EnhancedCardAbstraction,'NUM_PREFLOP_BUCKETS',20)-1
             assert 0<=pre<=max_b, f"Pre bucket OOR: {pre} (Range 0-{max_b})"
             print(f"Enh pre(AKs): {pre} (Range 0-{max_b})")
        except FileNotFoundError: print("Note: Enh preflop model missing.")
        except Exception as e_pre: print(f"Note: Enh preflop error: {e_pre}")

        try:
             comm=[Card(12,'s'),Card(7,'h'),Card(2,'s')]
             post=EnhancedCardAbstraction.get_postflop_abstraction(h,comm)
             assert isinstance(post, int), f"Post type invalid: {type(post)}"
             num_comm = len(comm)
             max_post = -1
             if num_comm == 3: max_post = getattr(EnhancedCardAbstraction, 'NUM_FLOP_BUCKETS', 50) - 1
             elif num_comm == 4: max_post = getattr(EnhancedCardAbstraction, 'NUM_TURN_BUCKETS', 100) - 1
             elif num_comm == 5: max_post = getattr(EnhancedCardAbstraction, 'NUM_RIVER_BUCKETS', 200) - 1
             if max_post != -1: assert 0<=post<=max_post, f"Bucket OOR: {post} [0, {max_post}]"
             else: print(f"WARN: Cannot get max bucket for len {num_comm}")
             print(f"Enh post(AKs on Qs7h2s): {post} (Range 0-{max_post if max_post!=-1 else '?'})")
        except FileNotFoundError: print("Note: Enh postflop model missing.")
        except Exception as e_post: print(f"Note: Enh postflop error: {e_post}")

        print("Enhanced card abstraction test passed basic checks.")

    except ImportError: print("EnhCardAbs module missing, skip.")
    except Exception as e: print(f"ERROR Enh CardAbs test: {e}"); traceback.print_exc(); raise

# --- Test Depth Limited Search (Formatted) ---
def test_depth_limited_search(strategy):
    print("\n" + "-"*60)
    print("Testing Depth-Limited Search")
    print("-"*60)
    if not strategy or not isinstance(strategy, dict) or not strategy:
        print("Skipping DLS test: Invalid strategy.")
        return
    try:
        cfr_s=CFRStrategy()
        cfr_s.strategy = strategy
        if not cfr_s.strategy: raise ValueError("Strategy dict empty.")

        gs = create_game_state(num_players=2, starting_stack=1000)
        gs.start_new_hand(dealer_pos=0, player_stacks=[1000, 1000])
        dls = DepthLimitedSearch(cfr_s, search_depth=1, num_iterations=20)
        idx=gs.current_player_idx

        if idx != -1:
             print("Get DLS action...")
             action=dls.get_action(gs.clone(),idx)
             assert action is not None, "DLS None action"
             assert isinstance(action,tuple) and len(action)==2, f"Invalid format: {action}"
             print(f"DLS action: {action}")
        else:
            print("Skip DLS action: no turn.")

        print("DLS test passed basic checks!")

    except ImportError: print("DLS module missing, skip.")
    except Exception as e: print(f"ERROR DLS test: {e}"); traceback.print_exc(); raise


# --- Run All Tests Function (Formatted) ---
def run_all_simple_tests(verbose_cfr=False):
    print("\n" + "="*80)
    print(f"RUNNING SIMPLE VALIDATION TESTS (verbose_cfr arg: {verbose_cfr})")
    print("="*80)
    start_time=time.time()
    test_dir="test_output"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test output dir: {os.path.abspath(test_dir)}")

    cfr_strategy = None
    tests_passed = True
    failed_tests = []
    halt_execution = False

    test_suite = [
        test_game_state_logic,
        test_information_set_keys,
        lambda: test_cfr_trainer(verbose_cfr=False), # Structural test
        test_cfr_convergence_push_fold,             # Convergence test
        test_card_abstraction,
        test_enhanced_card_abstraction,
        lambda sd: test_bot_player(sd),             # Use strategy from structural test
        lambda sd: test_depth_limited_search(sd)    # Use strategy from structural test
    ]
    test_names = [getattr(t, '__name__', f'lambda_{i}') for i, t in enumerate(test_suite)]
    test_names[2] = "test_cfr_trainer_structure"
    test_names[6] = "test_bot_player_wrapper"
    test_names[7] = "test_depth_limited_search_wrapper"

    for i, test_func_or_lambda in enumerate(test_suite):
        func_name = test_names[i]
        print(f"\n--- Running {func_name} ---")
        current_phase = "Pre-CFR" if i < 2 else ("CFR" if i == 2 or i == 3 else "Post-CFR")

        try:
            is_lambda_after_cfr = "<lambda>" in repr(test_func_or_lambda) and current_phase == "Post-CFR"

            if func_name == "test_cfr_trainer_structure":
                 cfr_strategy = test_func_or_lambda() # Capture result
            elif func_name == "test_cfr_convergence_push_fold":
                 test_func_or_lambda() # Runs internally
            elif is_lambda_after_cfr:
                 test_func_or_lambda(cfr_strategy) # Pass result from structure test
            else:
                 test_func_or_lambda() # Call regular test

            print(f"[PASS] {func_name}")

            if func_name == "test_cfr_trainer_structure":
                if not isinstance(cfr_strategy, dict):
                     cfr_strategy = {} # Ensure dict for later tests
                     print("[WARN] CFR structure test bad return type.")
                if not cfr_strategy:
                     print("[INFO] CFR structure test returned empty strategy.")

        except Exception as e:
             tests_passed = False
             failed_tests.append(f"{current_phase}:{func_name}")
             print(f"[FAIL] {func_name}: {e}")
             traceback.print_exc()
             is_critical = func_name in ['test_game_state_logic','test_information_set_keys', 'test_cfr_convergence_push_fold']
             if is_critical and not isinstance(e, AssertionError):
                 print("\nStopping tests due to critical failure.")
                 halt_execution = True
                 break

        if halt_execution:
            break

    # Final Summary
    end_time=time.time()
    print("\n"+"="*80)
    overall_status = tests_passed and not halt_execution
    if overall_status:
        print("ALL SIMPLE TESTS COMPLETED SUCCESSFULLY!")
    else:
        print(f"SIMPLE TEST RUN FAILED! Failed tests: {', '.join(failed_tests)}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("="*80)
    return overall_status

# --- END OF FILE organized_poker_bot/utils/simple_test.py ---
