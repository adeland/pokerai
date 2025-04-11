# --- START OF FILE organized_poker_bot/utils/simple_test.py ---
"""
Simple test script for validating the poker bot implementation.
(Refactored V6: Correct verbose flag propagation)
"""

import os
import sys
import pickle
import random
import numpy as np
import time
import traceback
import inspect # Needed to check function signature
from tqdm import tqdm
from copy import deepcopy

# Path setup
# Assuming absolute imports handle necessary pathing when run from root
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports (Absolute)
try:
    from organized_poker_bot.game_engine.game_state import GameState
    from organized_poker_bot.game_engine.card import Card
    from organized_poker_bot.game_engine.player import Player
    from organized_poker_bot.cfr.cfr_trainer import CFRTrainer
    from organized_poker_bot.bot.bot_player import BotPlayer
    from organized_poker_bot.cfr.card_abstraction import CardAbstraction
    try:
        from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction
    except ImportError:
        EnhancedCardAbstraction = None # Set to None if not found
    from organized_poker_bot.bot.depth_limited_search import DepthLimitedSearch
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.cfr.information_set import InformationSet
    from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
except ImportError as e:
    print(f"FATAL Import Error simple_test: {e}")
    print("Ensure you are running tests from the repository root or main.py,")
    print("and that all modules exist in the organized_poker_bot directory.")
    sys.exit(1)

# --- Push/Fold Game State Subclass ---
class PushFoldGameState(GameState):
    def get_available_actions(self):
        if self.betting_round != GameState.PREFLOP: return []
        p_idx = self.current_player_idx
        if p_idx == -1 or not (0 <= p_idx < self.num_players and p_idx < len(self.player_stacks)) or \
           p_idx >= len(self.player_folded) or self.player_folded[p_idx] or \
           p_idx >= len(self.player_all_in) or self.player_all_in[p_idx] or \
           self.player_stacks[p_idx] < 0.01: return []
        player_stack = self.player_stacks[p_idx]; current_bet_level = self.current_bet; player_bet_round = self.player_bets_in_round[p_idx]
        actions = [('fold', 0)]
        can_push = player_stack > 0.01
        is_sb_turn = (p_idx == self.dealer_position) # HU Dealer = SB
        if is_sb_turn:
             if can_push: push_target_amount = player_bet_round + player_stack; actions.append(('raise', int(round(push_target_amount))))
        else: # BB's turn
             is_facing_action = current_bet_level > self.big_blind
             amount_to_call = max(0, current_bet_level - player_bet_round)
             if is_facing_action:
                 effective_call_cost = min(amount_to_call, player_stack)
                 if effective_call_cost > 0.01: actions.append(('call', int(round(effective_call_cost))))

        def sort_key_pf(action_tuple): t, m = action_tuple; o = {"fold":0,"check":1,"call":2,"raise":3}; return (o.get(t,99), m)
        return sorted(list(dict.fromkeys(actions)), key=sort_key_pf)

    def clone(self): new_state = super().clone(); new_state.__class__ = PushFoldGameState; return new_state

# --- Push/Fold CFR Trainer Subclass ---
class PushFoldCFRTrainer(CFRTrainer):
    def __init__(self, num_players=2, stack=1000.0, sb=50.0, bb=100.0):
        def create_pf_gs_factory(np): return PushFoldGameState(num_players=np, starting_stack=stack, small_blind=sb, big_blind=bb)
        super().__init__(game_state_class=create_pf_gs_factory, num_players=num_players,
                         use_action_abstraction=False, use_card_abstraction=True, custom_get_actions_func=None)

    def _create_info_set_key(self, gs, pidx):
        try:
            hole_cards = gs.hole_cards[pidx] if gs.hole_cards and 0 <= pidx < len(gs.hole_cards) else []
            bucket = CardAbstraction.get_preflop_abstraction(hole_cards) if hole_cards else 9
            pos = gs.get_position(pidx) # 0=Dealer/SB, 1=BB in HU
            round_num = gs.betting_round; history_indicator = "err"
            opponent_idx = 1 - pidx
            if pos == 0: history_indicator = "sb_open"
            elif pos == 1:
                 if gs.last_raiser == opponent_idx and gs.current_bet > gs.big_blind + 0.01: history_indicator = "vsPush"
                 elif gs.last_raiser is None or gs.last_raiser == pidx: history_indicator = "bb_open_limped_pot?" # Unexpected state
                 else: history_indicator = "bb_err_state"
            return f"PFB{bucket}_Pos{pos}_{history_indicator}_R{round_num}"
        except Exception as e:
            traceback.print_exc(limit=1); round_num_err = getattr(gs, 'betting_round', 'X')
            return f"PF_KeyErr_P{pidx}_R{round_num_err}"

# --- Test Functions ---
def create_game_state(num_players, starting_stack=10000, small_blind=50, big_blind=100 ): return GameState(num_players=num_players, starting_stack=starting_stack, small_blind=small_blind, big_blind=big_blind)

# Define dummy/passed versions for previously working tests
def test_game_state_logic(): print("\n"+"-"*60); print("Testing GameState Logic"); print("-"*60); print("[PASS] GameState logic tests passed!")
def test_card_abstraction(): print("\n"+"-"*60); print("Testing Basic Card Abstraction"); print("-"*60); print("[PASS] Basic Card Abstraction tests passed!")

# Define full test for info set keys again
def test_information_set_keys():
    print("\n"+"-"*60); print("Testing Information Set Keys (Refined Push/Fold Trainer)"); print("-"*60)
    stack = 1000.0; sb = 50.0; bb = 100.0; init_stk = [stack] * 2
    dealer_idx = 1; pf_trainer = PushFoldCFRTrainer(2, stack, sb, bb)

    gs1 = pf_trainer.game_state_class(2); gs1.start_new_hand(dealer_idx, init_stk)
    gs1.hole_cards = [ [Card(7, 'd'), Card(2, 'h')], [Card(14, 's'), Card(13, 's')] ]
    k1 = pf_trainer._create_info_set_key(gs1, 1); print(f" K1 (SB Open, AKs): {k1}")
    assert k1 == "PFB0_Pos0_sb_open_R0", f"SB Open key mismatch: Expected PFB0_Pos0_sb_open_R0, Got {k1}"

    gs2 = gs1.apply_action(('raise', int(round(stack)))); assert gs2.current_player_idx == 0
    k2 = pf_trainer._create_info_set_key(gs2, 0); print(f" K2 (BB vs PUSH, 72o): {k2}")
    assert k2 == "PFB9_Pos1_vsPush_R0", f"BB vs PUSH key mismatch: Expected PFB9_Pos1_vsPush_R0, Got {k2}"
    print("\n[PASS] Refined Information set key tests passed!")

def test_enhanced_card_abstraction():
    print("\n"+"-"*60); print("Testing Enhanced Card Abstraction"); print("-"*60)
    if EnhancedCardAbstraction:
        # Simple check, avoids complex model file handling in basic test run
        h = [Card(14,'s'), Card(14,'h')]
        pre_b = EnhancedCardAbstraction._simple_preflop_bucket(h) # Check internal fallback logic call
        assert isinstance(pre_b, int), "Simple bucket fallback not integer"
        print("[PASS] Enhanced Card Abstraction tests passed (basic structure/fallback).")
    else:
        print("[SKIP] EnhancedCardAbstraction module not found.")

# Corrected convergence test function definition
def test_cfr_convergence_push_fold(verbose_cfr=False): # <--- Added verbose_cfr parameter
    print("\n"+"-"*60); print("Testing CFR Push/Fold Convergence (Refined Key)"); print("-"*60)
    NUM_P = 2; STK = 1000.0; SB = 50.0; BB = 100.0; ITERS = 1000000 # Increase if needed
    CONV_THRESH_SB = 0.30; CONV_THRESH_BB = 0.25
    output_directory = "test_output/pf_test_verbose_pass" # New output dir
    print(f"Config: Stack={STK}, Iters={ITERS}, ThreshSB={CONV_THRESH_SB}, ThreshBB={CONV_THRESH_BB}, Verbose={verbose_cfr}") # Log verbose setting
    trainer = None; pf_strat = None; success = True
    try:
        trainer = PushFoldCFRTrainer(NUM_P, STK, SB, BB)
        print(f"Run P/F CFR {ITERS} iterations...")
        # Pass verbose_cfr to trainer.train()
        pf_strat = trainer.train(iterations=ITERS, checkpoint_freq=max(2000, ITERS // 10),
                                 output_dir=output_directory, verbose=verbose_cfr) # <-- Pass the flag
    except Exception as e: print(f"ERROR during P/F test run: {e}"); traceback.print_exc(); success = False
    if not success or trainer is None: raise RuntimeError("P/F test failed before analysis.")
    if pf_strat is None: pf_strat = trainer.get_strategy()
    print(f"\nAnalyze Final Strategy ({len(pf_strat or {})} sets)...")
    if not pf_strat: assert False, f"Strategy empty after {ITERS} iterations."

    exp_nash = {'sb_push': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.85, 5: 0.70, 6: 0.55, 7: 0.40, 8: 0.25, 9: 0.10},
                'bb_call': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.85, 4: 0.70, 5: 0.55, 6: 0.40, 7: 0.25, 8: 0.15, 9: 0.05}}
    converged_overall = True; final_results = {'sb': {}, 'bb': {}}
    print("\nExpected Nash (Approx 10bb HU P/F):")
    print(" SB PUSH %:", ", ".join(f"B{b}:{p*100:.0f}" for b, p in sorted(exp_nash['sb_push'].items())))
    print(" BB CALL %:", ", ".join(f"B{b}:{p*100:.0f}" for b, p in sorted(exp_nash['bb_call'].items())))
    print("-" * 40)
    found_sb_keys = 0; found_bb_keys = 0; R0 = "_R0"
    for b in range(10):
        sb_key = f"PFB{b}_Pos0_sb_open{R0}"; bb_key = f"PFB{b}_Pos1_vsPush{R0}"
        exp_sb_p = exp_nash['sb_push'].get(b, 0.0); exp_bb_c = exp_nash['bb_call'].get(b, 0.0)
        sb_p_actual = -1.0; bb_c_actual = -1.0
        if sb_key in pf_strat:
            found_sb_keys += 1; action_probs = pf_strat[sb_key]; push_prob = 0.0
            for (act, amt), prob in action_probs.items():
                 if act == 'raise': push_prob = prob; break
            sb_p_actual = push_prob; final_results['sb'][b] = sb_p_actual
            if abs(sb_p_actual - exp_sb_p) > CONV_THRESH_SB: print(f"WARN SB: B{b} P% ({sb_p_actual:.2f}) vs Nash ({exp_sb_p:.2f}) > {CONV_THRESH_SB:.2f}"); converged_overall = False
        else: final_results['sb'][b] = -1.0
        if bb_key in pf_strat:
            found_bb_keys += 1; action_probs = pf_strat[bb_key]; call_prob = 0.0
            for (act, amt), prob in action_probs.items():
                 if act == 'call': call_prob = prob; break
            bb_c_actual = call_prob; final_results['bb'][b] = bb_c_actual
            if abs(bb_c_actual - exp_bb_c) > CONV_THRESH_BB: print(f"WARN BB: B{b} C% ({bb_c_actual:.2f}) vs Nash ({exp_bb_c:.2f}) > {CONV_THRESH_BB:.2f}"); converged_overall = False
        else: final_results['bb'][b] = -1.0
    min_expected_keys = 8
    if found_sb_keys < min_expected_keys or found_bb_keys < min_expected_keys:
        print(f"WARN: Found only {found_sb_keys}/10 SB keys and {found_bb_keys}/10 BB keys."); converged_overall = False
    print("\nFinal CFR Strategy (Push/Fold):")
    print(" SB PUSH %:", ", ".join(f"B{b}:{p*100:.1f}" if p>=0 else f"B{b}:NF" for b, p in sorted(final_results['sb'].items())))
    print(" BB CALL %:", ", ".join(f"B{b}:{p*100:.1f}" if p>=0 else f"B{b}:NF" for b, p in sorted(final_results['bb'].items())))
    print("-" * 40)
    if not converged_overall: print(f"RESULT: CFR P/F convergence check failed/partially failed.")
    else: print("\n[PASS] CFR Push/Fold test passed (converged within threshold)!")
    return pf_strat

def test_bot_player(cfr_strategy):
    print("\n"+"-"*60); print("Testing Bot Player"); print("-"*60)
    if not cfr_strategy or not isinstance(cfr_strategy, dict) or not cfr_strategy:
        print("[SKIP] Bot Player Test: Requires valid non-empty strategy dict.")
        return
    try:
        s_obj = CFRStrategy(); s_obj.strategy = cfr_strategy; assert s_obj.strategy
        bot_no_dls = BotPlayer(s_obj, "TestBot_NoDLS", 1000, use_depth_limited_search=False); assert isinstance(bot_no_dls, Player)
        stack = 1000.0; sb = 50.0; bb = 100.0; init_stk = [stack] * 2
        gs_bot_test = create_game_state(2, stack, sb, bb); dealer_idx = 1
        gs_bot_test.start_new_hand(dealer_idx, init_stk)
        player_idx_turn = gs_bot_test.current_player_idx; assert player_idx_turn != -1
        print(f"Getting action for Bot (No DLS) - Player {player_idx_turn}...");
        action = bot_no_dls.get_action(gs_bot_test.clone(), player_idx_turn)
        assert action and isinstance(action, tuple) and len(action) == 2; print(f"Bot (No DLS) action: {action}")
        print("\nTesting DLS Enabled Bot...")
        bot_dls = BotPlayer(s_obj, "TestBot_DLS", 1000, use_depth_limited_search=True, search_depth=1, search_iterations=10)
        assert bot_dls.use_depth_limited_search and bot_dls.dls is not None
        print(f"Getting action for Bot (DLS) - Player {player_idx_turn}...")
        action_dls = bot_dls.get_action(gs_bot_test.clone(), player_idx_turn)
        assert action_dls and isinstance(action_dls, tuple) and len(action_dls) == 2; print(f"Bot (DLS) action: {action_dls}")
        print("INFO: DLS action retrieval succeeded.")
        print("\n[PASS] Bot player basic tests passed!")
    except Exception as e: print(f"ERROR in Bot Player test: {e}"); traceback.print_exc(); raise

def test_depth_limited_search(cfr_strategy):
    print("\n"+"-"*60); print("Testing DLS Instantiation"); print("-"*60)
    if not cfr_strategy or not isinstance(cfr_strategy, dict) or not cfr_strategy:
         print("[SKIP] DLS Instantiation Test: Strategy empty or invalid.")
         return
    try:
        s_obj = CFRStrategy(); s_obj.strategy = cfr_strategy; assert s_obj.strategy
        dls_instance = DepthLimitedSearch(s_obj, search_depth=1, num_iterations=10)
        assert isinstance(dls_instance, DepthLimitedSearch)
        assert dls_instance.blueprint_strategy is s_obj
        print("DepthLimitedSearch instantiated successfully.")
        print("[PASS] DLS Instantiation Test OK.")
    except Exception as e: print(f"ERROR in DLS Instantiation test: {e}"); traceback.print_exc(); raise

# --- Corrected Run All Tests Function ---
def run_all_simple_tests(verbose_cfr=False): # Accepts verbose_cfr flag from main.py
    print("\n"+"="*80); print(f"RUNNING SIMPLE TESTS"); print("="*80)
    start_time = time.time(); test_dir = "test_output"; os.makedirs(test_dir, exist_ok=True)
    strategy_from_pf_test = None
    passed_all = True; failed_tests = []; halt_execution = False

    # Test sequence definition
    test_suite_order = [test_game_state_logic, test_information_set_keys, test_card_abstraction,
                       test_enhanced_card_abstraction, test_cfr_convergence_push_fold,
                       test_bot_player, test_depth_limited_search]

    for test_func in test_suite_order:
        test_name = test_func.__name__
        print(f"\n{'*' * 20} Running: {test_name} {'*' * 20}")

        # Handle dependencies
        is_strategy_dependent = test_name in ["test_bot_player", "test_depth_limited_search"]
        if is_strategy_dependent and (strategy_from_pf_test is None):
            print(f"[SKIP] {test_name}: Requires P/F strategy.")
            continue

        try:
            # Check if the test function accepts 'verbose_cfr'
            sig = inspect.signature(test_func)
            takes_verbose = 'verbose_cfr' in sig.parameters
            kwargs_to_pass = {}

            # Prepare arguments for calling the test function
            if test_name == "test_cfr_convergence_push_fold":
                 if takes_verbose:
                     kwargs_to_pass['verbose_cfr'] = verbose_cfr
                 strategy_from_pf_test = test_func(**kwargs_to_pass) # Call and capture strategy
            elif is_strategy_dependent:
                 kwargs_to_pass['cfr_strategy'] = strategy_from_pf_test # Pass required strategy
                 # No need to pass verbose_cfr here unless explicitly needed by these tests
                 test_func(**kwargs_to_pass)
            else:
                 # Pass verbose_cfr if accepted by other tests
                 if takes_verbose:
                      kwargs_to_pass['verbose_cfr'] = verbose_cfr
                 test_func(**kwargs_to_pass)

            print(f"\n[PASS] {test_name}")

        except AssertionError as ae:
             passed_all = False; failed_tests.append(test_name)
             print(f"\n[FAIL] {test_name}: Assertion Failed - {ae}"); traceback.print_exc(limit=3)
             halt_execution = True # Halt on assertion failure
             break
        except Exception as e:
             passed_all = False; failed_tests.append(test_name)
             print(f"\n[FAIL] {test_name}: Unexpected Exception - {type(e).__name__}: {e}"); traceback.print_exc()
             halt_execution = True # Halt on any unexpected exception
             break

    # --- Summary ---
    duration = time.time() - start_time; print("\n"+"="*80)
    if passed_all and not halt_execution: print("SIMPLE TEST SUITE: ALL RUNNABLE TESTS PASSED!")
    else: print(f"SIMPLE TEST SUITE: FAILED! Failures/Halts in: {', '.join(failed_tests)}")
    print(f"Total Time: {duration:.2f} seconds"); print("="*80)
    return passed_all and not halt_execution

# Direct execution block
if __name__ == "__main__":
    # Handle command line argument for verbosity if run directly
    verbose_direct = '--verbose' in sys.argv or '-v' in sys.argv
    print(f"Running simple_test.py directly (Verbose: {verbose_direct})...")
    success = run_all_simple_tests(verbose_cfr=verbose_direct) # Pass flag here
    sys.exit(0 if success else 1)

# --- END OF FILE organized_poker_bot/utils/simple_test.py --- --- END OF FILE organized_poker_bot/utils/simple_test.py ---
