"""
Evaluation module for measuring poker bot performance.
"""

import sys
import os
# Add parent directory to path for imports if running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pickle
from tqdm import tqdm # Ensure tqdm is imported
import random # Ensure random is imported

# Use absolute imports relative to the project structure expected by main.py
from organized_poker_bot.game_engine.poker_game import PokerGame
from organized_poker_bot.game_engine.player import Player
# Import BotPlayer if type checking or specific attributes needed
from organized_poker_bot.bot.bot_player import BotPlayer
# Import CFRStrategy if needed for analysis (used in measure_exploitability)
from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
# Hand Evaluator might be needed if doing deeper analysis or comparisons
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator


class BotEvaluator:
    """
    A class for evaluating poker bot performance.

    Provides methods for:
    - Measuring win rates against random opponents.
    - Measuring win rates against a checkpoint strategy.
    - Calculating a heuristic exploitability measure.
    - Performing basic strategy tendency analysis.
    """

    def __init__(self):
        """
        Initialize the bot evaluator.
        """
        pass # No specific initialization needed for now

    def evaluate_against_random(self, bot, num_games=100, num_opponents=5, starting_stack=10000):
        """
        Evaluate a bot against random opponents, resetting stacks each game.

        Args:
            bot (BotPlayer): Bot player object to evaluate.
            num_games (int): Number of games (hands) to simulate.
            num_opponents (int): Number of random opponents.
            starting_stack (int): The stack size each player should start with each game.

        Returns:
            dict: Evaluation results {'wins', 'total_profit', 'win_rate', 'avg_profit', 'games_played'}
        """
        if not isinstance(bot, Player):
            raise TypeError("Bot object passed must inherit from Player.")

        print(f"Evaluating bot '{getattr(bot, 'name', 'Bot')}' against {num_opponents} random opponents over {num_games} games...")

        results = {
            'wins': 0, # Games where bot profit > 0
            'total_profit': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'games_played': 0
        }

        # Create the list of players ONCE outside the loop
        all_players = [bot]
        for i in range(num_opponents):
            try:
                # Assume Player init: Player(name, stack=..., is_human=False, is_random=False)
                random_opponent = Player(name=f"Random-{i+1}", stack=starting_stack, is_human=False, is_random=True)
                all_players.append(random_opponent)
            except TypeError as e:
                 print(f"ERROR creating random Player: {e}. Check Player.__init__ signature.")
                 return results # Return empty results

        num_total_players = len(all_players)

        # Play games using tqdm progress bar
        for game_idx in tqdm(range(num_games), desc="Evaluating vs Random"):
            # --- Reset Stacks and State ---
            for p in all_players:
                try:
                    if hasattr(p, 'stack'):
                         p.stack = starting_stack
                    else:
                         print(f"Warning: Player {getattr(p, 'name', 'Unknown')} missing 'stack' during reset.")

                    if hasattr(p, 'reset_for_new_hand') and callable(p.reset_for_new_hand):
                         p.reset_for_new_hand()
                    elif hasattr(p, 'is_active'):
                         p.is_active = True if p.stack > 0 else False

                except Exception as e:
                     print(f"ERROR resetting player {getattr(p, 'name', 'Unknown')}: {e}")
            # --- End Reset ---

            # Create and run game
            try:
                game = PokerGame(
                    players=all_players,
                    small_blind=50, # Use consistent settings
                    big_blind=100,
                    interactive=False
                )
                game.run(num_hands=1) # Run one hand

                # Record results for the BOT (index 0)
                if hasattr(bot, 'stack'):
                    profit = bot.stack - starting_stack
                    results['total_profit'] += profit
                    if profit > 0: results['wins'] += 1
                else:
                     print(f"Warning: Cannot record results for bot in game {game_idx+1}.")

                results['games_played'] += 1

            except Exception as e:
                print(f"\nERROR during game simulation {game_idx+1} vs random: {e}")
                import traceback
                traceback.print_exc()
                print("Stopping evaluation due to error.")
                break # Stop

        # Calculate final statistics
        num_played = results['games_played']
        if num_played > 0:
            results['win_rate'] = results['wins'] / num_played
            results['avg_profit'] = results['total_profit'] / num_played
        else:
             print("Warning: No games were completed successfully vs random.")

        return results


    # --- NEW METHOD ---
    def evaluate_against_checkpoint(self, bot, checkpoint_strategy_path, num_games=100, starting_stack=10000):
        """
        Evaluate the current bot against a loaded checkpoint strategy (Heads-Up).

        Args:
            bot (BotPlayer): The bot player object to evaluate.
            checkpoint_strategy_path (str): Path to the .pkl file of the checkpoint strategy.
            num_games (int): Number of games (hands) to simulate.
            starting_stack (int): Starting stack for each bot.

        Returns:
            dict: Evaluation results for the main 'bot' against the checkpoint bot.
        """
        if not isinstance(bot, Player):
            raise TypeError("Bot object passed must inherit from Player.")

        print(f"Evaluating bot '{getattr(bot, 'name', 'Bot')}' against checkpoint '{os.path.basename(checkpoint_strategy_path)}' over {num_games} games...")

        # Load checkpoint strategy
        try:
            with open(checkpoint_strategy_path, 'rb') as f:
                checkpoint_strategy_dict = pickle.load(f)
            checkpoint_cfr_strategy = CFRStrategy()
            checkpoint_cfr_strategy.strategy = checkpoint_strategy_dict
        except Exception as e:
            print(f"ERROR loading checkpoint strategy from {checkpoint_strategy_path}: {e}")
            return None # Cannot proceed

        # Create checkpoint bot player
        try:
            checkpoint_bot = BotPlayer(
                strategy=checkpoint_cfr_strategy,
                name="CheckpointBot",
                stack=starting_stack,
                # Use same DLS settings? Or turn off for checkpoint? For fair compare, maybe match bot's setting.
                use_depth_limited_search=getattr(bot, 'use_depth_limited_search', False),
                search_depth=getattr(bot, 'search_depth', 1),
                search_iterations=getattr(bot, 'search_iterations', 100)
            )
        except Exception as e:
            print(f"ERROR creating checkpoint BotPlayer: {e}")
            return None

        results = {
            'wins': 0, # Games where main bot profit > 0
            'total_profit': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'games_played': 0
        }

        # Player list for Heads-Up game
        # Randomize starting position? Let's alternate for fairness.
        p1 = bot
        p2 = checkpoint_bot

        for game_idx in tqdm(range(num_games), desc="Evaluating vs Checkpoint"):
            # Alternate starting positions (dealer/SB vs BB)
            if game_idx % 2 == 0:
                 current_players = [p1, p2] # Bot is P0 (SB)
                 dealer_pos = 1 # Checkpoint is Dealer/Button
            else:
                 current_players = [p2, p1] # Checkpoint is P0 (SB)
                 dealer_pos = 1 # Bot is Dealer/Button

            # Reset stacks before each game
            for p in current_players:
                 try:
                     if hasattr(p, 'stack'):
                          p.stack = starting_stack
                     else:
                          print(f"Warning: Player {getattr(p, 'name', 'Unknown')} missing 'stack'.")

                     if hasattr(p, 'reset_for_new_hand') and callable(p.reset_for_new_hand):
                          p.reset_for_new_hand()
                     elif hasattr(p, 'is_active'):
                          p.is_active = True if p.stack > 0 else False
                 except Exception as e:
                      print(f"ERROR resetting player {getattr(p, 'name', 'Unknown')}: {e}")


            # Create and run HU game
            try:
                game = PokerGame(
                    players=current_players,
                    small_blind=50, # HU blinds often different, use standard for now
                    big_blind=100,
                    interactive=False
                )
                # Set dealer explicitly for HU alternation
                game.dealer_position = dealer_pos
                game.run(num_hands=1)

                # Record results for the MAIN bot (p1)
                if hasattr(p1, 'stack'):
                    profit = p1.stack - starting_stack
                    results['total_profit'] += profit
                    if profit > 0: results['wins'] += 1
                else:
                     print(f"Warning: Cannot record results for bot in game {game_idx+1}.")

                results['games_played'] += 1

            except Exception as e:
                print(f"\nERROR during game simulation {game_idx+1} vs checkpoint: {e}")
                import traceback
                traceback.print_exc()
                print("Stopping evaluation due to error.")
                break # Stop


        # Calculate final statistics
        num_played = results['games_played']
        if num_played > 0:
            results['win_rate'] = results['wins'] / num_played
            results['avg_profit'] = results['total_profit'] / num_played
        else:
             print("Warning: No games were completed successfully vs checkpoint.")

        return results
    # --- END NEW METHOD ---


    def measure_exploitability(self, strategy_dict):
        """
        Measure the exploitability of a strategy (simplified version).
        Lower score generally means less exploitable.

        Args:
            strategy_dict (dict): The strategy dictionary mapping info sets to action probs.

        Returns:
            float: Exploitability measure (heuristic)
        """
        if not strategy_dict:
            print("Warning: Cannot measure exploitability of empty strategy.")
            return 1.0

        deterministic_count = 0
        total_count = 0
        total_entropy = 0.0

        for info_set, action_probs in strategy_dict.items():
            if not action_probs: continue
            total_count += 1
            max_prob = max(action_probs.values()) if action_probs else 0
            if max_prob > 0.99: deterministic_count += 1

            entropy = 0.0
            num_actions = len(action_probs)
            if num_actions > 1:
                for prob in action_probs.values():
                    if prob > 0: entropy -= prob * np.log2(prob)
                max_entropy = np.log2(num_actions)
                if max_entropy > 0: total_entropy += (entropy / max_entropy)

        avg_action_diversity = total_entropy / total_count if total_count > 0 else 0
        deterministic_ratio = deterministic_count / total_count if total_count > 0 else 0
        exploitability = (deterministic_ratio * 0.7) + ((1.0 - avg_action_diversity) * 0.3)
        print(f"DEBUG Exploitability: Deterministic Ratio={deterministic_ratio:.4f}, Avg Diversity={avg_action_diversity:.4f}")
        return exploitability


    # --- NEW METHOD (Placeholder) ---
    def analyze_strategy_details(self, strategy_dict):
        """
        Perform detailed analysis of strategy tendencies (VPIP, PFR, 3Bet etc.).
        NOTE: Accurate calculation directly from abstracted strategy dict is very
              complex and often inaccurate. Simulation is preferred.
              This function provides a VERY basic, potentially misleading, estimation.
        """
        print("\n--- Detailed Strategy Analysis (Basic Estimation) ---")
        print("WARNING: Stats below are rough estimates based on static analysis")
        print("         and may not accurately reflect gameplay frequencies.")

        stats = {
            'total_info_sets': len(strategy_dict),
            'preflop_sets': 0,
            'opportunities_vpip': 0,
            'opportunities_pfr': 0,
            'opportunities_3bet': 0,
            'opportunities_f3bet': 0,
            'actions_vpip': 0.0, # Weighted probability sum
            'actions_pfr': 0.0,
            'actions_3bet': 0.0,
            'actions_fold_v_3bet': 0.0,
        }

        # This requires parsing info set keys reliably, which is hard with current format.
        # Example simplified parsing (likely needs heavy refinement based on actual key structure)
        for key, action_probs in strategy_dict.items():
            parts = key.split('|')
            is_preflop = any(p.startswith("preflop_bucket_") for p in parts) or any(p == "round_0" for p in parts)

            if not is_preflop: continue # Focus on preflop for these stats

            stats['preflop_sets'] += 1

            # --- Simplified Context Checks (HIGHLY Prone to Error) ---
            # VPIP Opportunity: Can player voluntarily put money in? (Not BB check, has call/raise option)
            # PFR Opportunity: Is this the first voluntary action preflop (no callers/raisers before)?
            # 3Bet Opportunity: Was there exactly one raise before this player's turn?
            # Fold to 3Bet Opportunity: Is player facing exactly one raise (a 3bet)?
            # Reliably determining these contexts from keys like "preflop_bucket_1|p10cb2n5|pos_3|round_0" is non-trivial.

            # Placeholder calculation (will be inaccurate):
            # Just check if call/bet/raise actions exist. This is NOT true VPIP/PFR etc.
            prob_vpip = 0.0
            prob_pfr = 0.0 # PFR is a subset of VPIP - raising when first in

            for action, prob in action_probs.items():
                action_type = action[0] if isinstance(action, tuple) else action
                if action_type in ['call', 'bet', 'raise', 'all_in']:
                     # This check is too broad for VPIP (includes BB check response?)
                     # For this placeholder, assume any non-fold/check contributes to VPIP probability
                     if action_type != 'fold' and action_type != 'check':
                          prob_vpip += prob
                if action_type in ['raise', 'bet', 'all_in']: # Bet counts as PFR if first in
                     # This is too broad for PFR (doesn't check if *first* voluntary action)
                     if action_type != 'call' and action_type != 'check' and action_type != 'fold':
                         prob_pfr += prob

            # Accumulate weighted probabilities (this doesn't give frequency %)
            stats['actions_vpip'] += prob_vpip
            stats['actions_pfr'] += prob_pfr
            # 3Bet / Fold to 3Bet require much more context from the key parsing

        # Calculate very rough averages (NOT percentages)
        avg_vpip_prob = stats['actions_vpip'] / stats['preflop_sets'] if stats['preflop_sets'] > 0 else 0
        avg_pfr_prob = stats['actions_pfr'] / stats['preflop_sets'] if stats['preflop_sets'] > 0 else 0

        print(f"Total Info Sets Analyzed: {stats['total_info_sets']}")
        print(f"Preflop Info Sets Found: {stats['preflop_sets']}")
        print(f"Avg. VPIP Action Probability (Estimate): {avg_vpip_prob:.3f}")
        print(f"Avg. PFR Action Probability (Estimate): {avg_pfr_prob:.3f}")
        print("NOTE: 3Bet / Fold-to-3Bet stats require more complex analysis or simulation.")
        print("-----------------------------------------------------")

        return stats # Return calculated stats dictionary
    # --- END NEW METHOD ---
