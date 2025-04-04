# --- START OF FILE main.py ---

#!/usr/bin/env python3
"""
Main entry point for the poker bot application.
Provides command-line interface for training, playing against, and evaluating the bot.
(V18: Correct SyntaxError in evaluate_bot)
"""

import os
import sys
import argparse
import pickle
from tqdm import tqdm
import traceback # For printing stack traces

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with absolute imports
try:
    from organized_poker_bot.cfr.cfr_trainer import CFRTrainer
    from organized_poker_bot.training.optimized_self_play_trainer import OptimizedSelfPlayTrainer
    from organized_poker_bot.game_engine.game_state import GameState
    from organized_poker_bot.game_engine.poker_game import PokerGame
    from organized_poker_bot.game_engine.player import Player
    from organized_poker_bot.bot.bot_player import BotPlayer
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.bot.bot_evaluator import BotEvaluator
    from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction
except ImportError as e:
    print(f"Error importing core modules in main.py: {e}")
    print("Ensure all submodules exist and PYTHONPATH is correct.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Poker Bot CLI')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'play', 'evaluate', 'test'],
                        help='Mode to run: train, play, evaluate, or test')

    # Training args
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--output_dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--num_players', type=int, default=6, help='Number of players (train/play bot vs bot)')
    parser.add_argument('--optimized', action='store_true', help='Use optimized parallel training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for optimized training')

    # Play args
    parser.add_argument('--strategy', type=str, help='Path to strategy file (.pkl)')
    parser.add_argument('--opponent', type=str, default='human', choices=['human', 'random', 'bot'], help='Opponent type')
    parser.add_argument('--num_opponents', type=int, default=5, help='Number of opponents (if opponent=human/random)')
    parser.add_argument('--small_blind', type=int, default=50, help='Small blind')
    parser.add_argument('--big_blind', type=int, default=100, help='Big blind')
    parser.add_argument('--use_dls', action='store_true', help='Enable Depth-Limited Search')
    parser.add_argument('--search_depth', type=int, default=2, help='DLS search depth')

    # Evaluate args
    parser.add_argument('--num_games', type=int, default=100, help='Number of evaluation games')

    # Test args
    parser.add_argument('--verbose_test', action='store_true', help='Enable verbose test output')

    return parser.parse_args()


# Game state factory needs to be picklable for multiprocessing if used
def create_game_state_func(num_players):
    return GameState(num_players) # Add blinds/stack if GameState requires


def train_bot(args):
    """Train the poker bot."""
    print(f"Training bot with {args.iterations} iterations...")
    os.makedirs(args.output_dir, exist_ok=True)

    def create_game_state_for_training(num_p):
        return GameState(num_p, small_blind=args.small_blind, big_blind=args.big_blind)

    trainer = None
    if args.optimized:
        print(f"Using optimized self-play with {args.num_workers} workers")
        try:
            trainer = OptimizedSelfPlayTrainer( game_state_class=create_game_state_for_training, num_players=args.num_players, num_workers=args.num_workers )
        except Exception as e: print(f"Error init OptimizedTrainer: {e}"); return None
    else:
        print("Using standard CFR training")
        num_cfr = args.num_players
        if num_cfr != 2: print(f"Warning: Std CFR often assumes 2 players. Training w/ {num_cfr}.")
        try: trainer = CFRTrainer( game_state_class=create_game_state_for_training, num_players=num_cfr )
        except Exception as e: print(f"Error init CFRTrainer: {e}"); return None
    if trainer is None: print("Failed init trainer."); return None

    train_args = { 'iterations': args.iterations, 'checkpoint_freq': args.checkpoint_freq, 'output_dir': args.output_dir }
    try: strategy = trainer.train(**train_args)
    except Exception as e: print(f"Error during training: {e}"); traceback.print_exc(); return None
    if not strategy: print("Train returned no strategy."); return None

    final_path = os.path.join(args.output_dir, "final_strategy.pkl")
    save_strat = strategy
    if hasattr(strategy, 'strategy') and isinstance(strategy.strategy, dict): save_strat = strategy.strategy
    try:
        with open(final_path, 'wb') as f: pickle.dump(save_strat, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Training complete! Final strategy saved: {final_path}")
    except Exception as e: print(f"Error save final strategy: {e}")
    return strategy


def play_against_bot(args):
    """Play against the poker bot."""
    if not args.strategy: print("Error: --strategy required"); return
    if not os.path.exists(args.strategy): print(f"Error: Strategy {args.strategy} not found"); return

    print(f"Loading strategy: {args.strategy}...")
    strat_obj = CFRStrategy()
    try:
        strat_obj.load(args.strategy)
    except Exception as e:
        print(f"Error loading strategy: {e}")
        traceback.print_exc()
        return
    if not strat_obj.strategy: print(f"Error: Strategy empty after load"); return

    bot = BotPlayer( strategy=strat_obj, name="PokerBot", stack=10000, use_depth_limited_search=args.use_dls, search_depth=args.search_depth )
    players = []; num_total = 0
    if args.opponent == 'human':
        players.append(Player("Human", is_human=True))
        for i in range(args.num_opponents): players.append(BotPlayer(strategy=strat_obj, name=f"Bot-{i+1}", use_depth_limited_search=args.use_dls, search_depth=args.search_depth))
        num_total = 1 + args.num_opponents
    elif args.opponent == 'random':
        players.append(Player("Human", is_human=True))
        for i in range(args.num_opponents): players.append(Player(f"Random-{i+1}", is_random=True))
        num_total = 1 + args.num_opponents
    elif args.opponent == 'bot':
        if args.num_players < 2: print("Error: Bot vs Bot needs --num_players >= 2"); return
        for i in range(args.num_players): players.append(BotPlayer(strategy=strat_obj, name=f"Bot-{i+1}", use_depth_limited_search=args.use_dls, search_depth=args.search_depth))
        num_total = args.num_players
    if not players: print("Error: No players."); return

    start_stack = 10000;
    for p in players: p.stack = start_stack

    print(f"\nStart game ({args.opponent}) w/ {num_total} players...")
    game = PokerGame( players=players, small_blind=args.small_blind, big_blind=args.big_blind, interactive=(args.opponent == 'human') )
    try: game.run()
    except Exception as e: print(f"\nError game run: {e}"); traceback.print_exc()


def evaluate_bot(args):
    """Evaluate the poker bot."""
    if not args.strategy: print("Error: --strategy required"); return
    if not os.path.exists(args.strategy): print(f"Error: Strategy {args.strategy} not found"); return

    print(f"Load strategy {args.strategy} for eval...")
    # --- CORRECTED STRATEGY LOAD ---
    strat_obj = CFRStrategy() # Initialize the strategy object
    try:
        strat_obj.load(args.strategy) # Load the strategy from file
    except Exception as e:
        print(f"Error loading strategy for evaluation: {e}")
        traceback.print_exc() # Show details
        return # Exit if loading fails
    # --- END CORRECTION ---

    if not strat_obj.strategy: print(f"Error: Strategy empty after load"); return

    bot_eval = BotPlayer( strategy=strat_obj, name="EvalBot", use_depth_limited_search=args.use_dls, search_depth=args.search_depth )
    evaluator = BotEvaluator()
    print(f"\nEval vs {args.num_opponents} random over {args.num_games} games...")
    try:
        results_rnd = evaluator.evaluate_against_random( bot=bot_eval, num_games=args.num_games, num_opponents=args.num_opponents )
        if results_rnd:
             print(f"\nResults vs Random:\n WinRate:{results_rnd.get('win_rate',0):.3f} | AvgProfit:{results_rnd.get('avg_profit',0):.2f} | TotProfit:{results_rnd.get('total_profit',0):.2f} | Games:{results_rnd.get('games_played',0)}")
        else: print("Eval vs Random failed.")
    except Exception as e: print(f"Error eval vs random: {e}"); traceback.print_exc()
    print("\nMeasure exploitability (heuristic)...")
    try:
        exploit = evaluator.measure_exploitability(strat_obj.strategy)
        print(f"  Exploit Score: {exploit:.4f} (Lower is better)")
    except Exception as e: print(f"Error exploit measure: {e}"); traceback.print_exc()


def run_tests(verbose=False):
    """Run tests to verify the poker bot implementation."""
    print("Running tests to verify poker bot implementation...")
    try:
        from organized_poker_bot.utils.simple_test import run_all_simple_tests
        tests_passed = run_all_simple_tests(verbose_cfr=verbose)
        return tests_passed
    except ImportError as e:
        print(f"ERROR: Could not import 'run_all_simple_tests' from simple_test.py: {e}")
        return False
    except Exception as e:
         print(f"ERROR running simple tests: {e}")
         traceback.print_exc()
         return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Poker Bot CLI')
    # Add args using a loop or keep explicit calls
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'play', 'evaluate', 'test'], help='Mode to run')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--output_dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--num_players', type=int, default=6, help='Number of players')
    parser.add_argument('--optimized', action='store_true', help='Use optimized parallel training')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers for parallel train')
    parser.add_argument('--strategy', type=str, help='Path to strategy file (.pkl)')
    parser.add_argument('--opponent', type=str, default='human', choices=['human', 'random', 'bot'], help='Opponent type for play')
    parser.add_argument('--num_opponents', type=int, default=5, help='Num opponents (if opponent=human/random)')
    parser.add_argument('--small_blind', type=int, default=50, help='Small blind')
    parser.add_argument('--big_blind', type=int, default=100, help='Big blind')
    parser.add_argument('--use_dls', action='store_true', help='Enable Depth-Limited Search')
    parser.add_argument('--search_depth', type=int, default=2, help='DLS search depth')
    parser.add_argument('--num_games', type=int, default=100, help='Num evaluation games')
    parser.add_argument('--verbose_test', action='store_true', help='Enable verbose test output')
    args = parser.parse_args()

    if args.mode == 'train':
        train_bot(args)
    elif args.mode == 'play':
        play_against_bot(args)
    elif args.mode == 'evaluate':
        evaluate_bot(args)
    elif args.mode == 'test':
        tests_passed = run_tests(verbose=args.verbose_test)
        if not tests_passed:
            print("\nOne or more tests failed.")
            return 1 # Exit with error code
    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help() # Show help
        return 1

    return 0 # Success

if __name__ == "__main__":
    sys.exit(main())
# --- END OF FILE main.py ---
