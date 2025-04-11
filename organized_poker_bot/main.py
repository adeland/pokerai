# --- START OF FILE main.py ---

#!/usr/bin/env python3
"""
Main entry point for the poker bot application.
Provides command-line interface for training, playing against, evaluating the bot,
and training abstraction models.
(V22: Use Factory Class for pickling, argument cleanup)
"""

import os
import sys
import argparse
import pickle
# import functools # No longer needed if using Factory class primarily
import traceback # For printing stack traces
import multiprocessing as mp

# --- Path Setup ---
# Add the project root to the path for consistent absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assume main.py is in the root directory relative to the 'organized_poker_bot' package folder
project_root = script_dir # The directory containing the 'organized_poker_bot' package

# Ensure the project root is in the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Absolute Imports ---
# Import core components after path setup
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
    print(f"ERROR Importing Core Modules in main.py: {e}")
    print(f"  Current sys.path: {sys.path}")
    print(f"  Detected project root: {project_root}")
    print("  -> Ensure you run this script correctly (e.g., from the project root directory)")
    print("     or ensure the 'organized_poker_bot' package is installed or in PYTHONPATH.")
    sys.exit(1)
# --- End Absolute Imports ---


# --- Factory Class for GameState Creation ---
class GameStateFactory:
    """
    A picklable factory class for creating GameState instances with specific config.
    This is generally more robust for multiprocessing than functools.partial.
    """
    def __init__(self, start_stack, small_blind, big_blind):
        self.start_stack = start_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        # Store other fixed parameters if needed

    def __call__(self, num_p):
        """
        Creates and returns a *new* GameState instance when the factory object is called.
        The 'num_p' (number of players) is provided at call time by the trainer.
        """
        # print(f"DEBUG: GameStateFactory called with num_p={num_p}") # Optional debug
        return GameState(
            num_players=num_p,
            starting_stack=self.start_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
# --- End Factory Class ---


# --- Training Function ---
def train_bot(args):
    """ Prepare and run bot training based on args. """
    print("Starting Bot Training...")
    print(f"  Mode: {'Optimized Parallel' if args.optimized else 'Standard Single-Thread'}")
    print(f"  Config: Iterations={args.iterations:,}, Players={args.num_players}, "
          f"Stack={args.start_stack:,}, Blinds={args.small_blind}/{args.big_blind}")
    abs_output_dir = os.path.abspath(args.output_dir)
    print(f"  Output Dir: {abs_output_dir}")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
         print(f"ERROR creating output directory '{args.output_dir}': {e}")
         return None

    # --- Create Factory Instance ---
    # This factory object holds the game config and is passed to the trainer.
    # It's picklable, making it suitable for multiprocessing.
    game_state_factory_instance = GameStateFactory(
        start_stack=args.start_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind
    )
    # --- End Factory Creation ---

    trainer = None
    try:
        if args.optimized:
            num_workers = args.num_workers
            # Validate/Adjust worker count
            try:
                cpu_cores = mp.cpu_count()
                if not (1 <= num_workers <= cpu_cores):
                    adjusted_workers = max(1, min(num_workers, cpu_cores))
                    print(f" WARN: num_workers ({num_workers}) invalid for cores ({cpu_cores}). Adjusted to {adjusted_workers}.")
                    num_workers = adjusted_workers
            except NotImplementedError:
                print(" WARN: Cannot determine CPU count. Using specified num_workers.")

            print(f" Using optimized trainer with {num_workers} workers.")
            trainer = OptimizedSelfPlayTrainer(
                # Pass the factory INSTANCE here
                game_state_class=game_state_factory_instance,
                num_players=args.num_players,
                num_workers=num_workers
            )
        else: # Standard Trainer
            print(" Using standard (single-thread) CFR trainer.")
            if args.num_players > 2:
                 print(f" WARN: Std CFR trainer primarily designed for 2 players. Running with {args.num_players}.")
            trainer = CFRTrainer(
                # Pass the factory INSTANCE here
                game_state_class=game_state_factory_instance,
                num_players=args.num_players,
                # Set abstraction usage (consider making these args)
                use_action_abstraction=True,
                use_card_abstraction=True # Assumes info_set_util uses this concept
            )
    except Exception as e:
        print(f"FATAL ERROR Initializing Trainer: {e}")
        traceback.print_exc()
        return None

    if trainer is None: # Should be caught by except, but check anyway
        print("FATAL ERROR: Failed to create trainer instance.")
        return None

    # --- Handle Checkpoint Loading ---
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"\nAttempting to load checkpoint: {args.resume_checkpoint}")
            loaded = trainer.load_checkpoint(args.resume_checkpoint)
            if loaded:
                # Get iteration count attribute (might be 'iteration' or 'iterations')
                current_iter = getattr(trainer, 'iteration', getattr(trainer, 'iterations', 0))
                print(f"Resuming training from iteration {current_iter + 1}")
            else:
                print(" WARN: Checkpoint loading failed. Starting fresh training.")
                # Reset iteration count on the trainer instance
                if hasattr(trainer, 'iteration'): trainer.iteration = 0
                if hasattr(trainer, 'iterations'): trainer.iterations = 0 # For standard trainer
        else:
            print(f" WARN: Checkpoint file not found: {args.resume_checkpoint}. Starting fresh.")

    # --- Prepare Training Arguments ---
    train_args = {
        'iterations': args.iterations,
        'checkpoint_freq': args.checkpoint_freq,
        'output_dir': args.output_dir,
        'verbose': args.verbose_cfr # Pass verbose flag for internal logging
    }
    # Add arguments specific to the optimized trainer if applicable
    if isinstance(trainer, OptimizedSelfPlayTrainer):
        train_args['batch_size_per_worker'] = args.batch_size_per_worker

    # --- Run Training Loop ---
    print("\n--- Starting Training Execution ---")
    final_strategy_map = None
    try:
        final_strategy_map = trainer.train(**train_args)
    except KeyboardInterrupt: # Allow graceful exit on Ctrl+C
        print("\n--- Training Interrupted by User ---")
        print(" Attempting to save current state...")
        # Fall through to finally block for saving
    except Exception as e: # Catch other runtime errors
        print(f"\nFATAL ERROR during training execution: {e}")
        traceback.print_exc()
        print(" Attempting to save current state...")
        # Fall through to finally block for saving
    finally:
        # Always try to save partial results on error or interrupt if trainer exists
        if trainer is not None and args.output_dir:
            current_strategy = None
            iteration_num = getattr(trainer, 'iteration', getattr(trainer, 'iterations', 0))
            print(f" Attempting to get partial strategy state at iteration {iteration_num}...")

            # Try to get computed strategy map first
            if hasattr(trainer, 'get_strategy') and callable(trainer.get_strategy):
                try: current_strategy = trainer.get_strategy()
                except Exception as get_e: print(f" Error calling get_strategy: {get_e}")
            if current_strategy is None and hasattr(trainer, '_compute_final_strategy') and callable(trainer._compute_final_strategy):
                 try: current_strategy = trainer._compute_final_strategy() # For OptTrainer
                 except Exception as comp_e: print(f" Error calling _compute_final_strategy: {comp_e}")

            # If strategy map retrieved, save it
            if current_strategy is not None:
                fail_filename = f"partial_strategy_iter_{iteration_num}.pkl"
                fail_path = os.path.join(args.output_dir, fail_filename)
                try:
                    with open(fail_path, 'wb') as f:
                         pickle.dump(current_strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f" Partial strategy map saved to: {fail_path}")
                except Exception as save_e:
                    print(f" ERROR saving partial strategy state: {save_e}")
            else:
                # Fallback: Save raw trainer state if strategy map unavailable
                state_to_save = None
                label = "unknown"
                if hasattr(trainer, 'information_sets'): # Standard CFRTrainer stores these
                    state_to_save = trainer.information_sets; label = "infosets"
                elif hasattr(trainer, 'regret_sum'): # OptimizedTrainer stores these
                    state_to_save = {'regret_sum': trainer.regret_sum, 'strategy_sum': trainer.strategy_sum}; label = "regrets_strategies"

                if state_to_save is not None:
                    fail_filename = f"partial_state_{label}_iter_{iteration_num}.pkl"
                    fail_path = os.path.join(args.output_dir, fail_filename)
                    try:
                        with open(fail_path, 'wb') as f:
                            pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f" Partial state ({label}) saved to: {fail_path}")
                    except Exception as save_e:
                        print(f" ERROR saving partial state ({label}): {save_e}")
                else:
                    print(" Could not retrieve any partial state to save.")

    # --- Final Status Report ---
    if final_strategy_map is not None:
        num_sets = len(final_strategy_map)
        print(f"\nTraining completed. Final strategy map generated with {num_sets:,} info sets.")
        if num_sets == 0: print(" WARN: Final strategy map is empty!")
    else:
        print("\nWARN: Training finished or interrupted without producing a final strategy map.")

    return final_strategy_map


# --- Play Function ---
def play_against_bot(args):
    """Set up and run a game against the poker bot."""
    # Validate strategy file
    if not args.strategy:
        print("Error: Bot strategy file path (--strategy) is required for play mode.")
        return
    if not os.path.exists(args.strategy):
        print(f"Error: Strategy file not found at: {args.strategy}")
        return

    print(f"Loading strategy: {args.strategy}...")
    strat_obj = CFRStrategy()
    loaded = strat_obj.load(args.strategy)
    if not loaded or not strat_obj.strategy:
        print("Error: Strategy file loaded but is empty or invalid.")
        return

    # Game setup variables
    num_total_players = 0
    player_config = [] # List of dicts describing each player type/name/instance
    start_stack = args.start_stack

    # Create the main BotPlayer instance
    main_bot = BotPlayer(
        strategy=strat_obj,
        name="PokerBot",
        stack=start_stack,
        use_depth_limited_search=args.use_dls,
        search_depth=args.search_depth
    )

    # Configure players based on opponent type
    if args.opponent == 'human':
        # Default to 1 opponent if not specified
        num_opponents = max(1, args.num_opponents if args.num_opponents is not None else 1)
        num_total_players = 1 + num_opponents
        player_config.append({"type": "human", "name": "Human"})
        player_config.append({"type": "bot", "name": main_bot.name, "instance": main_bot})
        # Add other bot opponents if needed
        for i in range(num_opponents - 1):
            player_config.append({"type": "bot", "name": f"Bot-{i+2}"})
        print(f"Config: 1 Human vs {num_opponents} Bot(s). Total players: {num_total_players}")

    elif args.opponent == 'random':
        num_opponents = max(1, args.num_opponents if args.num_opponents is not None else 1)
        num_total_players = 1 + num_opponents
        player_config.append({"type": "bot", "name": main_bot.name, "instance": main_bot})
        for i in range(num_opponents):
            player_config.append({"type": "random", "name": f"Random-{i+1}"})
        print(f"Config: 1 Bot vs {num_opponents} Random Player(s). Total players: {num_total_players}")

    elif args.opponent == 'bot':
        # Bot vs Bot mode uses num_players argument for total bots
        num_total_players = max(2, args.num_players) # Ensure at least 2 bots
        player_config.append({"type": "bot", "name": main_bot.name, "instance": main_bot})
        for i in range(num_total_players - 1):
            player_config.append({"type": "bot", "name": f"Bot-{i+2}"})
        print(f"Config: {num_total_players} Bots playing each other.")

    else:
        print(f"Error: Invalid opponent type '{args.opponent}'") # Should be caught by argparse
        return

    # Create Player/BotPlayer instances from config
    players = []
    for cfg in player_config:
        p_type = cfg["type"]
        p_name = cfg["name"]
        p_instance = cfg.get("instance") # Use if already created (main_bot)

        if p_instance:
            players.append(p_instance)
        elif p_type == "human":
            players.append(Player(name=p_name, stack=start_stack, is_human=True))
        elif p_type == "random":
            players.append(Player(name=p_name, stack=start_stack, is_random=True))
        elif p_type == "bot":
            # Create new instance for opponent bots
            players.append(BotPlayer(
                strategy=strat_obj,
                name=p_name,
                stack=start_stack,
                use_depth_limited_search=args.use_dls,
                search_depth=args.search_depth
            ))
        else:
            print(f"Error: Unknown player type '{p_type}' in config.")
            return

    # Final check
    if len(players) != num_total_players:
        print(f"Error creating players. Expected {num_total_players}, got {len(players)}.")
        return

    # --- Start Game ---
    print(f"\nStarting game with {num_total_players} players.")
    print(f" Stack: {start_stack:,.0f}, Blinds: {args.small_blind}/{args.big_blind}")
    is_interactive = (args.opponent == 'human')
    game = PokerGame(
        players=players,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        interactive=is_interactive
    )

    try:
        # Run indefinitely if interactive, otherwise for num_games
        num_hands_to_run = None if is_interactive else args.num_games
        game.run(num_hands=num_hands_to_run)
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"\nFATAL ERROR during game run: {e}")
        traceback.print_exc()


# --- Evaluate Function ---
def evaluate_bot(args):
    """Evaluate the poker bot."""
    # Validate strategy file
    if not args.strategy:
        print("Error: Bot strategy file path (--strategy) is required for evaluate mode.")
        return
    if not os.path.exists(args.strategy):
        print(f"Error: Strategy file not found at: {args.strategy}")
        return

    print(f"Loading strategy {args.strategy} for evaluation...")
    strat_obj = CFRStrategy()
    loaded = strat_obj.load(args.strategy)
    if not loaded or not strat_obj.strategy:
        print("Error: Strategy file loaded but is empty or invalid.")
        return

    # Create bot instance
    bot_to_evaluate = BotPlayer(
        strategy=strat_obj,
        name="EvalBot",
        stack=args.start_stack,
        use_depth_limited_search=args.use_dls,
        search_depth=args.search_depth
    )

    evaluator = BotEvaluator()
    # Ensure at least 1 opponent and 1 game
    num_eval_opponents = max(1, args.num_opponents if args.num_opponents is not None else 1)
    num_games_to_play = max(1, args.num_games)

    print(f"\nEvaluating Bot vs {num_eval_opponents} Random Opponent(s) over {num_games_to_play:,} games...")
    print(f" Stack: {args.start_stack:,.0f}, Blinds: {args.small_blind}/{args.big_blind}")

    try:
        results = evaluator.evaluate_against_random(
            bot=bot_to_evaluate,
            num_games=num_games_to_play,
            num_opponents=num_eval_opponents,
            starting_stack=args.start_stack,
            small_blind=args.small_blind, # Pass game params
            big_blind=args.big_blind
        )
        if results:
            # Extract and display results
            avg_profit = results.get('avg_profit_per_hand', 0)
            total_profit = results.get('total_profit', 0)
            win_rate = results.get('win_rate', 0)
            games_played = results.get('games_played', 0)
            bb_per_100 = (avg_profit / args.big_blind) * 100 if args.big_blind > 0 else 0

            print("\n--- Results vs Random ---")
            print(f" Games Played:      {games_played:,}")
            print(f" Total Profit:      {total_profit:,.2f}")
            print(f" Win Rate:          {win_rate*100:.2f}%")
            print(f" Avg Profit/Hand:   {avg_profit:.2f}")
            print(f" BB/100 Hands:      {bb_per_100:.2f}")
            print("-------------------------")
        else:
            print(" Evaluation vs Random returned no results (or failed).")
    except Exception as e:
        print(f"ERROR during evaluation vs random: {e}")
        traceback.print_exc()

    # --- Exploitability (Optional) ---
    if hasattr(evaluator, 'measure_exploitability') and callable(evaluator.measure_exploitability):
        print("\nMeasuring exploitability (heuristic, can be slow)...")
        try:
            exploit_score = evaluator.measure_exploitability(
                 strategy_map=strat_obj.strategy,
                 num_players=args.num_players, # Pass relevant params
                 start_stack=args.start_stack,
                 small_blind=args.small_blind,
                 big_blind=args.big_blind
             )
            print(f" Exploitability Score (mbb/hand): {exploit_score:.4f} (Lower is better)")
        except Exception as e:
            print(f" ERROR measuring exploitability: {e}")
            # traceback.print_exc() # Optional detail


# --- Run Tests Function ---
def run_tests(verbose=False):
    """Run verification tests."""
    print("Running tests...")
    tests_passed = False
    try:
        # Assuming tests are in a 'tests' sub-package
        from organized_poker_bot.tests.simple_test import run_all_simple_tests
        # Pass verbose flag for CFR internal logging during tests if needed
        tests_passed = run_all_simple_tests(verbose_cfr=verbose)
    except ImportError as e:
        print(f"ERROR: Could not import 'run_all_simple_tests'. Ensure tests exist at expected location: {e}")
    except Exception as e:
        print(f"ERROR running tests: {e}")
        traceback.print_exc()

    if tests_passed:
        print("--- All Simple Tests Passed ---")
    else:
        print("--- One or More Simple Tests Failed ---")
    return tests_passed


# --- Main Execution Block ---
def main():
    """Main entry point and CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Organized Poker Bot CLI (V22)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )

    # --- Modes ---
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'play', 'evaluate', 'test', 'train_abs'],
        help='Select the operating mode.'
    )

    # --- Game Parameters (Shared) ---
    game_param_group = parser.add_argument_group('Game Parameters (Used by multiple modes)')
    game_param_group.add_argument('--num_players', type=int, default=6, help='Number of players in the game (2-9). Used for train, bot-vs-bot play, evaluate.')
    game_param_group.add_argument('--start_stack', type=float, default=10000.0, help='Starting stack size for players.')
    game_param_group.add_argument('--small_blind', type=float, default=50.0, help='Small blind amount.')
    game_param_group.add_argument('--big_blind', type=float, default=100.0, help='Big blind amount.')

    # --- Training Arguments (`--mode train`) ---
    train_group = parser.add_argument_group('Training Arguments (`--mode train`)')
    train_group.add_argument('--iterations', type=int, default=10000, help='Number of training iterations to run in this session.')
    train_group.add_argument('--output_dir', type=str, default='./models/default_run', help='Directory to save trained models and checkpoints.')
    train_group.add_argument('--checkpoint_freq', type=int, default=1000, help='Save a checkpoint every N iterations.')
    train_group.add_argument('--resume_checkpoint', type=str, default=None, metavar='PATH', help='Path to .pkl checkpoint file to resume training from.')
    train_group.add_argument('--optimized', action='store_true', help='Use optimized parallel training via multiprocessing.')
    train_group.add_argument('--num_workers', type=int, default=max(1, mp.cpu_count() // 2), help='Number of worker processes for parallel training.')
    train_group.add_argument('--batch_size_per_worker', type=int, default=10, help='Hands simulated per worker per master iteration (optimized training).')
    train_group.add_argument('--verbose_cfr', action='store_true', help='Enable verbose internal logging within CFR trainers (can be very noisy).')

    # --- Play Arguments (`--mode play`) ---
    play_group = parser.add_argument_group('Play Arguments (`--mode play`)')
    play_group.add_argument('--strategy', type=str, metavar='PATH', help='Path to bot strategy file (.pkl) - REQUIRED for play/evaluate modes.')
    play_group.add_argument('--opponent', type=str, default='human', choices=['human', 'random', 'bot'], help='Opponent type when playing against the bot.')
    # Default num_opponents=None allows smarter defaults based on opponent type
    play_group.add_argument('--num_opponents', type=int, default=None, metavar='N', help='Number of NON-MAIN-BOT opponents (e.g., vs human/random, or other bots in bot-vs-bot).')
    play_group.add_argument('--num_games', type=int, default=1000, help='Number of hands to play (for non-interactive modes like bot-vs-random/bot, or evaluate).')

    # --- DLS Arguments (Used in Play/Evaluate) ---
    dls_group = parser.add_argument_group('Depth Limited Search Arguments (Optional, for BotPlayer)')
    dls_group.add_argument('--use_dls', action='store_true', help='Enable Depth Limited Search for bot decision making instead of direct strategy lookup.')
    dls_group.add_argument('--search_depth', type=int, default=2, help='Lookahead depth for DLS algorithm.')

    # --- Test Arguments (`--mode test`) ---
    test_group = parser.add_argument_group('Testing Arguments (`--mode test`)')
    test_group.add_argument('--verbose_test', action='store_true', help='Enable verbose output during test execution (shows test details).')

    # --- Abstraction Model Training Arguments (`--mode train_abs`) ---
    abs_group = parser.add_argument_group('Abstraction Model Training Arguments (`--mode train_abs`)')
    # (Currently uses defaults in EnhancedCardAbstraction.train_models, args could be added here)

    args = parser.parse_args()

    # --- Argument Validation ---
    # Validate shared parameters first
    if not (2 <= args.num_players <= 9):
        parser.error(f"num_players must be between 2 and 9, got {args.num_players}")
    if args.small_blind <= 0 or args.big_blind <= 0 or args.small_blind >= args.big_blind:
        parser.error(f"Invalid blinds: SB={args.small_blind}, BB={args.big_blind}. Must be positive and SB < BB.")
    if args.start_stack < args.big_blind * 2: # Basic check for minimum playable stack
        parser.error(f"Starting stack {args.start_stack} too small relative to big blind {args.big_blind}.")

    # Validate mode-specific requirements
    if args.mode in ['play', 'evaluate'] and not args.strategy:
         parser.error(f"--strategy is required for mode '{args.mode}'")

    # --- Mode Dispatch ---
    exit_code = 0 # Default success code
    try:
        if args.mode == 'train_abs':
            print("\n--- Training Enhanced Card Abstraction Models ---")
            # Check if class and method exist before calling
            if not EnhancedCardAbstraction or not hasattr(EnhancedCardAbstraction, 'train_models'):
                print("ERROR: EnhancedCardAbstraction or its train_models method not available.")
                exit_code = 1
            else:
                model_dir = getattr(EnhancedCardAbstraction, '_MODEL_DIR', 'models/') # Get path safely
                print(f" Models will be saved in directory: {model_dir}")
                print(">>> Calling EnhancedCardAbstraction.train_models()...")
                # Call the static training method - add args here if implemented
                returned_models = EnhancedCardAbstraction.train_models()
                print("<<< EnhancedCardAbstraction.train_models() finished.")
                if returned_models is not None:
                    print(f"    Training process completed (returned {len(returned_models)} models).")
                else:
                    # train_models might return None on failure (e.g., sklearn not installed)
                    print("    Training process returned None (check logs for errors).")
                    # Set exit code if training likely failed
                    if not hasattr(EnhancedCardAbstraction, 'SKLEARN_AVAILABLE') or not EnhancedCardAbstraction.SKLEARN_AVAILABLE:
                         exit_code = 1 # Fail if sklearn missing
                print("--- Abstraction Model Training Finished ---")

        elif args.mode == 'train':
            train_bot(args) # Let function handle internal logic and errors

        elif args.mode == 'play':
            play_against_bot(args)

        elif args.mode == 'evaluate':
            evaluate_bot(args)

        elif args.mode == 'test':
            # Pass the specific verbose flag for test output control
            tests_passed = run_tests(verbose=args.verbose_test)
            if not tests_passed:
                # Failure message printed within run_tests
                exit_code = 1 # Indicate test failure

        else:
            # This case should not be reachable due to argparse 'choices'
            print(f"Error: Unknown mode '{args.mode}'")
            parser.print_help()
            exit_code = 1

    except Exception as main_err:
        print(f"\n--- UNHANDLED ERROR in main execution ---")
        print(f" Mode: {args.mode}")
        print(f" Error: {main_err}")
        traceback.print_exc()
        exit_code = 1 # Indicate failure

    print(f"\nExiting with code: {exit_code}")
    return exit_code

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Necessary for multiprocessing freezing on Windows when creating executables
    if sys.platform.startswith('win'):
        mp.freeze_support()

    final_exit_code = main()
    sys.exit(final_exit_code)

# --- END OF FILE main.py ---
