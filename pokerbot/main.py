#!/usr/bin/env python3
"""
Main entry point for the poker bot application.
Provides command-line interface for training, playing against, and evaluating the bot.
"""

import os
import sys
import argparse
import pickle
from tqdm import tqdm

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with absolute imports
from organized_poker_bot.cfr.cfr_trainer import CFRTrainer
from organized_poker_bot.training.optimized_self_play_trainer import OptimizedSelfPlayTrainer
from organized_poker_bot.game_engine.game_state import GameState
from organized_poker_bot.game_engine.poker_game import PokerGame
from organized_poker_bot.game_engine.player import Player
from organized_poker_bot.bot.bot_player import BotPlayer
from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
from organized_poker_bot.bot.bot_evaluator import BotEvaluator
from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Poker Bot CLI')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'play', 'evaluate', 'test'],
                        help='Mode to run: train, play, evaluate, or test')
    
    # Training arguments
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations (for train mode)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models (for train mode)')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (for train mode)')
    parser.add_argument('--num_players', type=int, default=6,
                        help='Number of players for training (for train mode)')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized self-play training (for train mode)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for optimized training (for train mode)')
    
    # Play arguments
    parser.add_argument('--strategy', type=str, 
                        help='Path to strategy file (for play and evaluate modes)')
    parser.add_argument('--opponent', type=str, default='human',
                        choices=['human', 'random', 'bot'],
                        help='Type of opponent (for play mode)')
    parser.add_argument('--num_opponents', type=int, default=5,
                        help='Number of opponents (for play mode)')
    parser.add_argument('--small_blind', type=int, default=50,
                        help='Small blind amount (for play mode)')
    parser.add_argument('--big_blind', type=int, default=100,
                        help='Big blind amount (for play mode)')
    parser.add_argument('--use_dls', action='store_true',
                        help='Use depth-limited search (for play and evaluate modes)')
    parser.add_argument('--search_depth', type=int, default=2,
                        help='Search depth for DLS (for play and evaluate modes)')
    
    # Evaluate arguments
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games for evaluation (for evaluate mode)')
    
    return parser.parse_args()

def train_bot(args):
    """Train the poker bot."""
    print(f"Training bot with {args.iterations} iterations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create game state factory
    def create_game_state(num_players):
        return GameState(num_players)
    
    # Choose trainer based on optimized flag
    if args.optimized:
        print(f"Using optimized self-play training with {args.num_workers} workers")
        trainer = OptimizedSelfPlayTrainer(
            game_state_class=create_game_state,
            num_players=args.num_players,
            num_workers=args.num_workers
        )
    else:
        print("Using standard CFR training")
        trainer = CFRTrainer(
            game_state_class=create_game_state,
            num_players=2
        )
    
    # Train the bot
    strategy = trainer.train(
        iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir
    )
    
    # Save final strategy
    final_strategy_path = os.path.join(args.output_dir, "final_strategy.pkl")
    with open(final_strategy_path, 'wb') as f:
        pickle.dump(strategy, f)
    
    print(f"Training complete! Final strategy saved to {final_strategy_path}")
    return strategy

def play_against_bot(args):
    """Play against the poker bot."""
    # Check if strategy file exists
    if not args.strategy:
        print("Error: Strategy file must be specified with --strategy")
        return
    
    if not os.path.exists(args.strategy):
        print(f"Error: Strategy file {args.strategy} not found")
        return
    
    print(f"Loading strategy from {args.strategy}...")
    
    # Load strategy
    strategy = CFRStrategy()
    strategy.load(args.strategy)
    
    # Create bot player
    bot = BotPlayer(
        strategy=strategy,
        use_depth_limited_search=args.use_dls,
        search_depth=args.search_depth
    )
    
    # Create players based on opponent type
    players = []
    
    if args.opponent == 'human':
        # One human player vs. bots
        players.append(Player("Human", is_human=True))
        for i in range(args.num_opponents):
            players.append(BotPlayer(
                strategy=strategy,
                name=f"Bot-{i+1}",  # Pass a name
                use_depth_limited_search=args.use_dls,
                search_depth=args.search_depth
            ))
    elif args.opponent == 'random':
        # Human vs. random opponents
        players.append(Player("Human", is_human=True))
        for i in range(args.num_opponents):
            players.append(Player(f"Random-{i+1}", is_human=False, is_random=True))
    elif args.opponent == 'bot':
        # Bot vs. bot game (spectator mode)
        for i in range(args.num_players):
            players.append(BotPlayer(
                strategy=strategy,
                name=f"Bot-{i+1}",  # Pass a name
                use_depth_limited_search=args.use_dls,
                search_depth=args.search_depth
            ))
    
    # Create and run the game
    print(f"Starting poker game with {len(players)} players...")
    game = PokerGame(
        players=players,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        interactive=True
    )
    game.run()

def evaluate_bot(args):
    """Evaluate the poker bot."""
    # Check if strategy file exists
    if not args.strategy:
        print("Error: Strategy file must be specified with --strategy")
        return
    
    if not os.path.exists(args.strategy):
        print(f"Error: Strategy file {args.strategy} not found")
        return
    
    print(f"Loading strategy from {args.strategy}...")
    
    # Load strategy
    strategy = CFRStrategy()
    strategy.load(args.strategy)
    
    # Create bot player
    bot = BotPlayer(
        strategy=strategy,
        use_depth_limited_search=args.use_dls,
        search_depth=args.search_depth
    )
    
    # Create evaluator
    evaluator = BotEvaluator()
    
    # Evaluate against random opponents
    print(f"Evaluating bot against random opponents over {args.num_games} games...")
    results = evaluator.evaluate_against_random(
        bot=bot,
        num_games=args.num_games,
        num_opponents=args.num_opponents
    )
    print(f"Win rate against random opponents: {results['win_rate']:.2f}")
    print(f"Average profit per game: {results['avg_profit']:.2f}")
    
    # Evaluate exploitability
    print("Measuring exploitability...")
    exploitability = evaluator.measure_exploitability(bot.strategy_obj.strategy)
    print(f"Exploitability: {exploitability:.4f}")

def run_tests():
    """Run tests to verify the poker bot implementation."""
    print("Running tests to verify poker bot implementation...")
    
    # Import and run the simple test
    from organized_poker_bot.utils.simple_test import run_tests as run_simple_tests
    run_simple_tests()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        train_bot(args)
    elif args.mode == 'play':
        play_against_bot(args)
    elif args.mode == 'evaluate':
        evaluate_bot(args)
    elif args.mode == 'test':
        run_tests()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
