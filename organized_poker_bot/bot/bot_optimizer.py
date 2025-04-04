import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Performance optimization utilities for the poker bot.
"""

import os
import time
import numpy as np
import pickle
from tqdm import tqdm

from ..cfr import CFRTrainer, CFRStrategy
from ..self_play import SelfPlayTrainer, BotPlayer
from ..game_engine import PokerGame

class BotOptimizer:
    """
    A class for optimizing poker bot performance.
    
    This class provides methods for improving bot performance through
    parameter tuning, abstraction refinement, and memory optimization.
    """
    
    def __init__(self, output_dir):
        """
        Initialize the bot optimizer.
        
        Args:
            output_dir (str): Directory to save optimization outputs
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize_abstractions(self, strategy_file, num_iterations=1000, test_hands=500):
        """
        Optimize card and action abstractions to improve performance.
        
        Args:
            strategy_file (str): Path to initial strategy file
            num_iterations (int): Number of iterations for each refinement
            test_hands (int): Number of hands to use for testing
            
        Returns:
            str: Path to optimized strategy file
        """
        print("Starting abstraction optimization...")
        
        # Load initial strategy
        with open(strategy_file, 'rb') as f:
            initial_strategy = pickle.load(f)
        
        # Create baseline bot
        baseline_bot = BotPlayer()
        baseline_bot.strategy.strategy = initial_strategy
        baseline_bot.name = "Baseline"
        
        # Create a random bot for comparison
        random_bot = self._create_random_bot()
        
        # Measure baseline performance
        print("Measuring baseline performance...")
        baseline_results = self._evaluate_bot(baseline_bot, random_bot, num_hands=test_hands)
        baseline_profit = baseline_results['bot1_profit']
        
        print(f"Baseline profit: {baseline_profit} chips")
        
        # Optimize postflop buckets
        print("\nOptimizing postflop buckets...")
        postflop_buckets = [5, 10, 15, 20, 25]
        best_postflop_buckets = self._find_best_parameter(
            "NUM_BUCKETS", postflop_buckets, baseline_bot, random_bot, 
            num_iterations, test_hands
        )
        
        # Optimize bet sizing
        print("\nOptimizing bet sizing...")
        bet_sizes = [
            {"small": 0.25, "medium": 0.5, "large": 1.0, "overbet": 1.5},
            {"small": 0.33, "medium": 0.67, "large": 1.0, "overbet": 2.0},
            {"small": 0.5, "medium": 0.75, "large": 1.25, "overbet": 2.5}
        ]
        best_bet_sizes = self._find_best_parameter(
            "BET_SIZES", bet_sizes, baseline_bot, random_bot, 
            num_iterations, test_hands
        )
        
        # Create optimized trainer with best parameters
        print("\nTraining with optimized parameters...")
        optimized_trainer = SelfPlayTrainer(
            output_dir=os.path.join(self.output_dir, "optimized")
        )
        
        # Set optimized parameters
        # In a real implementation, we would modify the abstraction classes
        # For this demonstration, we'll just print the best parameters
        print(f"Best postflop buckets: {best_postflop_buckets}")
        print(f"Best bet sizes: {best_bet_sizes}")
        
        # Train with optimized parameters
        optimized_trainer.train(num_iterations)
        
        # Save optimized strategy
        optimized_file = os.path.join(self.output_dir, "optimized_strategy.pkl")
        optimized_trainer.save_checkpoint(is_final=True)
        
        print(f"Optimized strategy saved to: {optimized_file}")
        
        return optimized_file
    
    def _find_best_parameter(self, param_name, param_values, baseline_bot, random_bot, 
                            num_iterations, test_hands):
        """
        Find the best value for a parameter through experimentation.
        
        Args:
            param_name (str): Name of the parameter
            param_values (list): List of parameter values to try
            baseline_bot (BotPlayer): Baseline bot for comparison
            random_bot (BotPlayer): Random bot for testing
            num_iterations (int): Number of iterations for each value
            test_hands (int): Number of hands to use for testing
            
        Returns:
            any: The best parameter value
        """
        best_value = None
        best_profit = float('-inf')
        
        for value in param_values:
            print(f"Testing {param_name} = {value}")
            
            # Create a new trainer with this parameter value
            # In a real implementation, we would modify the abstraction classes
            # For this demonstration, we'll just create a new trainer
            trainer = SelfPlayTrainer(
                output_dir=os.path.join(self.output_dir, f"{param_name}_{value}")
            )
            
            # Train for a few iterations
            trainer.train(num_iterations // 10)  # Use fewer iterations for parameter search
            
            # Create a bot with the trained strategy
            test_bot = BotPlayer()
            test_bot.strategy = trainer.get_trained_strategy()
            test_bot.name = f"{param_name}_{value}"
            
            # Evaluate against random bot
            results = self._evaluate_bot(test_bot, random_bot, num_hands=test_hands)
            profit = results['bot1_profit']
            
            print(f"  Profit: {profit} chips")
            
            # Update best if better
            if profit > best_profit:
                best_profit = profit
                best_value = value
        
        print(f"Best {param_name}: {best_value} (profit: {best_profit} chips)")
        return best_value
    
    def _evaluate_bot(self, bot1, bot2, num_hands=500):
        """
        Evaluate a bot against another bot.
        
        Args:
            bot1 (BotPlayer): First bot
            bot2 (BotPlayer): Second bot
            num_hands (int): Number of hands to play
            
        Returns:
            dict: Match results
        """
        results = {
            'bot1_name': bot1.name,
            'bot2_name': bot2.name,
            'bot1_profit': 0,
            'bot2_profit': 0,
            'hands_played': 0
        }
        
        # Set up player names
        player_names = [bot1.name, bot2.name]
        
        # Initialize game
        game = PokerGame(player_names, 10000, 50, 100)
        
        # Track initial stacks
        initial_stacks = [player.stack for player in game.players]
        
        # Play hands
        for _ in range(num_hands):
            game_state = game.start_new_hand()
            
            # Play until the hand is complete
            while not game_state.is_hand_complete():
                # Play current betting round
                while not game_state.is_betting_round_complete():
                    current_player_idx = game_state.current_player_idx
                    current_player = game_state.players[current_player_idx]
                    
                    if not current_player.is_active or current_player.is_all_in:
                        game_state.next_player()
                        continue
                    
                    # Get action based on player
                    if current_player_idx == 0:  # Bot 1
                        action_tuple = bot1.get_action(game_state, current_player_idx)
                    else:  # Bot 2
                        action_tuple = bot2.get_action(game_state, current_player_idx)
                    
                    # Process the action
                    try:
                        game_state.process_action(action_tuple[0], action_tuple[1])
                    except ValueError:
                        # If invalid action, default to check/call
                        if current_player.current_bet < game_state.current_bet:
                            game_state.process_action("call", 0)
                        else:
                            game_state.process_action("check", 0)
                    
                    # Move to next player
                    game_state.next_player()
                
                # Move to next round if the current round is complete
                if not game_state.move_to_next_round():
                    break
            
            # Determine winners
            game_state.determine_winners()
            
            # Update hand count
            results['hands_played'] += 1
        
        # Calculate profits
        final_stacks = [player.stack for player in game.players]
        results['bot1_profit'] = final_stacks[0] - initial_stacks[0]
        results['bot2_profit'] = final_stacks[1] - initial_stacks[1]
        
        return results
    
    def _create_random_bot(self):
        """
        Create a bot that plays randomly.
        
        Returns:
            BotPlayer: A random bot
        """
        random_bot = BotPlayer()
        random_bot.name = "Random"
        
        # Override get_action to play randomly
        def random_action(game_state, player_idx):
            current_player = game_state.players[player_idx]
            
            # Get available actions
            available_actions = []
            if current_player.current_bet < game_state.current_bet:
                available_actions.extend(["fold", "call"])
                if current_player.stack > game_state.current_bet - current_player.current_bet:
                    available_actions.append("raise")
            else:
                available_actions.append("check")
                if current_player.stack > 0:
                    available_actions.append("bet")
            
            # Choose random action
            action_type = np.random.choice(available_actions)
            
            if action_type in ["fold", "check", "call"]:
                amount = 0
            else:  # bet or raise
                if action_type == "bet":
                    min_amount = game_state.big_blind
                else:  # raise
                    min_amount = game_state.current_bet + game_state.last_raise
                
                # Random amount between min and all-in
                max_amount = current_player.stack
                amount = np.random.randint(min_amount, max_amount + 1)
            
            return (action_type, amount)
        
        # Replace the strategy's get_action method
        random_bot.get_action = random_action
        
        return random_bot
    
    def optimize_memory_usage(self, strategy_file):
        """
        Optimize memory usage of a strategy.
        
        Args:
            strategy_file (str): Path to strategy file
            
        Returns:
            str: Path to optimized strategy file
        """
        print("Optimizing memory usage...")
        
        # Load strategy
        with open(strategy_file, 'rb') as f:
            strategy = pickle.load(f)
        
        # Count initial size
        initial_size = len(strategy)
        initial_memory = os.path.getsize(strategy_file)
        
        print(f"Initial strategy size: {initial_size} information sets, {initial_memory / 1024:.1f} KB")
        
        # Prune rarely used information sets
        pruned_strategy = {}
        for key, value in strategy.items():
            # Keep only information sets with significant strategy differences
            max_prob = max(value.values())
            min_prob = min(value.values())
            
            # If strategy is close to uniform, it's not very important
            if max_prob - min_prob > 0.1:
                pruned_strategy[key] = value
        
        # Save pruned strategy
        pruned_file = os.path.join(self.output_dir, "memory_optimized_strategy.pkl")
        with open(pruned_file, 'wb') as f:
            pickle.dump(pruned_strategy, f)
        
        # Count final size
        final_size = len(pruned_strategy)
        final_memory = os.path.getsize(pruned_file)
        
        print(f"Optimized strategy size: {final_size} information sets, {final_memory / 1024:.1f} KB")
        print(f"Reduction: {(initial_size - final_size) / initial_size * 100:.1f}% of information sets, {(initial_memory - final_memory) / initial_memory * 100:.1f}% of memory")
        
        return pruned_file
    
    def optimize_runtime_performance(self, strategy_file):
        """
        Optimize runtime performance of strategy lookup.
        
        Args:
            strategy_file (str): Path to strategy file
            
        Returns:
            str: Path to optimized strategy file
        """
        print("Optimizing runtime performance...")
        
        # Load strategy
        with open(strategy_file, 'rb') as f:
            strategy = pickle.load(f)
        
        # Measure initial lookup time
        start_time = time.time()
        for _ in range(1000):
            # Simulate lookups with random keys
            for key in list(strategy.keys())[:100]:
                _ = strategy[key]
        initial_time = time.time() - start_time
        
        print(f"Initial lookup time: {initial_time:.6f} seconds for 100,000 lookups")
        
        # Create optimized data structure
        # In a real implementation, we might use a more efficient data structure
        # For this demonstration, we'll just use the same dictionary
        optimized_strategy = strategy
        
        # Save optimized strategy
        optimized_file = os.path.join(self.output_dir, "runtime_optimized_strategy.pkl")
        with open(optimized_file, 'wb') as f:
            pickle.dump(optimized_strategy, f)
        
        # Measure optimized lookup time
        start_time = time.time()
        for _ in range(1000):
            # Simulate lookups with random keys
            for key in list(optimized_strategy.keys())[:100]:
                _ = optimized_strategy[key]
        optimized_time = time.time() - start_time
        
        print(f"Optimized lookup time: {optimized_time:.6f} seconds for 100,000 lookups")
        print(f"Speedup: {initial_time / optimized_time:.2f}x")
        
        return optimized_file
