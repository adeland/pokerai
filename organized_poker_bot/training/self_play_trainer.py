import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Self-play training framework for poker bot using CFR.
"""

import os
import pickle
import time
import numpy as np
from tqdm import tqdm

from ..game_engine import PokerGame
from ..cfr import CFRTrainer, CFRStrategy

class SelfPlayTrainer:
    """
    A class for training poker bots through self-play using CFR.
    
    This class manages the training process, including running CFR iterations,
    saving checkpoints, and evaluating progress.
    """
    
    def __init__(self, output_dir, small_blind=50, big_blind=100, starting_stack=10000):
        """
        Initialize the self-play trainer.
        
        Args:
            output_dir (str): Directory to save training outputs
            small_blind (int): Small blind amount
            big_blind (int): Big blind amount
            starting_stack (int): Starting chip stack for each player
        """
        self.output_dir = output_dir
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CFR trainer
        self.cfr_trainer = CFRTrainer(PokerGame)
        
        # Training statistics
        self.training_stats = {
            'iterations': 0,
            'avg_utility': [],
            'exploitability': [],
            'training_time': 0
        }
    
    def train(self, iterations, checkpoint_interval=1000):
        """
        Train the poker bot using CFR self-play.
        
        Args:
            iterations (int): Number of iterations to train for
            checkpoint_interval (int): How often to save checkpoints
            
        Returns:
            dict: Training statistics
        """
        start_time = time.time()
        
        # Run CFR iterations
        print(f"\n{'='*60}")
        print(f"STARTING POKER BOT TRAINING")
        print(f"{'='*60}")
        print(f"Total iterations planned: {iterations}")
        print(f"Checkpoint interval: {checkpoint_interval}")
        print(f"Output directory: {self.output_dir}")
        print(f"Game parameters: SB={self.small_blind}, BB={self.big_blind}, Stack={self.starting_stack}")
        print(f"{'='*60}\n")
        
        # Track utility for progress monitoring
        utility_sum = 0
        
        # Print initial information sets
        info_set_count = len(self.cfr_trainer.information_sets)
        print(f"Initial information sets: {info_set_count}")
        
        for i in tqdm(range(iterations)):
            # Run one iteration of CFR
            utility = self._run_cfr_iteration()
            utility_sum += utility
            
            # Update statistics
            self.training_stats['iterations'] += 1
            
            # Print progress for every iteration
            if i < 10 or (i + 1) % 10 == 0:  # More frequent updates in early iterations
                print(f"Iteration {i+1}/{iterations}: Utility = {utility:.4f}")
            
            # More detailed stats every 100 iterations
            if (i + 1) % 100 == 0 or i == iterations - 1:
                avg_utility = utility_sum / min(100, i+1)
                self.training_stats['avg_utility'].append(avg_utility)
                
                # Count current information sets
                current_info_sets = len(self.cfr_trainer.information_sets)
                info_set_growth = current_info_sets - info_set_count
                info_set_count = current_info_sets
                
                elapsed_time = time.time() - start_time
                time_per_iter = elapsed_time / (i + 1)
                estimated_remaining = time_per_iter * (iterations - i - 1)
                
                print(f"\n{'*'*40}")
                print(f"TRAINING PROGRESS - Iteration {i+1}/{iterations}")
                print(f"{'*'*40}")
                print(f"Average utility (last 100): {avg_utility:.6f}")
                print(f"Information sets: {current_info_sets} (+{info_set_growth} new)")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Time per iteration: {time_per_iter:.4f} seconds")
                print(f"Estimated remaining time: {estimated_remaining:.2f} seconds")
                
                # Reset utility sum for next batch
                utility_sum = 0
                
                # Print some example strategies
                if current_info_sets > 0:
                    print("\nExample strategies:")
                    strategy = self.cfr_trainer.get_strategy()
                    sample_keys = list(strategy.keys())[:3]  # Show up to 3 examples
                    for key in sample_keys:
                        print(f"  {key}: {strategy[key]}")
                print(f"{'*'*40}\n")
            
            # Save checkpoint if needed
            if (i + 1) % checkpoint_interval == 0 or i == iterations - 1:
                print(f"\nSaving checkpoint at iteration {i+1}...")
                self.save_checkpoint()
                
                # Estimate exploitability (simplified)
                print("Estimating exploitability (this may take a moment)...")
                exploitability = self._estimate_exploitability()
                self.training_stats['exploitability'].append(exploitability)
                print(f"Checkpoint saved. Estimated exploitability: {exploitability:.6f}")
                strategy_file = os.path.join(self.output_dir, f'strategy_iter_{self.training_stats["iterations"]}.pkl')
                print(f"Strategy saved to: {strategy_file}")
        
        # Update training time
        total_time = time.time() - start_time
        self.training_stats['training_time'] += total_time
        
        # Save final model
        print("\nTraining complete! Saving final strategy...")
        self.save_checkpoint(is_final=True)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {iterations}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final information sets: {len(self.cfr_trainer.information_sets)}")
        if self.training_stats['exploitability']:
            print(f"Final exploitability: {self.training_stats['exploitability'][-1]:.6f}")
        print(f"Final strategy saved to: {os.path.join(self.output_dir, 'final_strategy.pkl')}")
        print(f"{'='*60}\n")
        
        return self.training_stats
    
    def _run_cfr_iteration(self):
        """
        Run a single iteration of CFR.
        
        Returns:
            float: The utility for this iteration
        """
        # Initialize a new game for this iteration
        player_names = ["Player_0", "Player_1"]
        game = PokerGame(player_names, self.starting_stack, self.small_blind, self.big_blind)
        game.start_new_hand()
        
        # Run CFR on the initial game state
        utility = self.cfr_trainer._cfr_recursive(game.game_state, 1.0, 1.0)
        
        return utility
    
    def _estimate_exploitability(self, num_hands=1000):
        """
        Estimate the exploitability of the current strategy.
        
        In a full implementation, this would involve computing a best response
        strategy and measuring its performance against the current strategy.
        For simplicity, we'll use a heuristic approach here.
        
        Args:
            num_hands (int): Number of hands to simulate
            
        Returns:
            float: Estimated exploitability
        """
        # Get current strategy
        strategy = self.cfr_trainer.get_strategy()
        
        # Create a strategy object
        strategy_player = CFRStrategy()
        strategy_player.strategy = strategy
        
        # Create a simple exploiter (this would be a best response in a full implementation)
        # For simplicity, we'll use a random player as a baseline
        
        # Simulate hands
        total_utility = 0
        
        for _ in range(num_hands):
            # Initialize a new game
            player_names = ["Strategy", "Exploiter"]
            game = PokerGame(player_names, self.starting_stack, self.small_blind, self.big_blind)
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
                    if current_player_idx == 0:  # Strategy player
                        action_tuple = strategy_player.get_action(game_state, current_player_idx)
                    else:  # Random exploiter
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
                        
                        action_tuple = (action_type, amount)
                    
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
            winners = game_state.determine_winners()
            
            # Calculate utility for strategy player
            for player, _, pot_share in winners:
                if player == game_state.players[0]:
                    total_utility += pot_share
                    break
        
        # Calculate average utility
        avg_utility = total_utility / num_hands
        
        # Normalize by big blinds
        exploitability = avg_utility / self.big_blind
        
        return exploitability
    
    def save_checkpoint(self, is_final=False):
        """
        Save a checkpoint of the current training state.
        
        Args:
            is_final (bool): Whether this is the final checkpoint
        """
        # Save strategy
        strategy = self.cfr_trainer.get_strategy()
        
        if is_final:
            strategy_file = os.path.join(self.output_dir, "final_strategy.pkl")
        else:
            strategy_file = os.path.join(self.output_dir, f"strategy_iter_{self.training_stats['iterations']}.pkl")
        
        with open(strategy_file, 'wb') as f:
            pickle.dump(strategy, f)
        
        # Save training statistics
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        with open(stats_file, 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def load_checkpoint(self, checkpoint_file):
        """
        Load a checkpoint to resume training.
        
        Args:
            checkpoint_file (str): Path to the checkpoint file
        """
        # Load strategy
        with open(checkpoint_file, 'rb') as f:
            strategy = pickle.load(f)
        
        # Recreate information sets in the CFR trainer
        for key, strat in strategy.items():
            self.cfr_trainer.get_information_set(key, list(strat.keys()))
            
        # Load training statistics if available
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        if os.path.exists(stats_file):
            with open(stats_file, 'rb') as f:
                self.training_stats = pickle.load(f)
    
    def get_trained_strategy(self):
        """
        Get the current trained strategy.
        
        Returns:
            CFRStrategy: The trained strategy
        """
        strategy_obj = CFRStrategy()
        strategy_obj.strategy = self.cfr_trainer.get_strategy()
        return strategy_obj
