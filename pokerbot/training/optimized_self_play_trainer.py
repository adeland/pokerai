"""
Optimized self-play training implementation for poker CFR.
This module provides parallel training capabilities for faster convergence.
"""

import os
import sys
import pickle
import random
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_engine.poker_game import PokerGame
from game_engine.game_state import GameState
from cfr.cfr_trainer import CFRTrainer
from cfr.cfr_strategy import CFRStrategy

class OptimizedSelfPlayTrainer:
    """
    Optimized self-play training for poker CFR implementation.
    Uses parallel processing and linear CFR for faster convergence.
    """
    
    def __init__(self, game_state_class, num_players=6, num_workers=4):
        """
        Initialize the optimized self-play trainer.
        
        Args:
            game_state_class: Function that creates a new game state
            num_players: Number of players in the game
            num_workers: Number of parallel workers for training
        """
        self.game_state_class = game_state_class
        self.num_players = num_players
        self.num_workers = min(num_workers, mp.cpu_count())
        self.strategy = {}
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration = 0
    
    def train(self, iterations=1000, checkpoint_freq=100, output_dir="models"):
        """
        Train the strategy using optimized self-play.
        
        Args:
            iterations: Number of iterations to train
            checkpoint_freq: Frequency of saving checkpoints
            output_dir: Directory to save checkpoints
            
        Returns:
            Dictionary mapping information sets to action probabilities
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize strategy
        self.strategy = {}
        self.regret_sum = {}
        self.strategy_sum = {}
        self.iteration = 0
        
        # Create a pool of workers
        pool = mp.Pool(processes=self.num_workers)
        
        # Train for the specified number of iterations
        for i in tqdm(range(iterations)):
            self.iteration = i + 1
            
            # Divide work among workers
            batch_size = max(1, iterations // (self.num_workers * 10))
            batch_size = min(batch_size, 10)  # Cap batch size
            
            # Create batches of work
            batches = []
            for _ in range(self.num_workers):
                batches.append(batch_size)
            
            # Submit work to the pool
            results = pool.map(self._train_batch, batches)
            
            # Merge results
            for batch_regret_sum, batch_strategy_sum in results:
                self._merge_results(batch_regret_sum, batch_strategy_sum)
            
            # Save checkpoint if needed
            if (i + 1) % checkpoint_freq == 0:
                self._save_checkpoint(output_dir, i + 1)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Compute the final strategy
        final_strategy = self._compute_strategy()
        
        # Save the final strategy
        with open(os.path.join(output_dir, f"strategy_{iterations}.pkl"), "wb") as f:
            pickle.dump(final_strategy, f)
        
        return final_strategy
    
    def _train_batch(self, batch_size):
        """
        Train a batch of iterations.
        
        Args:
            batch_size: Number of iterations in the batch
            
        Returns:
            Tuple of (regret_sum, strategy_sum) for the batch
        """
        # Create local copies of regret_sum and strategy_sum
        local_regret_sum = {}
        local_strategy_sum = {}
        
        # Train for batch_size iterations
        for _ in range(batch_size):
            # Create a new game state
            game_state = self.game_state_class(self.num_players)
            
            # Traverse the game tree
            self._traverse_game_tree(game_state, local_regret_sum, local_strategy_sum)
        
        return local_regret_sum, local_strategy_sum
    
    def _traverse_game_tree(self, game_state, regret_sum, strategy_sum):
        """
        Traverse the game tree and update regrets and strategies.
        
        Args:
            game_state: Current game state
            regret_sum: Dictionary of regret sums
            strategy_sum: Dictionary of strategy sums
            
        Returns:
            Expected utility for each player
        """
        # If the game is terminal, return the utilities
        if game_state.is_terminal():
            return game_state.get_utilities()
        
        # Get the current player
        player = game_state.current_player
        
        # Get the information set key
        info_set_key = self._create_info_set_key(game_state, player)
        
        # Get available actions
        actions = game_state.get_available_actions()
        
        # If there are no actions, move to the next state
        if not actions:
            next_state = game_state.apply_action(None)
            return self._traverse_game_tree(next_state, regret_sum, strategy_sum)
        
        # Get the strategy for this information set
        strategy = self._get_strategy(info_set_key, actions, regret_sum)
        
        # Initialize action utilities
        action_utils = {action: 0 for action in actions}
        
        # Initialize expected utility
        util = [0] * self.num_players
        
        # Recursively traverse the game tree for each action
        for action in actions:
            # Apply the action to get the next state
            next_state = game_state.apply_action(action)
            
            # Recursively traverse the game tree
            action_util = self._traverse_game_tree(next_state, regret_sum, strategy_sum)
            
            # Update action utility
            action_utils[action] = action_util[player]
            
            # Update expected utility
            for p in range(self.num_players):
                util[p] += strategy[action] * action_util[p]
        
        # Update regrets and strategy sums
        if player != -1:  # Skip chance player
            # Update strategy sum (using linear CFR)
            if info_set_key not in strategy_sum:
                strategy_sum[info_set_key] = {action: 0 for action in actions}
            
            for action in actions:
                strategy_sum[info_set_key][action] += self.iteration * strategy[action]
            
            # Update regret sum
            if info_set_key not in regret_sum:
                regret_sum[info_set_key] = {action: 0 for action in actions}
            
            for action in actions:
                regret = action_utils[action] - util[player]
                regret_sum[info_set_key][action] += regret
        
        return util
    
    def _get_strategy(self, info_set_key, actions, regret_sum):
        """
        Get the current strategy for an information set.
        
        Args:
            info_set_key: Information set key
            actions: Available actions
            regret_sum: Dictionary of regret sums
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        # If this is a new information set, initialize it
        if info_set_key not in regret_sum:
            regret_sum[info_set_key] = {action: 0 for action in actions}
            return {action: 1.0 / len(actions) for action in actions}
        
        # Get the regret sum for this information set
        info_set_regret_sum = regret_sum[info_set_key]
        
        # Compute the strategy using regret matching
        strategy = {}
        normalizing_sum = 0
        
        # Use positive regrets only
        for action in actions:
            strategy[action] = max(0, info_set_regret_sum.get(action, 0))
            normalizing_sum += strategy[action]
        
        # Normalize the strategy
        if normalizing_sum > 0:
            for action in actions:
                strategy[action] /= normalizing_sum
        else:
            # If all regrets are negative or zero, use a uniform strategy
            for action in actions:
                strategy[action] = 1.0 / len(actions)
        
        return strategy
    
    def _merge_results(self, batch_regret_sum, batch_strategy_sum):
        """
        Merge batch results into the main results.
        
        Args:
            batch_regret_sum: Regret sums from the batch
            batch_strategy_sum: Strategy sums from the batch
        """
        # Merge regret sums
        for info_set_key, regrets in batch_regret_sum.items():
            if info_set_key not in self.regret_sum:
                self.regret_sum[info_set_key] = regrets.copy()
            else:
                for action, regret in regrets.items():
                    if action not in self.regret_sum[info_set_key]:
                        self.regret_sum[info_set_key][action] = regret
                    else:
                        self.regret_sum[info_set_key][action] += regret
        
        # Merge strategy sums
        for info_set_key, strategies in batch_strategy_sum.items():
            if info_set_key not in self.strategy_sum:
                self.strategy_sum[info_set_key] = strategies.copy()
            else:
                for action, strategy in strategies.items():
                    if action not in self.strategy_sum[info_set_key]:
                        self.strategy_sum[info_set_key][action] = strategy
                    else:
                        self.strategy_sum[info_set_key][action] += strategy
    
    def _compute_strategy(self):
        """
        Compute the average strategy.
        
        Returns:
            Dictionary mapping information sets to action probabilities
        """
        avg_strategy = {}
        
        for info_set_key, strategies in self.strategy_sum.items():
            avg_strategy[info_set_key] = {}
            normalizing_sum = sum(strategies.values())
            
            if normalizing_sum > 0:
                for action, strategy_sum in strategies.items():
                    avg_strategy[info_set_key][action] = strategy_sum / normalizing_sum
            else:
                # If the normalizing sum is zero, use a uniform strategy
                for action in strategies:
                    avg_strategy[info_set_key][action] = 1.0 / len(strategies)
        
        return avg_strategy
    
    def _save_checkpoint(self, output_dir, iteration):
        """
        Save a checkpoint of the current training state.
        
        Args:
            output_dir: Directory to save the checkpoint
            iteration: Current iteration number
        """
        checkpoint = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "iteration": iteration
        }
        
        with open(os.path.join(output_dir, f"checkpoint_{iteration}.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Also save the current strategy
        strategy = self._compute_strategy()
        with open(os.path.join(output_dir, f"strategy_{iteration}.pkl"), "wb") as f:
            pickle.dump(strategy, f)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.regret_sum = checkpoint["regret_sum"]
        self.strategy_sum = checkpoint["strategy_sum"]
        self.iteration = checkpoint["iteration"]
    
    def _create_info_set_key(self, game_state, player):
        """
        Create a key for an information set.
        
        Args:
            game_state: Current game state
            player: Player index
            
        Returns:
            String key for the information set
        """
        # Get the player's hole cards
        hole_cards = game_state.hole_cards[player] if hasattr(game_state, 'hole_cards') else []
        
        # Get the community cards
        community_cards = game_state.community_cards if hasattr(game_state, 'community_cards') else []
        
        # Create a key based on the cards
        if hole_cards:
            hole_cards_str = ''.join(str(card) for card in hole_cards)
        else:
            hole_cards_str = "no_hole_cards"
        
        if community_cards:
            community_cards_str = ''.join(str(card) for card in community_cards)
        else:
            community_cards_str = "no_community_cards"
        
        # Include betting history
        betting_history = game_state.betting_history if hasattr(game_state, 'betting_history') else []
        betting_history_str = ''.join(str(action) for action in betting_history)
        
        # Include position information
        position = game_state.get_position(player) if hasattr(game_state, 'get_position') else player
        position_key = f"pos_{position}"
        
        # Include round information
        round_key = f"round_{game_state.betting_round}" if hasattr(game_state, 'betting_round') else ""
        
        # Combine all components into a single key
        components = [comp for comp in [hole_cards_str, community_cards_str, betting_history_str, position_key, round_key] if comp]
        return "|".join(components)
