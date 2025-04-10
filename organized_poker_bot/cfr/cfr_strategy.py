"""
CFR strategy implementation for poker games.
This module provides a class for using trained CFR strategies.
"""

import random
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.cfr.card_abstraction import CardAbstraction

class CFRStrategy:
    """
    A class for using trained CFR strategies.
    
    This class provides methods for using a trained CFR strategy to make decisions
    in a poker game.
    
    Attributes:
        strategy: Dictionary mapping information set keys to action probabilities
    """
    
    def __init__(self):
        """
        Initialize the CFR strategy.
        """
        self.strategy = {}
    
    def get_action(self, game_state, player_idx):
        """
        Get the best action for the current game state.
        
        Args:
            game_state: The current game state
            player_idx: The player index
            
        Returns:
            tuple or str: The chosen action (either a tuple of (action_type, amount) or a string)
        """
        # Get the information set key for the current game state
        info_set_key = self._create_info_set_key(game_state, player_idx)
        
        # If we don't have a strategy for this information set, use a default strategy
        if info_set_key not in self.strategy:
            return self._default_strategy(game_state)
        
        # Get the strategy for this information set
        action_probs = self.strategy[info_set_key]
        
        # Choose an action based on the strategy
        return self._choose_action(action_probs, game_state)
    
    def _create_info_set_key(self, game_state, player_idx):
        """
        Create a key for an information set.
        
        Args:
            game_state: The current game state
            player_idx: The player index
            
        Returns:
            str: A string key for the information set
        """
        # Get the player's hole cards
        hole_cards = game_state.hole_cards[player_idx] if hasattr(game_state, 'hole_cards') else []
        
        # Get the community cards
        community_cards = game_state.community_cards if hasattr(game_state, 'community_cards') else []
        
        # Create a key based on the cards
        if hole_cards:
            if not community_cards:
                # Preflop
                hole_card_bucket = CardAbstraction.get_preflop_abstraction(hole_cards)
                cards_key = f"preflop_bucket_{hole_card_bucket}"
            else:
                # Postflop
                hole_card_bucket = CardAbstraction.get_postflop_abstraction(
                    hole_cards, community_cards)
                round_name = self._determine_round(community_cards)
                cards_key = f"{round_name}_bucket_{hole_card_bucket}"
        else:
            cards_key = "no_cards"
        
        # Include position information
        position = game_state.get_position(player_idx) if hasattr(game_state, 'get_position') else player_idx
        position_key = f"pos_{position}"
        
        # Include round information
        round_key = f"round_{game_state.betting_round}" if hasattr(game_state, 'betting_round') else ""
        
        # Include pot and stack information if available
        pot_key = ""
        stack_key = ""
        if hasattr(game_state, 'pot'):
            pot_key = f"pot_{game_state.pot // game_state.big_blind}"
        if hasattr(game_state, 'player_stacks') and len(game_state.player_stacks) > player_idx:
            stack_key = f"stack_{game_state.player_stacks[player_idx] // game_state.big_blind}"
        
        # Combine all components into a single key
        components = [comp for comp in [cards_key, position_key, round_key, pot_key, stack_key] if comp]
        return "|".join(components)
    
    def _determine_round(self, community_cards):
        """
        Determine the current betting round based on community cards.
        
        Args:
            community_cards: List of community cards
            
        Returns:
            str: The current round name
        """
        num_cards = len(community_cards)
        if num_cards == 0:
            return "preflop"
        elif num_cards == 3:
            return "flop"
        elif num_cards == 4:
            return "turn"
        elif num_cards == 5:
            return "river"
        else:
            return f"unknown_{num_cards}"
    
    def _choose_action(self, action_probs, game_state):
        """
        Choose an action based on the strategy.
        
        Args:
            action_probs: Dictionary mapping actions to probabilities
            game_state: The current game state
            
        Returns:
            tuple or str: The chosen action
        """
        # Get available actions
        available_actions = game_state.get_available_actions() if hasattr(game_state, 'get_available_actions') else []
        
        # Filter out actions that are not available
        valid_actions = {}
        for action, prob in action_probs.items():
            if action in available_actions:
                valid_actions[action] = prob
        
        # If there are no valid actions, use a default strategy
        if not valid_actions:
            return self._default_strategy(game_state)
        
        # Normalize probabilities
        total_prob = sum(valid_actions.values())
        if total_prob > 0:
            for action in valid_actions:
                valid_actions[action] /= total_prob
        
        # Choose an action based on the probabilities
        actions = list(valid_actions.keys())
        probs = list(valid_actions.values())
        
        # Choose an action
        action = random.choices(actions, weights=probs, k=1)[0]
        
        # Parse the action if it's a string representation of a tuple
        if isinstance(action, str) and action.startswith(("bet_", "raise_")):
            action_type, amount = action.split("_", 1)
            return (action_type, int(amount))
        
        return action
    
    def _default_strategy(self, game_state):
        """
        Use a default strategy when we don't have a trained strategy.
        
        Args:
            game_state: The current game state
            
        Returns:
            tuple or str: The chosen action
        """
        # Get available actions
        available_actions = game_state.get_available_actions() if hasattr(game_state, 'get_available_actions') else []
        
        # If there are no available actions, fold
        if not available_actions:
            return "fold"
        
        # Prefer checking or calling over folding
        if "check" in available_actions:
            return "check"
        elif "call" in available_actions:
            return "call"
        
        # If we can't check or call, choose a random action
        return random.choice(available_actions)
    
    def save(self, filename):
        """
        Save the strategy to a file.
        
        Args:
            filename: Path to save the strategy
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.strategy, f)
    
    def load(self, filename):
        """
        Load a strategy from a file.
        
        Args:
            filename: Path to load the strategy from
        """
        import pickle
        with open(filename, 'rb') as f:
            self.strategy = pickle.load(f)
    
    def __str__(self):
        """
        Get a string representation of the strategy.
        
        Returns:
            str: A string representation
        """
        return f"CFRStrategy({len(self.strategy)} info sets)"
