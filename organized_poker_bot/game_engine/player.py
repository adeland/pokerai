"""
Player class for representing a poker player.
"""

import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Player:
    """
    A class representing a poker player.
    
    Attributes:
        name (str): The name of the player
        stack (int): The amount of chips the player has
        hole_cards (list): The player's private cards
        position (int): The player's position at the table
        is_active (bool): Whether the player is active in the current hand
    """
    
    def __init__(self, name, stack=10000, position=0, is_human=False, is_random=False):
        """
        Initialize a player with a name, stack, and position.
        
        Args:
            name (str): The name of the player
            stack (int): The initial chip stack
            position (int): The position at the table (0-5 for 6-max)
        """
        self.name = name
        self.stack = stack
        self.position = position
        self.hole_cards = []
        self.is_active = True
        self.current_bet = 0
        self.total_bet = 0
        self.is_all_in = False
        self.is_human = is_human
        self.is_random = is_random
    
    def receive_cards(self, cards):
        """
        Give hole cards to the player.
        
        Args:
            cards (list): A list of Card objects
        """
        self.hole_cards = cards
    
    def place_bet(self, amount):
        """
        Place a bet of the specified amount.
        
        Args:
            amount (int): The amount to bet
            
        Returns:
            int: The actual amount bet (may be less if player doesn't have enough chips)
            
        Raises:
            ValueError: If the amount is negative or zero
        """
        if amount <= 0:
            raise ValueError("Bet amount must be positive")
        
        # Cap the bet at the player's stack
        actual_amount = min(amount, self.stack)
        
        self.stack -= actual_amount
        self.current_bet += actual_amount
        self.total_bet += actual_amount
        
        # Check if player is all-in
        if self.stack == 0:
            self.is_all_in = True
        
        return actual_amount
    
    def fold(self):
        """Fold the current hand."""
        self.is_active = False
    
    def reset_for_new_hand(self):
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.is_active = True
        self.current_bet = 0
        self.total_bet = 0
        self.is_all_in = False
    
    def __str__(self):
        """Return a string representation of the player."""
        status = "active" if self.is_active else "folded"
        cards = ", ".join(str(card) for card in self.hole_cards) if self.hole_cards else "no cards"
        return f"{self.name} (${self.stack}) - {status} with {cards}"
