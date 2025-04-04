"""
Card implementation for poker games.
This module provides a standard playing card for poker games.
"""

import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Card:
    """
    A standard playing card.
    
    This class represents a standard playing card with a rank and suit.
    
    Attributes:
        rank: Integer rank of the card (2-14, where 14 is Ace)
        suit: Character representing the suit ('h', 'd', 'c', 's')
    """
    
    # Constants for card ranks
    RANKS = {
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: 'T',
        11: 'J',
        12: 'Q',
        13: 'K',
        14: 'A'
    }
    
    # Constants for card suits
    SUITS = {
        'h': '♥',
        'd': '♦',
        'c': '♣',
        's': '♠'
    }
    
    def __init__(self, rank, suit):
        """
        Initialize a new card.
        
        Args:
            rank: Integer rank of the card (2-14, where 14 is Ace)
            suit: Character representing the suit ('h', 'd', 'c', 's')
        """
        self.rank = rank
        self.suit = suit
    
    def get_rank_char(self):
        """
        Get the character representation of the card's rank.
        
        Returns:
            str: Character representation of the rank
        """
        return self.RANKS[self.rank]
    
    def get_suit_char(self):
        """
        Get the character representation of the card's suit.
        
        Returns:
            str: Character representation of the suit
        """
        return self.SUITS[self.suit]
    
    def clone(self):
        """
        Create a deep copy of the card.
        
        Returns:
            Card: A new card with the same rank and suit
        """
        return Card(self.rank, self.suit)
    
    def __eq__(self, other):
        """
        Check if two cards are equal.
        
        Args:
            other: Another card to compare with
            
        Returns:
            bool: True if the cards are equal, False otherwise
        """
        if not isinstance(other, Card):
            return False
        
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        """
        Get a hash value for the card.
        
        Returns:
            int: Hash value
        """
        return hash((self.rank, self.suit))
    
    def __str__(self):
        """
        Get a string representation of the card.
        
        Returns:
            str: A string representation
        """
        return f"{self.get_rank_char()}{self.get_suit_char()}"
    
    def __repr__(self):
        """
        Get a string representation of the card for debugging.
        
        Returns:
            str: A string representation
        """
        return f"Card({self.rank}, '{self.suit}')"
