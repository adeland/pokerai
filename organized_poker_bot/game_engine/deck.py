"""
Deck implementation for poker games.
This module provides a standard deck of cards for poker games.
"""

import random
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.game_engine.card import Card

class Deck:
    """
    A standard deck of 52 playing cards.
    
    This class represents a standard deck of 52 playing cards,
    with methods for shuffling and dealing cards.
    
    Attributes:
        cards: List of Card objects in the deck
    """
    
    def __init__(self):
        """
        Initialize a new deck of cards.
        """
        self.cards = []
        self.reset()
    
    def reset(self):
        """
        Reset the deck to a full, unshuffled state.
        """
        self.cards = []
        for suit in ['h', 'd', 'c', 's']:  # hearts, diamonds, clubs, spades
            for rank in range(2, 15):  # 2-14 (Ace is 14)
                self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        """
        Shuffle the deck.
        """
        random.shuffle(self.cards)
    
    def deal(self):
        """
        Deal a card from the deck.
        
        Returns:
            Card: The top card from the deck
        """
        if not self.cards:
            raise ValueError("Cannot deal from an empty deck")
        
        return self.cards.pop()
    
    def clone(self):
        """
        Create a deep copy of the deck.
        
        Returns:
            Deck: A new deck with the same cards
        """
        new_deck = Deck()
        new_deck.cards = [card.clone() for card in self.cards]
        return new_deck
    
    def __len__(self):
        """
        Get the number of cards in the deck.
        
        Returns:
            int: The number of cards
        """
        return len(self.cards)
    
    def __str__(self):
        """
        Get a string representation of the deck.
        
        Returns:
            str: A string representation
        """
        return f"Deck({len(self.cards)} cards)"
