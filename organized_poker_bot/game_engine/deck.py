# --- START OF FILE organized_poker_bot/game_engine/deck.py ---
"""
Deck implementation for poker games.
(Refactored V2: Use Card.clone in Deck.clone)
"""

import random
import os
import sys
# Ensure Card can be imported - depends on project structure
try:
    from .card import Card # Relative import if in same directory
except ImportError:
    from organized_poker_bot.game_engine.card import Card # Absolute if path set

class Deck:
    """
    A standard deck of 52 playing cards.

    Attributes:
        cards: List of Card objects remaining in the deck.
    """

    def __init__(self):
        """ Initialize a new, full, ordered deck. """
        self.cards = []
        self.reset()

    def reset(self):
        """ Reset the deck to a full, ordered state. """
        self.cards = []
        # Use defined suits and ranks from Card class if available, else defaults
        suits_to_use = Card.SUITS.keys() if hasattr(Card, 'SUITS') else ['h', 'd', 'c', 's']
        ranks_to_use = Card.RANKS.keys() if hasattr(Card, 'RANKS') else range(2, 15)
        for suit in suits_to_use:
            for rank in ranks_to_use:
                self.cards.append(Card(rank, suit))

    def shuffle(self):
        """ Shuffle the remaining cards in the deck. """
        random.shuffle(self.cards)

    def deal(self):
        """
        Deal one card from the top of the deck (removes it).
        Returns Card or raises ValueError if empty.
        """
        if not self.cards:
            raise ValueError("Cannot deal from an empty deck")
        return self.cards.pop()

    def deal_specific(self, rank, suit):
        """ Deals a specific card if present, else returns None. """
        target_card = Card(rank, suit)
        if target_card in self.cards:
             self.cards.remove(target_card)
             return target_card
        return None

    def clone(self):
        """ Creates a deep copy of the deck state. """
        new_deck = Deck()
        # Ensure Card objects are cloned individually
        new_deck.cards = [card.clone() for card in self.cards]
        return new_deck

    def __len__(self):
        """ Returns number of cards remaining in the deck. """
        return len(self.cards)

    def __str__(self):
        """ String representation (e.g., "Deck(52 cards)"). """
        return f"Deck({len(self.cards)} cards)"

    def __repr__(self):
        """ Detailed representation showing some cards. """
        num_to_show = min(3, len(self.cards))
        cards_repr = ', '.join(repr(c) for c in self.cards[:num_to_show])
        if len(self.cards) > num_to_show:
            cards_repr += ', ...'
        return f"Deck([{cards_repr}])"

# --- END OF FILE organized_poker_bot/game_engine/deck.py ---
