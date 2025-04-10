# --- START OF FILE organized_poker_bot/game_engine/card.py ---
"""
Card implementation for poker games.
(Refactored V3: Added clone method)
"""

import os
import sys

# Add the parent directory to the path if necessary (depends on execution context)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Card:
    """
    A standard playing card.

    Attributes:
        rank: Integer rank (2-14, Ace=14).
        suit: Character suit ('h', 'd', 'c', 's').
    """

    # Constants for ranks and suits
    RANKS = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
        9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }
    SUITS = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
    INV_RANKS = {v: k for k, v in RANKS.items()}
    INV_SUITS = {v: k for k, v in SUITS.items()}

    def __init__(self, rank, suit):
        """ Initialize Card(rank, suit). """
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank
        self.suit = suit

    def get_rank_char(self):
        """ Get character for rank (e.g., 'A', 'K', 'T', '9'). """
        return self.RANKS.get(self.rank, '?')

    def get_suit_char(self):
        """ Get Unicode character for suit (e.g., ♥, ♦, ♣, ♠). """
        return self.SUITS.get(self.suit, '?')

    def clone(self):
        """ Creates a copy of the Card. """
        # Since rank/suit are immutable-like, creating new object is effective copy
        return Card(self.rank, self.suit)

    def __eq__(self, other):
        """ Check card equality. """
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        """ Compare card ranks for sorting (suitedness ignored). """
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank < other.rank

    def __hash__(self):
        """ Hash for use in sets/dicts. """
        return hash((self.rank, self.suit))

    def __str__(self):
        """ String representation (e.g., 'As', 'Th', '2d'). """
        return f"{self.get_rank_char()}{self.get_suit_char()}"

    def __repr__(self):
        """ Detailed representation for debugging. """
        return f"Card({self.rank}, '{self.suit}')"

# --- END OF FILE organized_poker_bot/game_engine/card.py ---
