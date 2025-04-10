"""
Hand evaluation module for poker hands.
"""

import os
import sys
from collections import Counter

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HandEvaluator:
    """
    A class for evaluating poker hands.
    
    This evaluator determines the best 5-card poker hand from a set of cards
    and assigns a numerical rank to the hand for comparison.
    
    Hand rankings (from highest to lowest):
    1. Royal Flush
    2. Straight Flush
    3. Four of a Kind
    4. Full House
    5. Flush
    6. Straight
    7. Three of a Kind
    8. Two Pair
    9. One Pair
    10. High Card
    """
    
    # Hand type values (higher is better)
    HAND_TYPES = {
        'high_card': 0,
        'pair': 1,
        'two_pair': 2,
        'three_of_a_kind': 3,
        'straight': 4,
        'flush': 5,
        'full_house': 6,
        'four_of_a_kind': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }
    
    @staticmethod
    def evaluate_hand(cards):
        """
        Evaluate a poker hand and return its rank.
        
        Args:
            cards (list): A list of Card objects (5-7 cards)
            
        Returns:
            tuple: (hand_type_value, [kickers]) where hand_type_value is an integer
                  representing the hand type and kickers is a list of card ranks
                  used for breaking ties
        """
        if len(cards) < 5:
            raise ValueError("At least 5 cards are required for hand evaluation")
        
        # Find the best 5-card hand if more than 5 cards are provided
        if len(cards) > 5:
            return HandEvaluator._find_best_hand(cards)
        
        # Sort cards by rank (high to low)
        sorted_cards = sorted(cards, key=lambda card: card.rank, reverse=True)
        
        # Check for flush
        is_flush = len(set(card.suit for card in cards)) == 1
        
        # Check for straight
        ranks = [card.rank for card in sorted_cards]
        # Handle Ace as low card (A-5-4-3-2)
        if set(ranks) == {14, 5, 4, 3, 2}:
            is_straight = True
            # Move Ace to the end for proper kicker order in a wheel straight
            ranks = [5, 4, 3, 2, 1]
        else:
            is_straight = (max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5)
        
        # Count occurrences of each rank
        rank_counts = Counter(ranks)
        count_values = sorted(rank_counts.values(), reverse=True)
        
        # Determine hand type
        if is_straight and is_flush:
            if ranks[0] == 14 and ranks[1] == 13:  # A, K, Q, J, 10
                return (HandEvaluator.HAND_TYPES['royal_flush'], [])
            else:
                return (HandEvaluator.HAND_TYPES['straight_flush'], ranks)
        
        if count_values[0] == 4:  # Four of a kind
            quads_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return (HandEvaluator.HAND_TYPES['four_of_a_kind'], [quads_rank, kicker])
        
        if count_values[0] == 3 and count_values[1] == 2:  # Full house
            trips_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return (HandEvaluator.HAND_TYPES['full_house'], [trips_rank, pair_rank])
        
        if is_flush:
            return (HandEvaluator.HAND_TYPES['flush'], ranks)
        
        if is_straight:
            return (HandEvaluator.HAND_TYPES['straight'], ranks)
        
        if count_values[0] == 3:  # Three of a kind
            trips_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return (HandEvaluator.HAND_TYPES['three_of_a_kind'], [trips_rank] + kickers)
        
        if count_values[0] == 2 and count_values[1] == 2:  # Two pair
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return (HandEvaluator.HAND_TYPES['two_pair'], pairs + [kicker])
        
        if count_values[0] == 2:  # One pair
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return (HandEvaluator.HAND_TYPES['pair'], [pair_rank] + kickers)
        
        # High card
        return (HandEvaluator.HAND_TYPES['high_card'], ranks)
    
    @staticmethod
    def _find_best_hand(cards):
        """
        Find the best 5-card hand from a set of cards.
        
        Args:
            cards (list): A list of Card objects (more than 5 cards)
            
        Returns:
            tuple: The best hand evaluation as returned by evaluate_hand
        """
        from itertools import combinations
        
        best_hand = None
        best_value = (-1, [])
        
        # Check all possible 5-card combinations
        for five_cards in combinations(cards, 5):
            hand_value = HandEvaluator.evaluate_hand(list(five_cards))
            if hand_value > best_value:
                best_value = hand_value
                best_hand = five_cards
        
        return best_value
    
    @staticmethod
    def evaluate_hand_with_type(cards):
        """
        Evaluate a poker hand and return its rank along with a human-readable hand type.
        
        Args:
            cards (list): A list of Card objects (5-7 cards)
            
        Returns:
            tuple: (hand_value, hand_type_string) where hand_value is the numerical
                  value of the hand and hand_type_string is a human-readable description
        """
        hand_value = HandEvaluator.evaluate_hand(cards)
        hand_type_string = HandEvaluator.hand_type_to_string(hand_value)
        
        return hand_value, hand_type_string
    
    @staticmethod
    def hand_type_to_string(hand_value):
        """
        Convert a hand value to a human-readable string.
        
        Args:
            hand_value (tuple): A hand value as returned by evaluate_hand
            
        Returns:
            str: A string describing the hand type
        """
        hand_type = hand_value[0]
        
        for name, value in HandEvaluator.HAND_TYPES.items():
            if value == hand_type:
                return name.replace('_', ' ').title()
        
        return "Unknown Hand"
        
    @staticmethod
    def compare_hands(hand1, hand2):
        """
        Compare two poker hands.
        
        Args:
            hand1 (list): A list of Card objects for the first hand
            hand2 (list): A list of Card objects for the second hand
            
        Returns:
            int: 1 if hand1 wins, -1 if hand2 wins, 0 if it's a tie
        """
        hand1_value = HandEvaluator.evaluate_hand(hand1)
        hand2_value = HandEvaluator.evaluate_hand(hand2)
        
        if hand1_value > hand2_value:
            return 1
        elif hand1_value < hand2_value:
            return -1
        else:
            return 0
