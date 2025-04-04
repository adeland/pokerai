# --- START OF FILE organized_poker_bot/cfr/card_abstraction.py ---
"""
Implementation of card abstraction techniques for poker CFR.
This module provides methods for abstracting card information to reduce the complexity
of the game state space while maintaining strategic relevance.
(Refactored V3: Add board features to postflop abstraction key)
"""

import numpy as np
import random
import itertools
from collections import Counter # Added Counter import
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
from organized_poker_bot.game_engine.card import Card

class CardAbstraction:
    """
    Card abstraction techniques for poker CFR implementation.
    Implements various methods for abstracting card information to reduce the
    complexity of the game state space.
    """

    # Preflop hand buckets (169 starting hands grouped into 10 buckets by strength)
    PREFLOP_BUCKETS = {
        # ... (Keep the PREFLOP_BUCKETS dictionary exactly as it was) ...
        0:["AA", "KK", "QQ", "AKs", "AKo"], 1:["JJ", "TT", "99", "AQs", "AQo", "AJs", "ATs", "KQs"],
        2:["88", "77", "AJo", "ATo", "KQo", "KJs", "KTs", "QJs", "QTs", "JTs"], 3:["66", "55", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s","KJo", "KTo", "QJo", "JTo", "T9s", "98s", "87s", "76s", "65s"],
        4:["44", "33", "22", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s","Q9s", "Q8s", "J9s", "T8s", "97s", "86s", "75s", "54s"],
        5:["A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o", "Q7s", "Q6s", "Q5s","Q4s", "Q3s", "Q2s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s"],
        6:["K9o", "K8o", "K7o", "K6o", "K5o", "K4o", "K3o", "K2o", "Q9o", "Q8o", "Q7o","T7s", "T6s", "T5s", "T4s", "T3s", "T2s", "96s", "95s", "94s", "93s", "92s"],
        7:["Q6o", "Q5o", "Q4o", "Q3o", "Q2o", "J9o", "J8o", "J7o", "T9o", "T8o", "98o","85s", "84s", "83s", "82s", "74s", "73s", "72s", "64s", "63s", "62s", "53s", "52s", "43s", "42s", "32s"],
        8:["J6o", "J5o", "J4o", "J3o", "J2o", "T7o", "T6o", "T5o", "T4o", "97o", "96o", "87o", "86o", "76o", "65o"],
        9:["T3o", "T2o", "95o", "94o", "93o", "92o", "85o", "84o", "83o", "82o", "75o", "74o", "73o", "72o","64o", "63o", "62o", "54o", "53o", "52o", "43o", "42o", "32o"]
    }

    @staticmethod
    def get_preflop_abstraction(hole_cards):
        """ Gets preflop bucket (0-9) based on PREFLOP_BUCKETS. """
        if not hole_cards or len(hole_cards) != 2: return 9
        hand_repr = CardAbstraction._get_hand_representation(hole_cards)
        for bucket, hands in CardAbstraction.PREFLOP_BUCKETS.items():
            if hand_repr in hands: return bucket
        return 9

    @staticmethod
    def _get_hand_representation(hole_cards):
        """ Converts two hole cards to standard string like "AKs", "T9o", "QQ". """
        if len(hole_cards) != 2: return "?"
        ranks_int = sorted([card.rank for card in hole_cards], reverse=True)
        rank_chars = [Card.RANKS.get(r, '?') for r in ranks_int]
        if rank_chars[0] == rank_chars[1]: return rank_chars[0] + rank_chars[1]
        else: suited = hole_cards[0].suit == hole_cards[1].suit; suffix = "s" if suited else "o"; return rank_chars[0] + rank_chars[1] + suffix

    # --- NEW: Board Feature Extraction ---
    @staticmethod
    def get_board_features(community_cards):
        """ Extracts deterministic board features suitable for abstraction keys. """
        if not community_cards or len(community_cards) < 3:
             return (0, 'n') # No pair, No significant suit

        ranks = [c.rank for c in community_cards]
        suits = [c.suit for c in community_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Paired board? (Value 1 if any pair, 2 if trips/quads/two pair - simplify to 0/1 for now)
        is_paired = 1 if any(count >= 2 for count in rank_counts.values()) else 0

        # Flush suit present? (Value is suit char if >=3 cards, else 'n')
        flush_suit = 'n'
        for suit, count in suit_counts.items():
            if count >= 3:
                flush_suit = suit
                break # Take first found flush suit

        # Could add straight features later (e.g., number of connectors, gaps)

        return (is_paired, flush_suit)


    # --- MODIFIED METHOD ---
    @staticmethod
    def get_postflop_abstraction(hole_cards, community_cards):
        """
        Get the postflop abstraction using deterministic hand strength AND board features.

        Args:
            hole_cards: List of two Card objects
            community_cards: List of Card objects (3, 4, or 5)

        Returns:
            tuple: (strength_bucket, board_paired_feature, board_flush_suit_feature)
                   Returns (9, 0, 'n') if invalid inputs.
        """
        if not hole_cards or len(hole_cards) != 2 or not community_cards or len(community_cards) < 3:
            return (9, 0, 'n') # Default/invalid bucket + features

        try:
            # Calculate hand strength bucket (deterministic)
            normalized_strength = CardAbstraction._calculate_exact_hand_strength(hole_cards, community_cards)
            clamped_strength = max(0.0, min(1.0, normalized_strength))
            strength_bucket = min(9, int((1.0 - clamped_strength) * 10))

            # Calculate board features
            board_paired, board_flush_suit = CardAbstraction.get_board_features(community_cards)

            return (strength_bucket, board_paired, board_flush_suit)

        except Exception as e:
            # print(f"WARN: Error during postflop abstraction V3: {e}")
            return (9, 0, 'n') # Return worst bucket on error


    @staticmethod
    def _calculate_exact_hand_strength(hole_cards, community_cards):
        """ Calculates normalized strength (0-1) based on HandEvaluator rank. """
        try:
            all_cards = hole_cards + community_cards
            if len(all_cards) < 5: return 0.0
            hand_evaluation_result = HandEvaluator.evaluate_hand(all_cards)
            hand_type_value = hand_evaluation_result[0]
            min_rank = min(HandEvaluator.HAND_TYPES.values()); max_rank = max(HandEvaluator.HAND_TYPES.values())
            if max_rank == min_rank: return 0.5
            normalized_strength = (hand_type_value - min_rank) / (max_rank - min_rank)
            return normalized_strength
        except Exception: return 0.0

    # --- calculate_equity and _monte_carlo_equity remain unchanged, ---
    # --- not used by get_postflop_abstraction for info keys. ---
    @staticmethod
    def calculate_equity(hole_cards, community_cards, num_samples=100):
        """ Calculates equity using Monte Carlo. """
        if not hole_cards or len(hole_cards) != 2: return 0.0
        return CardAbstraction._monte_carlo_equity(hole_cards, community_cards or [], num_samples)

    @staticmethod
    def _monte_carlo_equity(hole_cards, community_cards, num_samples=100):
        """ Monte Carlo equity simulation (unchanged from V3). """
        used_cards = hole_cards + community_cards; unique_used_cards = []
        for card in used_cards:
            if card not in unique_used_cards: unique_used_cards.append(card)
        deck_list = [Card(r, s) for r in range(2, 15) for s in ['h','d','c','s'] if Card(r,s) not in unique_used_cards]
        num_needed = 5 - len(community_cards);
        if num_needed < 0: return 0.0;
        if len(deck_list) < num_needed + 2: return 0.5;
        wins = 0.0; ties = 0.0; valid_sims = 0
        for _ in range(num_samples):
            try:
                deck_sample = deck_list[:]; random.shuffle(deck_sample); opponent_hole = deck_sample[:2]
                remaining_community = deck_sample[2 : 2 + num_needed]; final_community = community_cards + remaining_community
                hero_all_cards = hole_cards + final_community; opp_all_cards = opponent_hole + final_community
                if len(hero_all_cards) < 5 or len(opp_all_cards) < 5: continue
                hero_strength_eval = HandEvaluator.evaluate_hand(hero_all_cards)
                opponent_strength_eval = HandEvaluator.evaluate_hand(opp_all_cards)
                if hero_strength_eval > opponent_strength_eval: wins += 1.0
                elif hero_strength_eval == opponent_strength_eval: ties += 1.0
                valid_sims += 1
            except Exception: continue
        if valid_sims == 0: return 0.5;
        equity = (wins + (ties / 2.0)) / valid_sims; return equity

# Example usage (remains same)
if __name__ == '__main__':
    card1 = Card(14, 's'); card2 = Card(13, 's'); hole = [card1, card2]
    pre_bucket = CardAbstraction.get_preflop_abstraction(hole); print(f"AKs Preflop: {pre_bucket}")
    card1 = Card(7, 'h'); card2 = Card(2, 'd'); hole = [card1, card2]
    pre_bucket = CardAbstraction.get_preflop_abstraction(hole); print(f"72o Preflop: {pre_bucket}")
    hole = [Card(14, 's'), Card(10, 's')]; comm = [Card(13, 's'), Card(4, 's'), Card(2, 'h')] # Flush Draw
    post_abs = CardAbstraction.get_postflop_abstraction(hole, comm); print(f"AsTs on Ks4s2h Postflop Abstraction: {post_abs}") # (bucket, paired, flush_suit)
    comm = [Card(13, 'c'), Card(4, 'd'), Card(2, 'h'), Card(8, 's'), Card(9,'s')] # River board, Ace high
    post_abs = CardAbstraction.get_postflop_abstraction(hole, comm); print(f"AsTs on Kc4d2h8s9s Postflop Abstraction: {post_abs}") # Ace high, weak bucket
    hole_weak = [Card(7,'d'), Card(2,'h')]; comm = [Card(12,'s'), Card(7,'h'), Card(2,'s')] # Two pair
    post_abs = CardAbstraction.get_postflop_abstraction(hole_weak, comm); print(f"7d2h on Qs7h2s Postflop Abstraction: {post_abs}") # Two pair, medium bucket, board features

# --- END OF FILE organized_poker_bot/cfr/card_abstraction.py ---
