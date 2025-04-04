"""
Enhanced card abstraction implementation for poker CFR.
This module provides advanced methods for abstracting card information to reduce the complexity
of the game state space while maintaining strategic relevance.
"""

import numpy as np
import random
import itertools
from sklearn.cluster import KMeans
import pickle
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
from organized_poker_bot.game_engine.card import Card

class EnhancedCardAbstraction:
    """
    Enhanced card abstraction techniques for poker CFR implementation.
    Implements advanced methods for abstracting card information to reduce the
    complexity of the game state space.
    """
    
    # Number of buckets for different rounds
    NUM_PREFLOP_BUCKETS = 20
    NUM_FLOP_BUCKETS = 50
    NUM_TURN_BUCKETS = 100
    NUM_RIVER_BUCKETS = 200
    
    # Clustering models for different rounds
    _preflop_model = None
    _flop_model = None
    _turn_model = None
    _river_model = None
    
    @staticmethod
    def get_preflop_abstraction(hole_cards):
        """
        Get the preflop abstraction for a hand using enhanced clustering.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            
        Returns:
            Integer representing the bucket (0-19, with 0 being strongest)
        """
        # Extract features for preflop hand
        features = EnhancedCardAbstraction._extract_preflop_features(hole_cards)
        
        # Initialize or load the clustering model
        if EnhancedCardAbstraction._preflop_model is None:
            model_path = os.path.join("models", "preflop_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    EnhancedCardAbstraction._preflop_model = pickle.load(f)
            else:
                # If no model exists, create a simple mapping based on hand strength
                return EnhancedCardAbstraction._simple_preflop_bucket(hole_cards)
        
        # Predict the cluster
        cluster = EnhancedCardAbstraction._preflop_model.predict([features])[0]
        
        return cluster
    
    @staticmethod
    def _simple_preflop_bucket(hole_cards):
        """
        Simple preflop bucketing based on hand strength.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            
        Returns:
            Integer representing the bucket (0-19, with 0 being strongest)
        """
        # Calculate a simple hand strength score
        rank1, rank2 = hole_cards[0].rank, hole_cards[1].rank
        suited = hole_cards[0].suit == hole_cards[1].suit
        
        # Score is based on:
        # 1. Higher ranks are better
        # 2. Pairs are better
        # 3. Suited hands are better
        # 4. Connected hands are better
        
        # Base score from ranks
        score = (rank1 + rank2) / 2
        
        # Bonus for pairs
        if rank1 == rank2:
            score += 7
        
        # Bonus for suited
        if suited:
            score += 2
        
        # Bonus for connectedness (closer ranks)
        connectedness = 14 - abs(rank1 - rank2)
        score += connectedness / 7
        
        # Normalize to 0-19 range (higher score = lower bucket number = stronger hand)
        normalized_score = min(19, max(0, 19 - int(score / 30 * 19)))
        
        return normalized_score
    
    @staticmethod
    def _extract_preflop_features(hole_cards):
        """
        Extract features for preflop hand clustering.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            
        Returns:
            List of features
        """
        rank1, rank2 = hole_cards[0].rank, hole_cards[1].rank
        suited = 1 if hole_cards[0].suit == hole_cards[1].suit else 0
        
        # Sort ranks
        high_rank = max(rank1, rank2)
        low_rank = min(rank1, rank2)
        
        # Calculate gap
        gap = high_rank - low_rank
        
        # Calculate potential (higher for connected, suited hands)
        potential = 0
        if suited:
            potential += 2
        if gap <= 4:  # Connected or small gap
            potential += (5 - gap)
        
        # Is pair
        is_pair = 1 if rank1 == rank2 else 0
        
        # Normalize ranks
        norm_high = (high_rank - 2) / 12  # 2-14 -> 0-1
        norm_low = (low_rank - 2) / 12
        
        # Return features
        return [norm_high, norm_low, suited, gap / 12, potential / 7, is_pair]
    
    @staticmethod
    def get_postflop_abstraction(hole_cards, community_cards):
        """
        Get the postflop abstraction for a hand using enhanced clustering.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            
        Returns:
            Integer representing the bucket
        """
        # Determine the round
        num_community = len(community_cards)
        
        if num_community == 3:  # Flop
            return EnhancedCardAbstraction._get_flop_abstraction(hole_cards, community_cards)
        elif num_community == 4:  # Turn
            return EnhancedCardAbstraction._get_turn_abstraction(hole_cards, community_cards)
        elif num_community == 5:  # River
            return EnhancedCardAbstraction._get_river_abstraction(hole_cards, community_cards)
        else:
            # Fallback to simple abstraction
            return EnhancedCardAbstraction._simple_preflop_bucket(hole_cards)
    
    @staticmethod
    def _get_flop_abstraction(hole_cards, community_cards):
        """
        Get the flop abstraction for a hand.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of 3 Card objects representing flop
            
        Returns:
            Integer representing the bucket (0-49)
        """
        # Extract features
        features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
        
        # Initialize or load the clustering model
        if EnhancedCardAbstraction._flop_model is None:
            model_path = os.path.join("models", "flop_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    EnhancedCardAbstraction._flop_model = pickle.load(f)
            else:
                # If no model exists, use a simple mapping based on hand strength and potential
                return EnhancedCardAbstraction._simple_postflop_bucket(
                    hole_cards, community_cards, EnhancedCardAbstraction.NUM_FLOP_BUCKETS)
        
        # Predict the cluster
        cluster = EnhancedCardAbstraction._flop_model.predict([features])[0]
        
        return cluster
    
    @staticmethod
    def _get_turn_abstraction(hole_cards, community_cards):
        """
        Get the turn abstraction for a hand.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of 4 Card objects representing flop and turn
            
        Returns:
            Integer representing the bucket (0-99)
        """
        # Extract features
        features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
        
        # Initialize or load the clustering model
        if EnhancedCardAbstraction._turn_model is None:
            model_path = os.path.join("models", "turn_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    EnhancedCardAbstraction._turn_model = pickle.load(f)
            else:
                # If no model exists, use a simple mapping based on hand strength and potential
                return EnhancedCardAbstraction._simple_postflop_bucket(
                    hole_cards, community_cards, EnhancedCardAbstraction.NUM_TURN_BUCKETS)
        
        # Predict the cluster
        cluster = EnhancedCardAbstraction._turn_model.predict([features])[0]
        
        return cluster
    
    @staticmethod
    def _get_river_abstraction(hole_cards, community_cards):
        """
        Get the river abstraction for a hand.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of 5 Card objects representing flop, turn, and river
            
        Returns:
            Integer representing the bucket (0-199)
        """
        # Extract features
        features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
        
        # Initialize or load the clustering model
        if EnhancedCardAbstraction._river_model is None:
            model_path = os.path.join("models", "river_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    EnhancedCardAbstraction._river_model = pickle.load(f)
            else:
                # If no model exists, use a simple mapping based on hand strength
                return EnhancedCardAbstraction._simple_postflop_bucket(
                    hole_cards, community_cards, EnhancedCardAbstraction.NUM_RIVER_BUCKETS)
        
        # Predict the cluster
        cluster = EnhancedCardAbstraction._river_model.predict([features])[0]
        
        return cluster
    
    @staticmethod
    def _extract_postflop_features(hole_cards, community_cards):
        """
        Extract features for postflop hand clustering.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            
        Returns:
            List of features
        """
        # Current hand strength
        hand_strength = EnhancedCardAbstraction._calculate_hand_strength(hole_cards, community_cards)
        
        # Hand potential (AIPF)
        ahead_improve, behind_improve = EnhancedCardAbstraction._calculate_hand_potential(
            hole_cards, community_cards)
        
        # Hand type features
        has_pair, has_two_pair, has_trips, has_straight_draw, has_flush_draw = \
            EnhancedCardAbstraction._extract_hand_type_features(hole_cards, community_cards)
        
        # Board texture features
        board_pair, board_suited, board_connected = \
            EnhancedCardAbstraction._extract_board_features(community_cards)
        
        # Combine features
        features = [
            hand_strength,
            ahead_improve,
            behind_improve,
            has_pair,
            has_two_pair,
            has_trips,
            has_straight_draw,
            has_flush_draw,
            board_pair,
            board_suited,
            board_connected
        ]
        
        return features
    
    @staticmethod
    def _calculate_hand_strength(hole_cards, community_cards):
        """
        Calculate the current hand strength.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            
        Returns:
            Float between 0 and 1 representing hand strength
        """
        # Use HandEvaluator to get the hand rank
        all_cards = hole_cards + community_cards
        hand_rank = HandEvaluator.evaluate_hand(all_cards)
        
        # Normalize to 0-1 range
        # Straight flush (8) is the highest, high card (0) is the lowest
        normalized_rank = hand_rank / 8
        
        return normalized_rank
    
    @staticmethod
    def _calculate_hand_potential(hole_cards, community_cards):
        """
        Calculate the hand potential using AIPF (Ahead, Improve, Finish) method.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            
        Returns:
            Tuple of (ahead_improve, behind_improve) representing potential
        """
        # For simplicity, we'll use a Monte Carlo approach with a limited number of simulations
        num_simulations = 100
        
        # Current hand strength
        current_hand = hole_cards + community_cards
        current_rank = HandEvaluator.evaluate_hand(current_hand)
        
        # Cards left in the deck
        all_cards = []
        for suit in range(4):
            for rank in range(2, 15):
                all_cards.append(Card(rank, suit))
        
        deck = [card for card in all_cards if card not in hole_cards and card not in community_cards]
        
        # Number of cards to draw
        cards_to_draw = 5 - len(community_cards)
        
        # If we're already at the river, no potential to improve
        if cards_to_draw == 0:
            return 0, 0
        
        # Track ahead/behind stats
        ahead_improve = 0
        behind_improve = 0
        
        # Run simulations
        for _ in range(num_simulations):
            # Sample opponent hole cards
            opponent_hole = random.sample(deck, 2)
            remaining_deck = [card for card in deck if card not in opponent_hole]
            
            # Sample remaining community cards
            additional_community = random.sample(remaining_deck, cards_to_draw)
            final_community = community_cards + additional_community
            
            # Evaluate hands
            player_hand = hole_cards + final_community
            opponent_hand = opponent_hole + final_community
            
            player_rank = HandEvaluator.evaluate_hand(player_hand)
            opponent_rank = HandEvaluator.evaluate_hand(opponent_hand)
            
            # Compare current vs final
            currently_ahead = current_rank > HandEvaluator.evaluate_hand(opponent_hole + community_cards)
            ends_ahead = player_rank > opponent_rank
            
            # Update stats
            if currently_ahead and ends_ahead:
                # Stayed ahead
                pass
            elif currently_ahead and not ends_ahead:
                # Was ahead but fell behind
                behind_improve += 1
            elif not currently_ahead and ends_ahead:
                # Was behind but improved
                ahead_improve += 1
            else:
                # Stayed behind
                pass
        
        # Normalize
        ahead_improve = ahead_improve / num_simulations
        behind_improve = behind_improve / num_simulations
        
        return ahead_improve, behind_improve
    
    @staticmethod
    def _extract_hand_type_features(hole_cards, community_cards):
        """
        Extract features about the hand type.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            
        Returns:
            Tuple of binary features (has_pair, has_two_pair, has_trips, has_straight_draw, has_flush_draw)
        """
        all_cards = hole_cards + community_cards
        
        # Count ranks
        rank_counts = {}
        for card in all_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        # Count suits
        suit_counts = {}
        for card in all_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Check for pairs, trips
        has_pair = 0
        has_two_pair = 0
        has_trips = 0
        
        pairs = 0
        for count in rank_counts.values():
            if count == 2:
                pairs += 1
            elif count == 3:
                has_trips = 1
        
        if pairs >= 1:
            has_pair = 1
        if pairs >= 2:
            has_two_pair = 1
        
        # Check for straight draw
        ranks = sorted([card.rank for card in all_cards])
        has_straight_draw = 0
        
        # Check for open-ended straight draw
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] == 3:  # 4 consecutive ranks with 1 gap
                has_straight_draw = 1
                break
        
        # Check for flush draw
        has_flush_draw = 0
        for count in suit_counts.values():
            if count >= 4:
                has_flush_draw = 1
                break
        
        return has_pair, has_two_pair, has_trips, has_straight_draw, has_flush_draw
    
    @staticmethod
    def _extract_board_features(community_cards):
        """
        Extract features about the board texture.
        
        Args:
            community_cards: List of Card objects representing community cards
            
        Returns:
            Tuple of features (board_pair, board_suited, board_connected)
        """
        if not community_cards:
            return 0, 0, 0
        
        # Count ranks
        rank_counts = {}
        for card in community_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        # Count suits
        suit_counts = {}
        for card in community_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Check for paired board
        board_pair = 0
        for count in rank_counts.values():
            if count >= 2:
                board_pair = 1
                break
        
        # Check for suited board
        board_suited = 0
        for count in suit_counts.values():
            if count >= 3:
                board_suited = 1
                break
        
        # Check for connected board
        ranks = sorted([card.rank for card in community_cards])
        board_connected = 0
        
        for i in range(len(ranks) - 1):
            if ranks[i+1] - ranks[i] <= 2:  # Connected or 1-gap
                board_connected = 1
                break
        
        return board_pair, board_suited, board_connected
    
    @staticmethod
    def _simple_postflop_bucket(hole_cards, community_cards, num_buckets):
        """
        Simple postflop bucketing based on hand strength and potential.
        
        Args:
            hole_cards: List of two Card objects representing hole cards
            community_cards: List of Card objects representing community cards
            num_buckets: Number of buckets to use
            
        Returns:
            Integer representing the bucket (0 to num_buckets-1, with 0 being strongest)
        """
        # Calculate hand strength
        hand_strength = EnhancedCardAbstraction._calculate_hand_strength(hole_cards, community_cards)
        
        # Calculate potential
        ahead_improve, behind_improve = EnhancedCardAbstraction._calculate_hand_potential(
            hole_cards, community_cards)
        
        # Combined score (70% strength, 30% potential)
        score = 0.7 * hand_strength + 0.3 * (ahead_improve + behind_improve) / 2
        
        # Normalize to bucket range (higher score = lower bucket number = stronger hand)
        normalized_bucket = min(num_buckets - 1, max(0, int((1 - score) * num_buckets)))
        
        return normalized_bucket
    
    @staticmethod
    def train_models(training_data_path=None):
        """
        Train clustering models for card abstraction.
        
        Args:
            training_data_path: Path to training data (if None, generate synthetic data)
            
        Returns:
            Tuple of trained models (preflop_model, flop_model, turn_model, river_model)
        """
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train preflop model
        preflop_model = KMeans(n_clusters=EnhancedCardAbstraction.NUM_PREFLOP_BUCKETS, random_state=42)
        
        # Generate synthetic preflop data if no training data provided
        if training_data_path is None:
            preflop_data = EnhancedCardAbstraction._generate_synthetic_preflop_data()
        else:
            # Load preflop data from file
            with open(os.path.join(training_data_path, "preflop_data.pkl"), 'rb') as f:
                preflop_data = pickle.load(f)
        
        # Fit preflop model
        preflop_model.fit(preflop_data)
        
        # Save preflop model
        with open(os.path.join("models", "preflop_model.pkl"), 'wb') as f:
            pickle.dump(preflop_model, f)
        
        # Train postflop models (simplified for now)
        flop_model = KMeans(n_clusters=EnhancedCardAbstraction.NUM_FLOP_BUCKETS, random_state=42)
        turn_model = KMeans(n_clusters=EnhancedCardAbstraction.NUM_TURN_BUCKETS, random_state=42)
        river_model = KMeans(n_clusters=EnhancedCardAbstraction.NUM_RIVER_BUCKETS, random_state=42)
        
        # Generate synthetic postflop data if no training data provided
        if training_data_path is None:
            flop_data = EnhancedCardAbstraction._generate_synthetic_postflop_data(3)  # 3 community cards
            turn_data = EnhancedCardAbstraction._generate_synthetic_postflop_data(4)  # 4 community cards
            river_data = EnhancedCardAbstraction._generate_synthetic_postflop_data(5)  # 5 community cards
        else:
            # Load postflop data from file
            with open(os.path.join(training_data_path, "flop_data.pkl"), 'rb') as f:
                flop_data = pickle.load(f)
            with open(os.path.join(training_data_path, "turn_data.pkl"), 'rb') as f:
                turn_data = pickle.load(f)
            with open(os.path.join(training_data_path, "river_data.pkl"), 'rb') as f:
                river_data = pickle.load(f)
        
        # Fit postflop models
        flop_model.fit(flop_data)
        turn_model.fit(turn_data)
        river_model.fit(river_data)
        
        # Save postflop models
        with open(os.path.join("models", "flop_model.pkl"), 'wb') as f:
            pickle.dump(flop_model, f)
        with open(os.path.join("models", "turn_model.pkl"), 'wb') as f:
            pickle.dump(turn_model, f)
        with open(os.path.join("models", "river_model.pkl"), 'wb') as f:
            pickle.dump(river_model, f)
        
        return preflop_model, flop_model, turn_model, river_model
    
    @staticmethod
    def _generate_synthetic_preflop_data():
        """
        Generate synthetic preflop data for training.
        
        Returns:
            List of feature vectors
        """
        data = []
        
        # Generate all possible hole card combinations
        all_cards = []
        for suit in range(4):
            for rank in range(2, 15):
                all_cards.append(Card(rank, suit))
        
        # Sample a subset of combinations to keep training manageable
        num_samples = 1000
        hole_card_combinations = random.sample(list(itertools.combinations(all_cards, 2)), num_samples)
        
        # Extract features for each combination
        for hole_cards in hole_card_combinations:
            features = EnhancedCardAbstraction._extract_preflop_features(hole_cards)
            data.append(features)
        
        return data
    
    @staticmethod
    def _generate_synthetic_postflop_data(num_community):
        """
        Generate synthetic postflop data for training.
        
        Args:
            num_community: Number of community cards (3 for flop, 4 for turn, 5 for river)
            
        Returns:
            List of feature vectors
        """
        data = []
        
        # Generate all possible cards
        all_cards = []
        for suit in range(4):
            for rank in range(2, 15):
                all_cards.append(Card(rank, suit))
        
        # Sample a subset of combinations to keep training manageable
        num_samples = 1000
        
        for _ in range(num_samples):
            # Sample cards
            sampled_cards = random.sample(all_cards, 2 + num_community)  # 2 hole cards + community cards
            hole_cards = sampled_cards[:2]
            community_cards = sampled_cards[2:]
            
            # Extract features
            features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
            data.append(features)
        
        return data
