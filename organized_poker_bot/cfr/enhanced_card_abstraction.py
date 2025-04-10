# --- START OF FILE organized_poker_bot/cfr/enhanced_card_abstraction.py ---
"""
Enhanced card abstraction implementation for poker CFR.
This module provides advanced methods for abstracting card information to reduce the complexity
of the game state space while maintaining strategic relevance.
(Refactored V2: Fixed TypeError in _calculate_hand_strength)
"""

import numpy as np
import random
import itertools
from sklearn.cluster import KMeans
import pickle
import os
import sys
from collections import Counter # Ensure Counter is imported

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

    # Clustering models for different rounds (Paths for loading)
    _preflop_model = None
    _flop_model = None
    _turn_model = None
    _river_model = None
    _preflop_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'preflop_model.pkl')
    _flop_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'flop_model.pkl')
    _turn_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'turn_model.pkl')
    _river_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'river_model.pkl')

    # --- Preflop Abstraction Methods ---
    @staticmethod
    def get_preflop_abstraction(hole_cards):
        """ Gets preflop abstraction using clustering model or simple fallback. """
        if not hole_cards or len(hole_cards) != 2: return EnhancedCardAbstraction.NUM_PREFLOP_BUCKETS - 1 # Worst bucket if invalid

        # Lazy load model
        if EnhancedCardAbstraction._preflop_model is None:
            try:
                abs_path = os.path.abspath(EnhancedCardAbstraction._preflop_model_path)
                #print(f"DEBUG: Trying to load preflop model from: {abs_path}") # Debug print
                if os.path.exists(abs_path):
                    with open(abs_path, 'rb') as f:
                        EnhancedCardAbstraction._preflop_model = pickle.load(f)
                        #print("DEBUG: Preflop model loaded successfully.")
                else:
                    #print("DEBUG: Preflop model file not found.")
                    pass # Fallback will be used
            except Exception as e:
                print(f"WARN: Failed to load preflop model: {e}. Using simple fallback.")

        if EnhancedCardAbstraction._preflop_model:
            try:
                features = EnhancedCardAbstraction._extract_preflop_features(hole_cards)
                # Model expects 2D array
                cluster = EnhancedCardAbstraction._preflop_model.predict(np.array(features).reshape(1, -1))[0]
                return cluster
            except Exception as e:
                print(f"WARN: Error predicting preflop cluster: {e}. Using simple fallback.")
                return EnhancedCardAbstraction._simple_preflop_bucket(hole_cards)
        else:
             return EnhancedCardAbstraction._simple_preflop_bucket(hole_cards) # Use fallback if no model loaded

    @staticmethod
    def _simple_preflop_bucket(hole_cards):
        """ Simple preflop bucketing based on hand strength. """
        rank1, rank2 = hole_cards[0].rank, hole_cards[1].rank
        suited = hole_cards[0].suit == hole_cards[1].suit
        score = (rank1 + rank2) / 2
        if rank1 == rank2: score += 7 # Pair bonus
        if suited: score += 2 # Suited bonus
        connectedness = 14 - abs(rank1 - rank2); score += connectedness / 7 # Connect bonus
        # Scale score approx 0-30 -> bucket 19-0 (inverted)
        num_buckets = EnhancedCardAbstraction.NUM_PREFLOP_BUCKETS
        normalized_score = (score / 30.0) * (num_buckets -1) # Normalize to ~[0, num_buckets-1]
        bucket = num_buckets - 1 - int(normalized_score) # Invert: higher score -> lower bucket
        return max(0, min(num_buckets - 1, bucket)) # Clamp to range

    @staticmethod
    def _extract_preflop_features(hole_cards):
        """ Extract features for preflop hand clustering. """
        rank1, rank2 = hole_cards[0].rank, hole_cards[1].rank
        suited = 1 if hole_cards[0].suit == hole_cards[1].suit else 0
        high_rank = max(rank1, rank2); low_rank = min(rank1, rank2); gap = high_rank - low_rank
        potential = (2 if suited else 0) + (max(0, 5 - gap)); is_pair = 1 if rank1 == rank2 else 0
        norm_high = (high_rank - 2) / 12; norm_low = (low_rank - 2) / 12 # Scale rank 2-14 -> 0-1
        return [norm_high, norm_low, suited, gap / 12, potential / 7, is_pair]

    # --- Postflop Abstraction Methods ---
    @staticmethod
    def get_postflop_abstraction(hole_cards, community_cards):
        """ Gets postflop abstraction using model or fallback based on round. """
        if not hole_cards or len(hole_cards)!=2 or not community_cards or len(community_cards) < 3:
             num_buckets = EnhancedCardAbstraction.NUM_FLOP_BUCKETS # Default if invalid
             if len(community_cards) == 4: num_buckets = EnhancedCardAbstraction.NUM_TURN_BUCKETS
             elif len(community_cards) == 5: num_buckets = EnhancedCardAbstraction.NUM_RIVER_BUCKETS
             return num_buckets - 1 # Return worst bucket

        num_community = len(community_cards)
        model = None; model_path = None; num_buckets = EnhancedCardAbstraction.NUM_FLOP_BUCKETS # Default
        round_name = "Flop"

        if num_community == 3: # Flop
            if EnhancedCardAbstraction._flop_model is None: EnhancedCardAbstraction._lazy_load_model('_flop_model', EnhancedCardAbstraction._flop_model_path)
            model = EnhancedCardAbstraction._flop_model
            num_buckets = EnhancedCardAbstraction.NUM_FLOP_BUCKETS
            round_name = "Flop"
        elif num_community == 4: # Turn
            if EnhancedCardAbstraction._turn_model is None: EnhancedCardAbstraction._lazy_load_model('_turn_model', EnhancedCardAbstraction._turn_model_path)
            model = EnhancedCardAbstraction._turn_model
            num_buckets = EnhancedCardAbstraction.NUM_TURN_BUCKETS
            round_name = "Turn"
        elif num_community == 5: # River
            if EnhancedCardAbstraction._river_model is None: EnhancedCardAbstraction._lazy_load_model('_river_model', EnhancedCardAbstraction._river_model_path)
            model = EnhancedCardAbstraction._river_model
            num_buckets = EnhancedCardAbstraction.NUM_RIVER_BUCKETS
            round_name = "River"
        else:
             return EnhancedCardAbstraction._simple_preflop_bucket(hole_cards) # Should not happen if check above works

        if model:
            try:
                 features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
                 cluster = model.predict(np.array(features).reshape(1, -1))[0]
                 return cluster
            except Exception as e:
                 print(f"WARN: Error predicting {round_name} cluster: {e}. Using simple fallback.")
                 return EnhancedCardAbstraction._simple_postflop_bucket(hole_cards, community_cards, num_buckets)
        else:
             # Fallback if model still not loaded
             return EnhancedCardAbstraction._simple_postflop_bucket(hole_cards, community_cards, num_buckets)

    @staticmethod
    def _lazy_load_model(model_attr_name, model_path):
         """ Helper to load a model file on demand. """
         try:
              abs_path = os.path.abspath(model_path)
              #print(f"DEBUG: Trying lazy load: {abs_path}") # Debug print
              if os.path.exists(abs_path):
                   with open(abs_path, 'rb') as f:
                        setattr(EnhancedCardAbstraction, model_attr_name, pickle.load(f))
                        #print(f"DEBUG: Loaded model for {model_attr_name}")
              # else: print(f"DEBUG: Model file not found: {abs_path}")
         except Exception as e:
              print(f"WARN: Failed lazy load {model_attr_name}: {e}")

    # *** MODIFIED _calculate_hand_strength ***
    @staticmethod
    def _calculate_hand_strength(hole_cards, community_cards):
        """ Calculate normalized hand strength [0, 1] using HandEvaluator rank. """
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5: return 0.0

        try:
            hand_eval_result = HandEvaluator.evaluate_hand(all_cards)

            # Extract numeric type value from tuple (hand_type_val, [kickers])
            if isinstance(hand_eval_result, tuple):
                hand_type_value = hand_eval_result[0]
            else: # Fallback if somehow not a tuple
                hand_type_value = float(hand_eval_result)

            # Determine min/max rank from HandEvaluator constants
            try:
                 type_values = HandEvaluator.HAND_TYPES.values()
                 min_rank = min(type_values) # e.g., 0
                 max_rank = max(type_values) # e.g., 9
            except AttributeError: # Fallback if HAND_TYPES not defined/accessible
                 print("WARN: Cannot access HandEvaluator.HAND_TYPES, using ranks 0-9.")
                 min_rank = 0; max_rank = 9

            if max_rank <= min_rank: return 0.5 # Avoid division by zero

            # Normalize the numeric rank value
            normalized_rank = (hand_type_value - min_rank) / (max_rank - min_rank)
            return max(0.0, min(1.0, normalized_rank)) # Clamp to [0, 1]

        except Exception as e:
            print(f"Error calculating hand strength: {e}")
            return 0.0 # Return worst strength on error

    # --- Other methods _calculate_hand_potential, feature extraction, _simple_postflop_bucket etc remain unchanged ---
    # --- They were not the source of the TypeError ---

    @staticmethod
    def _calculate_hand_potential(hole_cards, community_cards):
        """ Estimate hand potential AIPF via Monte Carlo. """
        num_simulations = 50 # Reduced default for faster testing/fallback
        current_hand = hole_cards + community_cards
        if len(current_hand) < 5: return 0.0, 0.0 # Cannot eval hand yet
        current_rank = HandEvaluator.evaluate_hand(current_hand)

        used_cards = frozenset(current_hand)
        deck_list = [Card(r, s) for r in range(2, 15) for s in ['h','d','c','s'] if Card(r,s) not in used_cards]

        cards_to_draw = 5 - len(community_cards)
        if cards_to_draw <= 0 or len(deck_list) < cards_to_draw + 2: return 0.0, 0.0

        ahead_wins, ahead_losses = 0, 0
        behind_wins, behind_losses = 0, 0
        ahead_count, behind_count = 0, 0

        for _ in range(num_simulations):
             try:
                 deck_sample = deck_list[:]; random.shuffle(deck_sample); opponent_hole = deck_sample[:2]
                 remaining_deck = deck_sample[2:]
                 additional_community = remaining_deck[:cards_to_draw]
                 final_community = community_cards + additional_community

                 player_hand_final = hole_cards + final_community
                 opponent_hand_final = opponent_hole + final_community

                 if len(player_hand_final) < 5 or len(opponent_hand_final) < 5: continue

                 # Compare current strength vs opponent's potential CURRENT hand
                 opp_current_hand = opponent_hole + community_cards
                 if len(opp_current_hand) < 5: opp_current_rank = HandEvaluator.evaluate_hand(opponent_hole + community_cards[:max(0, 5-len(opponent_hole))]) # Approximation
                 else: opp_current_rank = HandEvaluator.evaluate_hand(opp_current_hand)

                 currently_ahead = current_rank > opp_current_rank

                 # Final comparison
                 player_rank_final = HandEvaluator.evaluate_hand(player_hand_final)
                 opponent_rank_final = HandEvaluator.evaluate_hand(opponent_hand_final)
                 ends_ahead = player_rank_final > opponent_rank_final

                 if currently_ahead:
                     ahead_count += 1
                     if ends_ahead: ahead_wins += 1
                     else: ahead_losses += 1
                 else: # Currently behind or tied
                     behind_count += 1
                     if ends_ahead: behind_wins += 1
                     else: behind_losses += 1
             except Exception: continue # Ignore errors in simulation

        # PPN = Positive Potential = Prob(Win | Currently Behind)
        Ppot = (behind_wins / behind_count) if behind_count > 0 else 0.0
        # NPN = Negative Potential = Prob(Lose | Currently Ahead)
        Npot = (ahead_losses / ahead_count) if ahead_count > 0 else 0.0

        # Some variants return Ppot, Npot directly. Others use EHS = ahead_wins/ahead_count
        # Let's return a combined potential metric (more complex metrics exist)
        # Return Ppot and (1-Npot) perhaps? Or just Ppot and Npot. Let's use that.
        return Ppot, Npot # Return Positive and Negative potential

    @staticmethod
    def _extract_hand_type_features(hole_cards, community_cards):
        """ Extract binary features about the current best hand and draws. """
        all_cards = hole_cards + community_cards
        if len(all_cards)<5: return (0,0,0,0,0) # Default if too few cards

        # Get current best 5-card hand type
        hand_rank, kickers = HandEvaluator.evaluate_hand(all_cards)

        # Feature flags based on current best hand rank
        has_pair = 1 if hand_rank >= HandEvaluator.HAND_TYPES['pair'] else 0
        has_two_pair = 1 if hand_rank >= HandEvaluator.HAND_TYPES['two_pair'] else 0
        has_trips = 1 if hand_rank >= HandEvaluator.HAND_TYPES['three_of_a_kind'] else 0

        # Check for draws (using all 5, 6, or 7 cards available)
        ranks = sorted([c.rank for c in all_cards], reverse=True)
        suits = [c.suit for c in all_cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Flush draw (4 cards of same suit)
        has_flush_draw = 1 if any(count == 4 for count in suit_counts.values()) else 0

        # Straight draw (4 cards to a straight)
        has_straight_draw = 0
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        # OESD (8 outs)
        for i in range(len(unique_ranks) - 3):
            # Check for 4 consecutive ranks (e.g., 8,7,6,5 for OESD)
            if unique_ranks[i] - unique_ranks[i+3] == 3:
                 # Ensure not already a straight (unless Ace-low handled separately)
                 if not (len(unique_ranks)>=5 and unique_ranks[i] - unique_ranks[i+4]==4):
                     has_straight_draw = 1; break
        # Gutshot (4 outs) - check if removing one card leaves 4 consecutive
        if not has_straight_draw and len(unique_ranks)>=4:
            for i in range(len(unique_ranks) - 3):
                # Check for 4 out of 5 consecutive ranks (e.g., 8,7,5,4 needs a 6)
                if unique_ranks[i] - unique_ranks[i+3] == 4 and (unique_ranks[i]-unique_ranks[i+1] > 1 or unique_ranks[i+1]-unique_ranks[i+2]>1 or unique_ranks[i+2]-unique_ranks[i+3]>1):
                     has_straight_draw = 1; break
        # Check for Ace-low gutshots (A234 needs 5, A345 needs 2 etc.) handled implicitly if Ace=14

        return has_pair, has_two_pair, has_trips, has_straight_draw, has_flush_draw

    @staticmethod
    def _extract_board_features(community_cards):
        """ Extract features about the board texture. """
        if not community_cards or len(community_cards)<3: return (0, 0, 0)
        ranks = sorted([c.rank for c in community_cards], reverse=True)
        suits = [c.suit for c in community_cards]
        rank_counts = Counter(ranks); suit_counts = Counter(suits)

        board_pair = 1 if any(count >= 2 for count in rank_counts.values()) else 0 # Paired board
        board_suited = 1 if any(count >= 3 for count in suit_counts.values()) else 0 # 3+ cards of same suit
        board_connected = 0 # Connected (e.g., 3+ cards within 5 rank range?) - Simplified: 2+ cards within 2 ranks
        unique_ranks = sorted(list(set(ranks)))
        for i in range(len(unique_ranks) - 1):
            if unique_ranks[i+1] - unique_ranks[i] <= 2: board_connected = 1; break
        return board_pair, board_suited, board_connected

    @staticmethod
    def _simple_postflop_bucket(hole_cards, community_cards, num_buckets):
        """ Simple postflop bucketing using strength and potential estimate. """
        # Fallback uses hand strength primarily
        hand_strength = EnhancedCardAbstraction._calculate_hand_strength(hole_cards, community_cards)
        # Could add a *very* rough potential score here if needed without MC
        score = hand_strength
        # Normalize score [0,1] to bucket range [num_buckets-1, 0] (inverted)
        bucket = num_buckets - 1 - int(score * (num_buckets -1))
        return max(0, min(num_buckets - 1, bucket)) # Clamp

    # --- Training Methods (Placeholder - require data generation/loading) ---
    @staticmethod
    def train_models(training_data_path=None, num_preflop=10000, num_postflop=20000):
        """ Train and save clustering models. Needs data or uses synthetic. """
        os.makedirs("models", exist_ok=True)
        models = {}
        data = {}

        # --- Preflop ---
        model_name = 'preflop'; model_key = '_preflop_model'
        num_clusters = EnhancedCardAbstraction.NUM_PREFLOP_BUCKETS
        model_path = EnhancedCardAbstraction._preflop_model_path
        try:
             if training_data_path and os.path.exists(os.path.join(training_data_path, f"{model_name}_data.pkl")):
                  with open(os.path.join(training_data_path, f"{model_name}_data.pkl"), 'rb') as f: data[model_name] = pickle.load(f)
             else: data[model_name] = EnhancedCardAbstraction._generate_synthetic_preflop_data(num_preflop)

             if data[model_name]:
                  model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                  print(f"Fitting {model_name} model ({len(data[model_name])} samples, {num_clusters} clusters)...")
                  model.fit(np.array(data[model_name]))
                  with open(model_path, 'wb') as f: pickle.dump(model, f)
                  models[model_name] = model; print(f"Saved {model_name} model.")
             else: print(f"WARN: No data for {model_name} model.")
        except Exception as e: print(f"ERROR training {model_name} model: {e}")

        # --- Postflop (Flop, Turn, River) ---
        for round_info in [('flop', 3, EnhancedCardAbstraction.NUM_FLOP_BUCKETS, EnhancedCardAbstraction._flop_model_path),
                           ('turn', 4, EnhancedCardAbstraction.NUM_TURN_BUCKETS, EnhancedCardAbstraction._turn_model_path),
                           ('river', 5, EnhancedCardAbstraction.NUM_RIVER_BUCKETS, EnhancedCardAbstraction._river_model_path)]:
             model_name, num_comm, num_clusters, model_path = round_info
             try:
                  if training_data_path and os.path.exists(os.path.join(training_data_path, f"{model_name}_data.pkl")):
                       with open(os.path.join(training_data_path, f"{model_name}_data.pkl"), 'rb') as f: data[model_name] = pickle.load(f)
                  else: data[model_name] = EnhancedCardAbstraction._generate_synthetic_postflop_data(num_comm, num_postflop)

                  if data[model_name]:
                       model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                       print(f"Fitting {model_name} model ({len(data[model_name])} samples, {num_clusters} clusters)...")
                       model.fit(np.array(data[model_name]))
                       with open(model_path, 'wb') as f: pickle.dump(model, f)
                       models[model_name] = model; print(f"Saved {model_name} model.")
                  else: print(f"WARN: No data for {model_name} model.")
             except Exception as e: print(f"ERROR training {model_name} model: {e}")

        # Set models on class if needed (or rely on lazy load)
        if 'preflop' in models: EnhancedCardAbstraction._preflop_model = models['preflop']
        if 'flop' in models: EnhancedCardAbstraction._flop_model = models['flop']
        if 'turn' in models: EnhancedCardAbstraction._turn_model = models['turn']
        if 'river' in models: EnhancedCardAbstraction._river_model = models['river']
        return models

    @staticmethod
    def _generate_synthetic_preflop_data(num_samples=1000):
        """ Generate synthetic preflop features. """
        data = []
        all_cards = [Card(r, s) for r in range(2, 15) for s in ['h','d','c','s']]
        count = 0
        # Sample combinations until enough valid feature sets generated
        while count < num_samples:
             try:
                  hole_cards = random.sample(all_cards, 2)
                  features = EnhancedCardAbstraction._extract_preflop_features(hole_cards)
                  if features: # Ensure features were generated
                       data.append(features); count+=1
             except Exception: continue # Skip if feature extraction fails
             if count > num_samples * 1.5: break # Safety break
        return data

    @staticmethod
    def _generate_synthetic_postflop_data(num_community, num_samples=1000):
        """ Generate synthetic postflop features. """
        data = []
        all_cards = [Card(r, s) for r in range(2, 15) for s in ['h','d','c','s']]
        count = 0
        while count < num_samples:
             try:
                  if 2 + num_community > len(all_cards): break # Not possible
                  sampled_cards = random.sample(all_cards, 2 + num_community)
                  hole_cards = sampled_cards[:2]
                  community_cards = sampled_cards[2:]
                  features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community_cards)
                  if features: # Ensure features were generated
                       data.append(features); count+=1
             except Exception: continue # Skip if feature extraction fails
             if count > num_samples * 1.5: break # Safety break
        return data

# --- END OF FILE organized_poker_bot/cfr/enhanced_card_abstraction.py ---
