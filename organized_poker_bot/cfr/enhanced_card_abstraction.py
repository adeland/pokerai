# --- START OF FILE organized_poker_bot/cfr/enhanced_card_abstraction.py ---
"""
Enhanced card abstraction using Scikit-learn K-Means clustering.
(Refactored V5: Added Debug Logging)
"""

import numpy as np
import random
import itertools
import pickle
import os
import sys
from collections import Counter
import time # For timing training steps

# --- Scikit-learn Import ---
try:
    from sklearn.cluster import KMeans
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARN [EnhancedCardAbs]: scikit-learn not found. EnhancedCardAbstraction will use simple fallbacks.")
    KMeans = None # Flag that sklearn is unavailable
    NotFittedError = type('NotFittedError', (Exception,), {}) # Dummy error class
    SKLEARN_AVAILABLE = False

# --- Absolute Imports for Game Engine/Utils ---
try:
    from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
    from organized_poker_bot.game_engine.card import Card
except ImportError as e:
    print(f"FATAL Import Error in enhanced_card_abstraction.py: {e}")
    sys.exit(1)
# --- End Imports ---


class EnhancedCardAbstraction:
    """ Enhanced card abstraction using K-Means clustering. """

    # --- Configuration ---
    NUM_PREFLOP_BUCKETS = 20
    NUM_FLOP_BUCKETS = 50
    NUM_TURN_BUCKETS = 50
    NUM_RIVER_BUCKETS = 50

    _MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    _preflop_model_path = os.path.join(_MODEL_DIR, 'preflop_kmeans_model.pkl')
    _flop_model_path    = os.path.join(_MODEL_DIR, 'flop_kmeans_model.pkl')
    _turn_model_path    = os.path.join(_MODEL_DIR, 'turn_kmeans_model.pkl')
    _river_model_path   = os.path.join(_MODEL_DIR, 'river_kmeans_model.pkl')

    # --- Model Loading Status ---
    # None: Not attempted yet
    # False: Attempted but failed or unavailable
    # KMeans object: Successfully loaded
    _preflop_model = None
    _flop_model = None
    _turn_model = None
    _river_model = None

    # --- Debugging Flag ---
    # Set to True to see detailed feature/prediction logs during abstraction calls
    DETAILED_LOGGING = False # Set to True manually for deep debugging

    @classmethod
    def _log(cls, message, level="INFO"):
        """ Simple internal logger. """
        # Simple logger prefix, could be expanded (e.g., add timestamp)
        print(f"[{level} EnhancedCardAbs]: {message}")

    @classmethod
    def _lazy_load_model(cls, model_attr_name, model_path):
        """ Helper to load a model file on first use with logging. Returns True on success. """
        model = getattr(cls, model_attr_name)

        # Check status: Already loaded (True) or failed previously (False)
        if model is not None:
            return model is not False # Return True if loaded, False if failed

        # --- Attempt to load ---
        cls._log(f"Attempting lazy load for {model_attr_name} from {model_path}...")

        if not os.path.exists(model_path):
            cls._log(f"Model file not found at '{model_path}'. Abstraction unavailable.", "WARN")
            setattr(cls, model_attr_name, False) # Mark as failed
            return False

        if not SKLEARN_AVAILABLE:
            cls._log(f"Scikit-learn unavailable. Cannot load model.", "WARN")
            setattr(cls, model_attr_name, False)
            return False

        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            # Basic validation of the loaded object
            if isinstance(loaded_model, KMeans) and hasattr(loaded_model, 'predict'):
                setattr(cls, model_attr_name, loaded_model)
                cls._log(f"Successfully loaded model for {model_attr_name}.")
                return True
            else:
                cls._log(f"Loaded object from '{model_path}' is not a valid KMeans model.", "ERROR")
                setattr(cls, model_attr_name, False)
                return False
        except Exception as e:
            cls._log(f"Failed to load/unpickle model '{model_path}': {e}", "ERROR")
            setattr(cls, model_attr_name, False) # Mark as failed
            return False

    # --- Abstraction Getters ---
    @classmethod
    def get_preflop_abstraction(cls, hole_cards):
        """ Gets preflop abstraction bucket. """
        if cls.DETAILED_LOGGING:
            card_str = ' '.join(map(str, hole_cards)) if hole_cards else "None"
            cls._log(f"get_preflop_abstraction called for: {card_str}", "DEBUG")

        if not hole_cards or len(hole_cards) != 2 or not all(isinstance(c, Card) for c in hole_cards):
            cls._log("Invalid hole cards for preflop abs. Returning worst bucket.", "WARN")
            return cls.NUM_PREFLOP_BUCKETS - 1

        model_loaded_successfully = cls._lazy_load_model('_preflop_model', cls._preflop_model_path)
        current_model = cls._preflop_model # Get potential model object

        if model_loaded_successfully and isinstance(current_model, KMeans):
            cls._log("Using K-Means model for preflop abstraction.", "DEBUG")
            try:
                features = cls._extract_preflop_features(hole_cards)
                if not features: # Check if feature extraction failed
                    raise ValueError("Feature extraction returned empty list")

                if cls.DETAILED_LOGGING:
                    cls._log(f"  Preflop features: {np.round(features, 3)}", "DEBUG")

                cluster = current_model.predict(np.array(features).reshape(1, -1))[0]
                bucket = int(np.clip(cluster, 0, cls.NUM_PREFLOP_BUCKETS - 1))

                if cls.DETAILED_LOGGING:
                    cls._log(f"  Predicted preflop bucket: {bucket}", "DEBUG")
                return bucket

            except NotFittedError: # Catch specific sklearn error
                cls._log("Model loaded but not fitted! Using simple fallback.", "WARN")
                setattr(cls, '_preflop_model', False) # Mark as failed
                return cls._simple_preflop_bucket(hole_cards)
            except Exception as e:
                cls._log(f"Error during prediction: {e}. Using simple fallback.", "WARN")
                # Optionally mark model as failed if prediction errors persist
                # setattr(cls, '_preflop_model', False)
                return cls._simple_preflop_bucket(hole_cards)
        else:
            cls._log("Using simple fallback for preflop abstraction.", "DEBUG")
            return cls._simple_preflop_bucket(hole_cards)

    @classmethod
    def get_postflop_abstraction(cls, hole_cards, community_cards):
        """ Gets postflop abstraction bucket. """
        num_community = len(community_cards) if community_cards else 0
        street_name = {3: "Flop", 4: "Turn", 5: "River"}.get(num_community, "Unknown")

        if cls.DETAILED_LOGGING:
             hole_str = ' '.join(map(str, hole_cards)) if hole_cards else "None"
             comm_str = ' '.join(map(str, community_cards)) if community_cards else "None"
             cls._log(f"get_postflop_abstraction ({street_name}) called for: {hole_str} | {comm_str}", "DEBUG")

        # Determine model config based on street
        num_buckets = -1
        model_attr = None
        model_path = None
        model = None # Variable to hold the loaded model object or False

        if num_community == 3:
            num_buckets = cls.NUM_FLOP_BUCKETS
            model_attr = '_flop_model'
            model_path = cls._flop_model_path
        elif num_community == 4:
            num_buckets = cls.NUM_TURN_BUCKETS
            model_attr = '_turn_model'
            model_path = cls._turn_model_path
        elif num_community == 5:
            num_buckets = cls.NUM_RIVER_BUCKETS
            model_attr = '_river_model'
            model_path = cls._river_model_path
        else:
            cls._log(f"Invalid community card count ({num_community}) for postflop abs. Returning worst bucket.", "WARN")
            # Use flop buckets as a default size if needed
            num_buckets = cls.NUM_FLOP_BUCKETS
            return num_buckets - 1

        # Basic input validation
        if not hole_cards or len(hole_cards) != 2 or not all(isinstance(c, Card) for c in hole_cards) or \
           not community_cards or not all(isinstance(c, Card) for c in community_cards):
            cls._log("Invalid cards for postflop abs. Returning worst bucket.", "WARN")
            return num_buckets - 1

        # Attempt to load model
        model_loaded_successfully = cls._lazy_load_model(model_attr, model_path)
        if model_loaded_successfully:
             model = getattr(cls, model_attr) # Get potential model object

        # Use model if available and loaded correctly
        if model_loaded_successfully and isinstance(model, KMeans):
            cls._log(f"Using K-Means model for {street_name} abstraction.", "DEBUG")
            try:
                features = cls._extract_postflop_features(hole_cards, community_cards)
                if not features: # Check feature extraction result
                    raise ValueError("Feature extraction returned empty list")

                if cls.DETAILED_LOGGING:
                    cls._log(f"  {street_name} features: {np.round(features, 3)}", "DEBUG")

                cluster = model.predict(np.array(features).reshape(1, -1))[0]
                bucket = int(np.clip(cluster, 0, num_buckets - 1))

                if cls.DETAILED_LOGGING:
                    cls._log(f"  Predicted {street_name} bucket: {bucket}", "DEBUG")
                return bucket

            except NotFittedError: # Catch specific sklearn error
                cls._log(f"{street_name} Model loaded but not fitted! Using simple fallback.", "WARN")
                setattr(cls, model_attr, False) # Mark as failed
                return cls._simple_postflop_bucket(hole_cards, community_cards, num_buckets)
            except Exception as e:
                cls._log(f"Error during {street_name} prediction: {e}. Using fallback.", "WARN")
                # Optionally mark model as failed if prediction errors persist
                # setattr(cls, model_attr, False)
                return cls._simple_postflop_bucket(hole_cards, community_cards, num_buckets)
        else: # Fallback if model not loaded/available
            cls._log(f"Using simple fallback for {street_name} abstraction.", "DEBUG")
            return cls._simple_postflop_bucket(hole_cards, community_cards, num_buckets)


    # --- Fallback Abstraction Methods ---
    @classmethod
    def _simple_preflop_bucket(cls, hole_cards):
        """ Simple deterministic preflop bucketing based on score heuristic. """
        # (Implementation identical to previous version, but ensure it's present and formatted)
        if len(hole_cards) != 2: return cls.NUM_PREFLOP_BUCKETS - 1
        try:
            rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
            suited = hole_cards[0].suit == hole_cards[1].suit
            is_pair = rank1 == rank2
            gap = rank1 - rank2 if not is_pair else -1.0

            score = (rank1 + rank2) / 2.0
            if is_pair: score += 5.0
            if suited: score += 2.0
            if gap >= 0 and gap < 5: score += (4.0 - gap) # Connector bonus

            min_score, max_score = 5.0, 21.0 # Approximate score range
            if max_score <= min_score: return 0 # Avoid division by zero

            normalized = (score - min_score) / (max_score - min_score)
            # Invert: High score -> Low bucket index
            bucket = cls.NUM_PREFLOP_BUCKETS - 1 - int(np.clip(normalized, 0, 1) * (cls.NUM_PREFLOP_BUCKETS - 1))
            return int(np.clip(bucket, 0, cls.NUM_PREFLOP_BUCKETS - 1)) # Ensure valid range
        except Exception as e:
             cls._log(f"Error in _simple_preflop_bucket: {e}", "ERROR")
             return cls.NUM_PREFLOP_BUCKETS - 1 # Default to worst bucket on error

    @classmethod
    def _simple_postflop_bucket(cls, hole_cards, community_cards, num_buckets):
        """ Simple postflop bucketing based primarily on normalized hand strength. """
        # (Implementation identical to previous version)
        try:
             strength = cls._calculate_normalized_strength(hole_cards, community_cards)
             # Invert: High strength -> Low bucket index
             bucket = num_buckets - 1 - int(strength * (num_buckets - 1))
             return int(np.clip(bucket, 0, num_buckets - 1)) # Clamp
        except Exception as e:
             cls._log(f"Error in _simple_postflop_bucket: {e}", "ERROR")
             return num_buckets - 1 # Default to worst bucket on error

    # --- Feature Extraction ---
    @classmethod
    def _extract_preflop_features(cls, hole_cards):
        """ Extract numerical features for preflop K-Means clustering. Returns list or None on error. """
        try:
             # (Implementation identical to previous version, wrapped in try/except)
             rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
             suited = 1.0 if hole_cards[0].suit == hole_cards[1].suit else 0.0
             is_pair = 1.0 if rank1 == rank2 else 0.0
             gap = float(rank1 - rank2) if not is_pair else -1.0
             norm_rank1 = (rank1 - 2.0) / 12.0
             norm_rank2 = (rank2 - 2.0) / 12.0
             norm_gap = max(0.0, gap) / 12.0
             features = [norm_rank1, norm_rank2, suited, is_pair, norm_gap]
             return features
        except Exception as e:
             cls._log(f"Error extracting preflop features: {e}", "ERROR")
             return None


    @classmethod
    def _extract_postflop_features(cls, hole_cards, community_cards, include_potential=False):
        """ Extract numerical features for postflop K-Means clustering. Returns list or None on error. """
        try:
            if not hole_cards or len(hole_cards) != 2 or not community_cards or len(community_cards) < 3:
                 return None # Invalid input

            hand_strength = cls._calculate_normalized_strength(hole_cards, community_cards)
            bp, bs, bc = cls._extract_board_features(community_cards)
            hp, htp, htr, hsd, hfd = cls._extract_hand_type_features(hole_cards, community_cards)

            # Base features
            features = [hand_strength, bp, bs, bc, float(hp), float(htp), float(htr), float(hsd), float(hfd)]

            # Potential features (optional)
            ppot, npot = 0.0, 0.0
            if include_potential:
                try: # Wrap potential calculation as it can be slower/fail
                    ppot, npot = cls._calculate_hand_potential(hole_cards, community_cards)
                except Exception as e:
                    cls._log(f"Error calculating potential features: {e}", "WARN")
                    # Keep ppot, npot as 0.0

            # Always extend, either with calculated values or defaults
            features.extend([ppot, npot])

            return features
        except Exception as e:
             cls._log(f"Error extracting postflop features: {e}", "ERROR")
             return None


    @classmethod
    def _calculate_normalized_strength(cls, hole_cards, community_cards):
        """ Calculate hand strength normalized to [0, 1]. Returns float or raises error. """
        # (Implementation identical to previous version, assumes HandEvaluator exists)
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5: return 0.0

        hand_eval_result = HandEvaluator.evaluate_hand(all_cards) # Can raise error
        if not isinstance(hand_eval_result, tuple) or len(hand_eval_result) == 0:
             raise ValueError("Invalid result from HandEvaluator")
        hand_type_value = hand_eval_result[0]

        try:
            all_ranks = list(HandEvaluator.HAND_TYPES.values())
            min_r, max_r = min(all_ranks), max(all_ranks)
        except AttributeError: min_r, max_r = 0, 9 # Fallback

        if max_r <= min_r: return 0.5 # Avoid division by zero

        normalized = (hand_type_value - min_r) / float(max_r - min_r)
        return float(np.clip(normalized, 0.0, 1.0))


    @classmethod
    def _extract_hand_type_features(cls, hole_cards, community_cards):
        """ Extract binary hand type/draw features. Returns tuple or raises error. """
        # (Implementation identical to previous version, assumes HandEvaluator exists)
        all_cards = hole_cards + community_cards; n = len(all_cards)
        if n < 5: return (0,0,0,0,0)

        hand_rank_val, _ = HandEvaluator.evaluate_hand(all_cards) # Can raise error

        h_p = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('pair', 1) else 0
        h_tp = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('two_pair', 2) else 0
        h_tr = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('three_of_a_kind', 3) else 0

        ranks = sorted([c.rank for c in all_cards], reverse=True); suits = [c.suit for c in all_cards]
        s_counts = Counter(suits); u_ranks = sorted(list(set(ranks)), reverse=True)

        h_fd = 1 if any(c == 4 for c in s_counts.values()) else 0
        h_sd = 0
        if len(u_ranks) >= 4:
             # Simplified OESD/Gutshot check
             for i in range(len(u_ranks) - 3):
                 if u_ranks[i] - u_ranks[i+3] <= 4: # Check if 4 ranks span 4 or 5 positions
                      h_sd = 1; break
             # Wheel draw check (A + 3 low cards)
             if not h_sd and 14 in u_ranks and len({r for r in u_ranks if r <= 5}) >= 3:
                  h_sd = 1

        return h_p, h_tp, h_tr, h_sd, h_fd

    @classmethod
    def _extract_board_features(cls, community_cards):
        """ Extract numerical board features (paired, suited, connected). Returns tuple or raises error. """
        # (Implementation identical to previous version)
        n = len(community_cards)
        if n < 3: return (0.0, 0.0, 0.0)

        ranks = [c.rank for c in community_cards]; suits = [c.suit for c in community_cards]
        r_counts = Counter(ranks); s_counts = Counter(suits); u_ranks = sorted(list(set(ranks)))

        pair_f = 0.0; p_counts = [c for c in r_counts.values() if c>=2]
        if len(p_counts) >= 2 or any(c>=3 for c in p_counts): pair_f = 1.0
        elif len(p_counts)==1: pair_f = 0.5

        suit_f = 0.0; max_s_count = max(s_counts.values()) if s_counts else 0
        if max_s_count >= 5: suit_f = 1.0
        elif max_s_count == 4: suit_f = 2.0/3.0
        elif max_s_count == 3: suit_f = 1.0/3.0

        conn_f = 0.0
        if len(u_ranks) >= 3:
             is_v_conn, is_m_conn = False, False
             for i in range(len(u_ranks) - 2):
                 if u_ranks[i+2] - u_ranks[i] == 2: is_v_conn = True; break
             if not is_v_conn:
                 for i in range(len(u_ranks) - 2):
                      if u_ranks[i+2] - u_ranks[i] <= 3: is_m_conn = True; break
             # Wheel check (Ace + 2 low cards) implies medium connectivity if not already higher
             if not is_v_conn and not is_m_conn and 14 in u_ranks and len({r for r in u_ranks if r <= 5}) >= 2:
                 is_m_conn = True

             if is_v_conn: conn_f = 1.0
             elif is_m_conn: conn_f = 0.5

        return pair_f, suit_f, conn_f

    @staticmethod
    def _calculate_hand_potential(hole_cards, community_cards, num_simulations=50):
        """ Estimates Ppot, Npot via Monte Carlo. Returns tuple or raises error. """
        # (Implementation identical to previous version, assumes HandEvaluator exists)
        current_hand = hole_cards + community_cards; n_curr = len(current_hand)
        if n_curr < 5: return 0.0, 0.0 # Cannot calculate potential pre-flop

        curr_rank = HandEvaluator.evaluate_hand(current_hand) # Can raise error

        used_set = frozenset(current_hand)
        deck_list = [Card(r, s) for r in range(2, 15) for s in Card.SUITS if Card(r,s) not in used_set]
        n_draw = max(0, 5 - len(community_cards))
        if n_draw == 0: return 0.0, 0.0 # No potential on river

        if len(deck_list) < 2 + n_draw: # Need opp hole cards + runout cards
             raise RuntimeError("Not enough cards in deck for potential simulation")

        pp_wins, np_losses = 0, 0
        pp_count, np_count = 0, 0

        for _ in range(num_simulations):
            deck_samp = deck_list[:]; random.shuffle(deck_samp)
            opp_hole = deck_samp[:2]
            runout = deck_samp[2 : 2+n_draw]; final_comm = community_cards + runout

            player_final = hole_cards + final_comm
            opp_final = opp_hole + final_comm
            if len(player_final) < 5 or len(opp_final) < 5: continue

            player_final_rank = HandEvaluator.evaluate_hand(player_final)
            opp_final_rank = HandEvaluator.evaluate_hand(opp_final)

            opp_curr = opp_hole + community_cards
            opp_curr_rank = HandEvaluator.evaluate_hand(opp_curr) if len(opp_curr) >= 5 else (-1,[])

            is_ahead = curr_rank > opp_curr_rank
            is_behind = curr_rank < opp_curr_rank
            wins_sd = player_final_rank > opp_final_rank

            if is_ahead: np_count += 1; np_losses += (0 if wins_sd else 1)
            elif is_behind: pp_count += 1; pp_wins += (1 if wins_sd else 0)

        Ppot = (pp_wins / pp_count) if pp_count > 0 else 0.0
        Npot = (np_losses / np_count) if np_count > 0 else 0.0
        return float(Ppot), float(Npot)

    # --- K-Means Model Training ---
    @staticmethod
    def train_models(num_samples_preflop=20000, num_samples_postflop=50000, random_state=42, include_potential_in_features=False):
        """
        Trains K-Means clustering models for each street and saves them.
        Uses synthetic data generation. Ensure 'models/' directory exists.
        """
        if not SKLEARN_AVAILABLE: # Check if sklearn was imported
            EnhancedCardAbstraction._log("Cannot train models: scikit-learn not installed.", "ERROR")
            return None

        EnhancedCardAbstraction._log("--- Starting Enhanced Card Abstraction Model Training ---")
        os.makedirs(EnhancedCardAbstraction._MODEL_DIR, exist_ok=True)
        trained_models = {}
        overall_start_time = time.time()

        # --- Preflop Model ---
        model_name = 'preflop'; num_clusters = EnhancedCardAbstraction.NUM_PREFLOP_BUCKETS
        model_path = EnhancedCardAbstraction._preflop_model_path
        EnhancedCardAbstraction._log(f"\nTraining {model_name} model (k={num_clusters})...")
        start_t = time.time()
        EnhancedCardAbstraction._log(f" Generating {num_samples_preflop:,} synthetic samples...")
        preflop_data = EnhancedCardAbstraction._generate_synthetic_preflop_data(num_samples_preflop)
        data_t = time.time()
        if preflop_data:
            EnhancedCardAbstraction._log(f" Fitting {model_name} K-Means ({len(preflop_data)} samples)...")
            try:
                 # <<< MODIFICATION: Add verbose=1 and n_jobs=-1 >>>
                 preflop_kmeans = KMeans(
                     n_clusters=num_clusters,
                     random_state=random_state,
                     n_init=10, # Default is 10, explicit for clarity
                     verbose=1, # Print progress messages
                 )
                 # <<< END MODIFICATION >>>

                 preflop_kmeans.fit(np.array(preflop_data))
                 fit_t = time.time()
                 with open(model_path, 'wb') as f: pickle.dump(preflop_kmeans, f)
                 save_t = time.time()
                 EnhancedCardAbstraction._log(f" SUCCESS: {model_name} model saved ({save_t - start_t:.2f}s total: gen={data_t-start_t:.2f}s, fit={fit_t-data_t:.2f}s, save={save_t-fit_t:.2f}s)")
                 trained_models['preflop'] = preflop_kmeans
            except Exception as e: EnhancedCardAbstraction._log(f" ERROR training/saving {model_name}: {e}", "ERROR")
        else: EnhancedCardAbstraction._log(" No preflop data generated.", "WARN")

        # --- Postflop Models ---
        postflop_configs = [
            ("Flop", 3, EnhancedCardAbstraction.NUM_FLOP_BUCKETS, EnhancedCardAbstraction._flop_model_path),
            ("Turn", 4, EnhancedCardAbstraction.NUM_TURN_BUCKETS, EnhancedCardAbstraction._turn_model_path),
            ("River", 5, EnhancedCardAbstraction.NUM_RIVER_BUCKETS, EnhancedCardAbstraction._river_model_path)
        ]
        EnhancedCardAbstraction._log(f"\nGenerating {num_samples_postflop:,} synthetic postflop samples (Features based on River state, Potential={include_potential_in_features})...")
        start_t_post_gen = time.time()
        postflop_data = EnhancedCardAbstraction._generate_synthetic_postflop_data(5, num_samples_postflop, include_potential_in_features)
        EnhancedCardAbstraction._log(f" Postflop data generated ({time.time() - start_t_post_gen:.2f}s).")

        if not postflop_data:
             EnhancedCardAbstraction._log(" No postflop data generated. Skipping Flop/Turn/River models.", "WARN")
        else:
            postflop_data_np = np.array(postflop_data)
            for round_name, num_comm, num_clusters, model_path in postflop_configs:
                  EnhancedCardAbstraction._log(f"\nTraining {round_name} model (k={num_clusters})...")
                  start_t = time.time()
                  try:
                       # <<< MODIFICATION: Add verbose=1 and n_jobs=-1 >>>
                       kmeans = KMeans(
                           n_clusters=num_clusters,
                           random_state=random_state,
                           n_init=10,
                           verbose=1, # Print progress messages
                       )
                       # <<< END MODIFICATION >>>

                       EnhancedCardAbstraction._log(f" Fitting {round_name} K-Means ({len(postflop_data_np)} samples)...")
                       kmeans.fit(postflop_data_np) # Fit on the generated data
                       fit_t = time.time()
                       with open(model_path, 'wb') as f: pickle.dump(kmeans, f)
                       save_t = time.time()
                       EnhancedCardAbstraction._log(f" SUCCESS: {round_name} model saved ({save_t - start_t:.2f}s total: fit={fit_t-start_t:.2f}s, save={save_t-fit_t:.2f}s)")
                       trained_models[round_name.lower()] = kmeans
                  except Exception as e: EnhancedCardAbstraction._log(f" ERROR training/saving {round_name}: {e}", "ERROR")

        total_t = time.time() - overall_start_time
        EnhancedCardAbstraction._log(f"\n--- Model Training Finished ({total_t:.2f} seconds total) ---")
        # Update class attributes (Optional: lazy loading might be preferred)
        # ...
        return trained_models

    # --- Synthetic Data Generation ---
    # (Static methods, implementations identical to previous version, ensure formatted)
    @staticmethod
    def _generate_synthetic_preflop_data(num_samples):
        """ Generate features for random preflop hands. """
        log = EnhancedCardAbstraction._log # Use class logger
        data = []; count = 0; attempts = 0
        all_cards = [Card(r,s) for r in range(2, 15) for s in Card.SUITS]
        max_attempts = num_samples * 5 # Allow more attempts

        while count < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                c1, c2 = random.sample(all_cards, 2)
                features = EnhancedCardAbstraction._extract_preflop_features([c1, c2])
                if features is not None: # Check if features were extracted
                     data.append(features)
                     count += 1
            except Exception: continue # Skip errors silently

        if count < num_samples:
             log(f" Preflop data gen only produced {count}/{num_samples} samples.", "WARN")
        return data

    @staticmethod
    def _generate_synthetic_postflop_data(num_community_cards, num_samples, include_potential):
        """ Generate features for random postflop hands/boards. """
        log = EnhancedCardAbstraction._log
        data = []; count = 0; attempts = 0
        all_cards = [Card(r,s) for r in range(2, 15) for s in Card.SUITS]
        cards_needed = 2 + num_community_cards
        max_attempts = num_samples * 10 # Allow more attempts

        if len(all_cards) < cards_needed:
            log("Not enough cards in deck to generate samples.", "ERROR")
            return []

        for _ in range(max_attempts): # Use a for loop with break for clarity
             if count >= num_samples: break
             try:
                 sampled_cards = random.sample(all_cards, cards_needed)
                 hole_cards = sampled_cards[:2]
                 community = sampled_cards[2:]
                 features = EnhancedCardAbstraction._extract_postflop_features(hole_cards, community, include_potential)
                 if features is not None: # Check if features extracted
                      data.append(features)
                      count += 1
             except Exception: continue # Skip errors

        if count < num_samples:
            log(f" Postflop data gen only produced {count}/{num_samples} samples.", "WARN")
        return data

# --- Example Usage / Training Trigger ---
if __name__ == '__main__':
    print("="*50)
    print(" Example: Training Enhanced Card Abstraction Models (Small Sample)")
    print("="*50)
    # Ensure the models/ directory exists relative to project root when running this directly
    # Note: os.makedirs is called within train_models, so this is just informational.
    print(f"Models will be saved in: {EnhancedCardAbstraction._MODEL_DIR}")

    # Run training with small sample size for demonstration
    EnhancedCardAbstraction.train_models(
        num_samples_preflop=1000,
        num_samples_postflop=2000,
        include_potential_in_features=False # Set True to include slower potential calculation
    )

    print("\n" + "="*50)
    print(" Example Usage after Training/Loading:")
    print("="*50)
    # Example hand and community cards
    h = [Card(14, 's'), Card(13, 's')] # AKs
    c_flop = [Card(12, 's'), Card(7, 'h'), Card(2, 's')] # Flop Qs 7h 2s
    c_turn = c_flop + [Card(8, 'd')] # Turn 8d
    c_river = c_turn + [Card(11, 'c')] # River Jc

    # Get abstractions (lazy loading should happen here if models trained/exist)
    EnhancedCardAbstraction.DETAILED_LOGGING = True # Enable detailed logs for example usage
    pre_b = EnhancedCardAbstraction.get_preflop_abstraction(h)
    print(f"\nPreflop Bucket for {' '.join(map(str, h))}: {pre_b}")

    flop_b = EnhancedCardAbstraction.get_postflop_abstraction(h, c_flop)
    print(f"\nFlop Bucket for {' '.join(map(str, h))} on {' '.join(map(str, c_flop))}: {flop_b}")

    turn_b = EnhancedCardAbstraction.get_postflop_abstraction(h, c_turn)
    print(f"\nTurn Bucket for {' '.join(map(str, h))} on {' '.join(map(str, c_turn))}: {turn_b}")

    river_b = EnhancedCardAbstraction.get_postflop_abstraction(h, c_river)
    print(f"\nRiver Bucket for {' '.join(map(str, h))} on {' '.join(map(str, c_river))}: {river_b}")
    EnhancedCardAbstraction.DETAILED_LOGGING = False # Disable after example

# --- END OF FILE ---
