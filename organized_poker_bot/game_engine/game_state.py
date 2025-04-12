# --- START OF FILE organized_poker_bot/game_engine/game_state.py ---
"""
Game state implementation for poker games.
(Refactored V38: Integrate apply_action/is_round_over/try_advance fixes, keep debug logs)
"""

import random
import math
import sys
import os
import traceback
from collections import defaultdict, Counter # Added Counter
from copy import deepcopy
import numpy as np # For NaN/Inf checks

# Path setup / Absolute Imports
try:
    # Make sure these imports work relative to your project structure
    from organized_poker_bot.game_engine.deck import Deck
    from organized_poker_bot.game_engine.card import Card
    from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
except ImportError as e:
    print(f"ERROR importing engine components in GameState: {e}")
    # Attempt absolute import if relative fails (useful if running from root)
    try:
        from game_engine.deck import Deck
        from game_engine.card import Card
        from game_engine.hand_evaluator import HandEvaluator
    except ImportError:
         print(f"ERROR: Could not import GameState dependencies via relative or absolute paths.")
         sys.exit(1)


class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River", 4: "Showdown", 5: "Hand Over"}
    MAX_RAISES_PER_STREET = 7 # Example cap

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        if not (2 <= num_players <= 9):
            raise ValueError("Num players must be 2-9")
        self.num_players = int(num_players)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        # State Init
        self.player_stacks = [float(starting_stack)] * self.num_players
        self.hole_cards = [[] for _ in range(self.num_players)]
        self.player_total_bets_in_hand = [0.0] * self.num_players
        self.player_bets_in_round = [0.0] * self.num_players
        self.player_folded = [False] * self.num_players
        self.player_all_in = [False] * self.num_players
        # Initial assumption, refined in start_new_hand
        self.active_players = list(range(self.num_players)) # Tracks who STARTED hand with chips
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.deck = Deck()
        self.dealer_position = 0
        self.current_player_idx = -1 # Who's turn is it?
        self.current_bet = 0.0 # Highest total bet amount placed in current round
        self.last_raiser = None # Index of last player to bet/raise
        self.last_raise = 0.0 # Store the size of the last raise increment
        self.players_acted_this_round = set() # Tracks who acted since last bet/raise
        self.raise_count_this_street = 0 # Counts bets/raises in current round
        self.action_sequence = [] # List of action strings
        self.verbose_debug = False # Can be set externally if needed

    # --- Helper Methods ---
    def _get_next_active_player(self, start_idx):
        """ Finds the next player index who has chips and hasn't folded, starting after start_idx. (Corrected V2)"""
        if self.num_players == 0:
            return None # Handle edge case

        # Calculate the index to start checking from (player *after* start_idx)
        # Use modulo arithmetic to wrap around the table
        # Ensure start_idx is valid before calculating current_idx
        valid_start = start_idx if 0 <= start_idx < self.num_players else -1
        if valid_start == -1:
            # If start_idx was invalid, start search from a default position (e.g., 0)
            current_idx = 0
        else:
            current_idx = (valid_start + 1) % self.num_players

        search_start_idx = current_idx # Remember where the search loop began

        # Loop through players up to num_players times to check everyone once
        for _ in range(self.num_players):
            # --- CORE CHECK ---
            # Check if player index is valid AND player has chips AND is not folded
            # No dependency on the self.active_players list from start_new_hand needed here
            try:
                # Bounds checks first
                if 0 <= current_idx < self.num_players and \
                   current_idx < len(self.player_stacks) and \
                   current_idx < len(self.player_folded):
                    # Check actual state
                    if self.player_stacks[current_idx] > 0.01 and \
                       not self.player_folded[current_idx]:
                        # Found the next active player
                        return current_idx
                # else: Index out of bounds or lists not initialized - skip
            except IndexError:
                # This catch block is less critical now with explicit bounds checks above
                pass # Ignore this index and continue search

            # Move to the next player index, wrapping around
            current_idx = (current_idx + 1) % self.num_players

            # Optimization: If we've looped back to the start, we've checked everyone
            if current_idx == search_start_idx:
                break

        # If the loop completes without finding a valid player
        return None

    def _find_player_relative_to_dealer(self, offset):
        """ Finds a player with chips at a specific offset from the dealer. """
        # Relies on self.active_players only to find the *seat*, then uses current stacks
        # This seems intended to find initial SB/BB based on *who started* the hand
        if self.num_players == 0: return None

        dealer = self.dealer_position % self.num_players
        potential_start_idx = (dealer + offset) % self.num_players
        current_check_idx = potential_start_idx

        # Loop around the table once
        for _ in range(self.num_players):
            # Check if player *started* with chips (was in active_players list initially)
            # AND check their *current* stack (they might have busted immediately on blinds)
            if 0 <= current_check_idx < len(self.player_stacks) and \
               self.player_stacks[current_check_idx] > 0.01:
                 return current_check_idx # Found a player at the relative position with chips

            # Move to the next seat index
            current_check_idx = (current_check_idx + 1) % self.num_players

        # If loop completes without finding a player with chips at that relative position
        return None

    # --- Hand Setup Methods ---
    def start_new_hand(self, dealer_pos, player_stacks):
        """ Sets up the game state for the beginning of a new hand. """
        # Reset hand-specific state
        self.hole_cards = [[] for _ in range(self.num_players)]
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.player_bets_in_round = [0.0] * self.num_players
        self.player_total_bets_in_hand = [0.0] * self.num_players
        self.player_folded = [False] * self.num_players
        self.player_all_in = [False] * self.num_players
        self.current_player_idx = -1
        self.current_bet = 0.0
        self.last_raiser = None
        self.last_raise = 0.0
        self.players_acted_this_round = set()
        self.action_sequence = []
        self.raise_count_this_street = 0

        # Set game parameters for this hand
        self.dealer_position = dealer_pos % self.num_players
        self.deck = Deck()
        self.deck.shuffle()

        # Update player stacks and determine active players for *this* hand
        if len(player_stacks) != self.num_players:
             raise ValueError("Provided player_stacks length does not match num_players")
        self.player_stacks = [float(s) for s in player_stacks]
        # Store who starts the hand with chips - IMPORTANT for relative pos finder
        self.active_players = [i for i, s in enumerate(self.player_stacks) if s > 0.01]

        # Proceed with dealing and blinds if enough players
        if len(self.active_players) >= 2:
            self._deal_hole_cards()
            # Check if dealing failed (e.g., deck empty)
            if self.betting_round == self.HAND_OVER: return
            self._post_blinds()
            # Check if posting blinds failed (or only one player left)
            if self.betting_round == self.HAND_OVER: return
            # ---- Start FIRST betting round (preflop) ---
            self._start_betting_round()
            # Check if starting round failed (e.g., only one player left after blinds)
            if self.current_player_idx == -1:
                 self.betting_round = self.HAND_OVER # Mark hand over if no one can act preflop
        else: # Not enough active players
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1


    def _deal_hole_cards(self):
        """ Deals two cards to each player who started the hand with chips. """
        if len(self.active_players) < 2: # Cannot deal if fewer than 2 players started with stacks
            self.betting_round = self.HAND_OVER
            return

        # Deal one card at a time, starting left of dealer, only to players in self.active_players
        try:
            player_to_receive_card = (self.dealer_position + 1) % self.num_players
            for _ in range(2): # Two passes
                for i in range(self.num_players):
                    current_player_idx = (player_to_receive_card + i) % self.num_players
                    # Only deal to players who were active at the start of the hand
                    if current_player_idx in self.active_players:
                        if not self.deck: # Check if deck is empty
                             print("ERROR: Deck empty during hole card deal.")
                             self.betting_round = self.HAND_OVER
                             return # Cannot deal
                        # Deal the card if they haven't received the right number yet
                        if len(self.hole_cards[current_player_idx]) < (_ + 1):
                             self.hole_cards[current_player_idx].append(self.deck.deal())
        except Exception as e:
             print(f"ERROR occurred during hole card dealing: {e}")
             self.betting_round = self.HAND_OVER

    def _deduct_bet(self, player_idx, amount_to_deduct):
        """ Internal helper to deduct bet, update pot/state. Returns actual amount deducted. """
        if not (0 <= player_idx < self.num_players and amount_to_deduct >= 0):
            # print(f"WARN _deduct_bet: Invalid input P{player_idx}, Amt={amount_to_deduct}")
            return 0.0 # Invalid input

        # Determine actual amount (capped by stack)
        # Add tolerance: Allow betting slightly more than stack if difference is negligible
        actual_deduction = min(amount_to_deduct, self.player_stacks[player_idx])
        if amount_to_deduct > self.player_stacks[player_idx] + 0.001:
            # If requested amount is significantly more than stack, only deduct the stack
            actual_deduction = self.player_stacks[player_idx]

        # Ensure deduction is positive
        actual_deduction = max(0.0, actual_deduction)

        # Apply deduction and update state if significant amount
        if actual_deduction > 0.001: # Use small threshold
            self.player_stacks[player_idx] -= actual_deduction
            self.player_bets_in_round[player_idx] += actual_deduction
            self.player_total_bets_in_hand[player_idx] += actual_deduction
            self.pot += actual_deduction

            # Clamp stack to zero if negative after deduction (due to float issues)
            if self.player_stacks[player_idx] < 0:
                self.player_stacks[player_idx] = 0.0

            # Check if player went all-in (stack is now effectively zero)
            if abs(self.player_stacks[player_idx]) < 0.01:
                self.player_all_in[player_idx] = True

        return actual_deduction


    def _post_blinds(self):
        """ Posts small and big blinds based on dealer position and active players. """
        if len(self.active_players) < 2: # Need 2+ for blinds
            self.betting_round = self.HAND_OVER # Not enough players left
            return

        sb_player, bb_player = None, None
        # Use _find_player_relative_to_dealer which finds players *with chips currently*
        if len(self.active_players) == 2: # HU: Use relative positions from dealer
            sb_player = self._find_player_relative_to_dealer(0) # Dealer is SB
            bb_player = self._find_player_relative_to_dealer(1) # Non-dealer is BB
        else: # 3+ players: Normal positions relative to dealer
            sb_player = self._find_player_relative_to_dealer(1)
            bb_player = self._find_player_relative_to_dealer(2)

        # Check if SB/BB could be found (they must have > 0 stack at this point)
        if sb_player is None or bb_player is None:
             print(f"ERROR: Cannot find distinct SB ({sb_player}) or BB ({bb_player}) players with chips.")
             self.betting_round = self.HAND_OVER
             return
        # This check might be redundant if _find guarantees distinct players
        if self.num_players > 2 and sb_player == bb_player:
             print(f"ERROR: SB ({sb_player}) and BB ({bb_player}) positions are the same.")
             self.betting_round = self.HAND_OVER
             return

        self.raise_count_this_street = 0 # Reset preflop raise count before blinds

        # Post Small Blind
        sb_posted_amount = 0.0
        sb_amount_to_post = min(self.small_blind, self.player_stacks[sb_player])
        sb_posted_amount = self._deduct_bet(sb_player, sb_amount_to_post)
        if sb_posted_amount > 0.001:
            # Use the amount *added* to the round bet for the log
            # Need total round bet after deduction for proper logging
            log_sb_total_in_round = self.player_bets_in_round[sb_player]
            self.action_sequence.append(f"P{sb_player}:sb{int(round(log_sb_total_in_round))}") # Log total amount IN for round

        # Post Big Blind
        bb_posted_amount = 0.0
        bb_amount_to_post = min(self.big_blind, self.player_stacks[bb_player])
        bb_posted_amount = self._deduct_bet(bb_player, bb_amount_to_post)
        if bb_posted_amount > 0.001:
            # Log total amount IN POT for round after BB post
            log_bb_total_in_round = self.player_bets_in_round[bb_player]
            self.action_sequence.append(f"P{bb_player}:bb{int(round(log_bb_total_in_round))}")

        # Set initial betting level and raiser state
        # current_bet is the level players must CALL TO
        self.current_bet = max(self.player_bets_in_round[sb_player], self.player_bets_in_round[bb_player], self.big_blind)
        self.last_raise = self.big_blind # Min raise increment baseline is BB size

        # Determine who made the last 'aggressive' action (the BB usually)
        if self.player_bets_in_round[bb_player] >= self.big_blind - 0.01:
            # If BB posted full amount (or went all-in for at least BB)
            self.last_raiser = bb_player
            self.raise_count_this_street = 1 # BB post counts as the first 'raise'
        elif self.player_bets_in_round[sb_player] > 0.01: # Fallback if BB short/missing, check SB
            self.last_raiser = sb_player
            # current_bet already includes SB's bet
            self.last_raise = self.player_bets_in_round[sb_player] # Next raise must be >= SB's contribution
            self.raise_count_this_street = 1 # SB post counts as first 'raise' if BB didn't complete
        else: # No effective 'raise' if neither could post significant blind
            self.last_raiser = None
            # current_bet should still be 0 if no blinds posted significantly
            self.last_raise = self.big_blind # Next aggression must be at least BB
            self.raise_count_this_street = 0

        # Update current_bet based on the maximum bet placed
        self.current_bet = max(self.player_bets_in_round[sb_player], self.player_bets_in_round[bb_player])


    # --- Round Progression ---
    def _start_betting_round(self):
        """ Initializes state for the start of a betting round. Sets current_player_idx. """
        # Reset round-specific state if POST-FLOP
        if self.betting_round != self.PREFLOP:
            self.current_bet = 0.0
            self.last_raiser = None
            self.last_raise = self.big_blind # Minimum bet/raise size based on BB post-flop
            self.raise_count_this_street = 0
            # Clear bets *in this round* (keep total bets in hand)
            self.player_bets_in_round = [0.0] * self.num_players

        self.players_acted_this_round = set() # Clear who acted relative to last aggression
        first_player_to_act = None

        if self.betting_round == self.PREFLOP:
            # --- Find player UTG (Under The Gun) ---
            # Calculate expected BB index robustly
            bb_idx = -1
            if self.num_players == 2: bb_idx = (self.dealer_position + 1) % 2
            elif self.num_players > 2: bb_idx = (self.dealer_position + 2) % self.num_players

            if bb_idx != -1:
                # Start search from player *after* BB
                first_player_to_act = self._get_next_active_player(bb_idx)
            else: # Fallback if BB calculation failed
                first_player_to_act = self._get_next_active_player(self.dealer_position) # Should not happen

        else: # --- Postflop rounds: First active Player LEFT of dealer acts first ---
            first_player_to_act = self._get_next_active_player(self.dealer_position)


        # Set current player index or handle failure
        if first_player_to_act is None:
            # This can happen if only 1 player remains after folds/blinds/etc.
             print(f"!!! WARN _start_betting_round: FAILED to find first_player_to_act (Rnd={self.betting_round}, Dlr={self.dealer_position}, Stacks={self.player_stacks}, Folded={self.player_folded})") # Add Folded state
             self.current_player_idx = -1 # Indicate no player's turn
        else:
             # print(f"    DEBUG _start: Setting current_player_idx = {first_player_to_act}")
             self.current_player_idx = first_player_to_act

        # Final check: If <=1 player can actually make a decision, turn should remain -1
        # This prevents starting betting if checks revealed only one non-all-in player remains
        if self._check_all_active_are_allin():
             # print(f"DEBUG _start_betting_round: Skipping betting, players all-in or only one can act. Round: {self.betting_round}")
             self.current_player_idx = -1 # Skip betting


    def _deal_community_card(self, burn=True):
        """ Deals one card, optionally burning. Returns True if successful. """
        try:
            if burn:
                if not self.deck: print("WARN: Cannot burn, deck empty."); return False
                self.deck.deal() # Burn card
            if not self.deck: print("WARN: Cannot deal community, deck empty."); return False
            self.community_cards.append(self.deck.deal())
            return True
        except Exception as e:
             print(f"ERROR dealing community card: {e}")
             return False

    def deal_flop(self):
        """ Deals the flop cards. (Revised: Doesn't start round) """
        # Need 4 cards: Burn + 3 Flop
        if len(self.community_cards) != 0: # Ensure flop not already dealt
             print("ERROR: Attempting to deal flop when community cards exist."); return False
        if len(self.deck) < 4:
             print("ERROR: Not enough cards in deck for flop."); self.betting_round = self.HAND_OVER; return False
        try:
            if not self._deal_community_card(True): return False # Burn
            if not self._deal_community_card(False): return False # Flop 1
            if not self._deal_community_card(False): return False # Flop 2
            if not self._deal_community_card(False): return False # Flop 3

            self.betting_round = self.FLOP # Advance round state ONLY
            return True
        except Exception as e:
             print(f"ERROR dealing flop: {e}")
             self.betting_round = self.HAND_OVER; return False

    def deal_turn(self):
        """ Deals the turn card. (Revised: Doesn't start round) """
        # Need 2 cards: Burn + Turn
        if len(self.community_cards) != 3:
             print("ERROR: Incorrect number of community cards for turn."); return False
        if len(self.deck) < 2:
             print("ERROR: Not enough cards in deck for turn."); self.betting_round = self.HAND_OVER; return False
        try:
            if not self._deal_community_card(True): # Burn and Deal
                return False
            self.betting_round = self.TURN # Advance round state ONLY
            return True
        except Exception as e:
             print(f"ERROR dealing turn: {e}")
             self.betting_round = self.HAND_OVER; return False

    def deal_river(self):
        """ Deals the river card. (Revised: Doesn't start round) """
         # Need 2 cards: Burn + River
        if len(self.community_cards) != 4:
             print("ERROR: Incorrect number of community cards for river."); return False
        if len(self.deck) < 2:
             print("ERROR: Not enough cards in deck for river."); self.betting_round = self.HAND_OVER; return False
        try:
            if not self._deal_community_card(True): # Burn and Deal
                 return False
            self.betting_round = self.RIVER # Advance round state ONLY
            return True
        except Exception as e:
             print(f"ERROR dealing river: {e}")
             self.betting_round = self.HAND_OVER; return False


    def _check_all_active_are_allin(self):
        """ Checks if <=1 player is NOT folded AND NOT all-in AND has chips > 0.01 """
        # Re-evaluate based on current folded/all_in status and stack > 0
        count_can_still_act_voluntarily = 0
        # Check all players
        for p_idx in range(self.num_players):
             try:
                 # Check bounds for safety before accessing lists
                 if p_idx < len(self.player_folded) and not self.player_folded[p_idx] and \
                    p_idx < len(self.player_all_in) and not self.player_all_in[p_idx] and \
                    p_idx < len(self.player_stacks) and self.player_stacks[p_idx] > 0.01:
                        count_can_still_act_voluntarily += 1
             except IndexError:
                 continue # Should not happen with initial check

        # If 0 or 1 player can make a decision, betting should stop/be skipped
        return count_can_still_act_voluntarily <= 1


    def _move_to_next_player(self):
        """ Finds next active player index who can act, handles no player found. Modifies self.current_player_idx """
        if self.current_player_idx != -1: # Only move if someone's turn exists
             next_p_idx = self._get_next_active_player(self.current_player_idx)
             # Set to -1 if no next active player is found (e.g., everyone else folded/all-in)
             self.current_player_idx = next_p_idx if next_p_idx is not None else -1
        # If current_player_idx was already -1 (e.g., round ended), it stays -1


    # --- Action Handling ---
    def apply_action(self, action):
        """
        Validates and applies action to a clone, returns new state.
        Handles game progression (moving turn or advancing round) correctly. (Revised Flow V3 - Explicitly modify clone on round end)
        """
        # --- Basic Action Format Validation ---
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError(f"Action must be a tuple of length 2: {action}")
        action_type, amount_input = action

        # --- Amount Validation ---
        amount = 0.0
        try:
            # Allow numeric types directly or strings that can be converted
            if isinstance(amount_input, (int, float)) and not (np.isnan(amount_input) or np.isinf(amount_input)):
                amount = float(amount_input)
            else:
                    # Try converting string, handle potential errors
                    amount = float(amount_input)
            if amount < 0:
                    raise ValueError("Action amount cannot be negative")
        except (ValueError, TypeError):
                raise ValueError(f"Invalid action amount format or value: {amount_input}")

        acting_player_idx = self.current_player_idx

        # --- Pre-action State and Player Validation ---
        if acting_player_idx == -1:
            raise ValueError("Invalid action: No player's turn indicated (idx is -1)")
        if not (0 <= acting_player_idx < self.num_players):
                raise ValueError(f"Invalid acting_player_idx: {acting_player_idx} (num_players={self.num_players})")
        # Check list bounds for safety - necessary before accessing state lists
        if acting_player_idx >= len(self.player_folded) or \
            acting_player_idx >= len(self.player_all_in) or \
            acting_player_idx >= len(self.player_stacks):
            raise ValueError(f"Player index {acting_player_idx} out of bounds for state lists")
        # Check player status
        if self.player_folded[acting_player_idx]:
                raise ValueError(f"Invalid action: Player {acting_player_idx} has already folded")
        if self.player_all_in[acting_player_idx]:
            # If player is already all-in, they cannot act voluntarily. Just advance turn check.
            new_state_skip = self.clone() # Clone state
            new_state_skip._move_to_next_player() # Move turn indicator
            # Even after skipping, the round *could* be over if they were the last to act conceptually
            # Check the state of the CLONE after moving the player
            if new_state_skip._is_betting_round_over():
                    new_state_skip._try_advance_round() # Attempt to deal next card/go to showdown ON THE CLONE
            return new_state_skip # Return the potentially advanced clone
        # --- End pre-action checks ---


        # --- Clone and Apply Action Logic ---
        new_state = self.clone() # << CLONE is created here
        try:
            # This internal call MUTATES the 'new_state' (the clone)
            new_state._apply_action_logic(acting_player_idx, action_type, amount)
        except ValueError as e:
            # If action logic itself fails validation (e.g., invalid raise amount)
            raise # Re-raise validation errors from internal logic


        # --- Game Progression Logic ---
        # Check the state *of the clone* after the action was successfully applied

        # Case 1: Hand ended immediately due to the action (e.g., final opponent folded in _apply_action_logic)
        if new_state.betting_round == self.HAND_OVER:
            # If _apply_action_logic set the hand over
            new_state.current_player_idx = -1 # Ensure turn is off

        # Case 2: Hand did not end, check if the betting round ended ON THE CLONE
        elif new_state._is_betting_round_over():
            # If the betting round is over, attempt to advance the state ON THE CLONE
            # _try_advance_round will modify the betting_round and current_player_idx of new_state
            new_state._try_advance_round()

        # Case 3: Hand did not end, and betting round did not end
        else:
            # If the round is NOT over, simply move to the next active player ON THE CLONE
            new_state._move_to_next_player()

        # --- RETURN THE MODIFIED CLONE ---
        # The new_state object now reflects the state AFTER the action AND any subsequent round/turn advancement
        return new_state

    def _apply_action_logic(self, p_idx, action_type, amount):
        """ Internal logic, MUTATES self state based on validated action. """
        # Check bounds just in case internal state is inconsistent
        if not (0 <= p_idx < self.num_players and \
                p_idx < len(self.player_stacks) and \
                p_idx < len(self.player_bets_in_round)):
             raise IndexError(f"Invalid player index {p_idx} in _apply_action_logic")

        player_stack = self.player_stacks[p_idx]
        current_round_bet = self.player_bets_in_round[p_idx] # How much p_idx already put in *this round*
        self.players_acted_this_round.add(p_idx) # Mark as acted in this sequence of actions
        action_log_repr = f"P{p_idx}:" # Start building log string

        # --- Fold ---
        if action_type == "fold":
            self.player_folded[p_idx] = True
            action_log_repr += "f"
            # Check if hand ends now (only one player left not folded)
            # Check against self.num_players as loop bound is safer than list len if lists differ
            unfolded_count = 0
            for i in range(self.num_players):
                 if i < len(self.player_folded) and not self.player_folded[i]:
                      unfolded_count += 1
            if unfolded_count <= 1:
                self.betting_round = self.HAND_OVER
                self.current_player_idx = -1 # Hand is over

        # --- Check ---
        elif action_type == "check":
            # Validate check: Current bet level must be <= player's bet already in round (allow tolerance)
            if self.current_bet - current_round_bet > 0.01:
                raise ValueError(f"Invalid check P{p_idx}: Facing bet={self.current_bet:.2f}, HasBet={current_round_bet:.2f}")
            action_log_repr += "k" # Log as check

        # --- Call ---
        elif action_type == "call":
            amount_needed = self.current_bet - current_round_bet # How much more needed to match current bet level
            # If no amount needed, treat as check conceptually (but still log as call if input was call)
            if amount_needed <= 0.01:
                action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}" # Log call (potentially of 0 effective amount) with total in round
            else: # Actual call needed
                call_cost = min(amount_needed, player_stack) # Cost to call is capped by stack
                if call_cost < 0: call_cost = 0 # Safety
                self._deduct_bet(p_idx, call_cost) # Deduct the actual cost
                # Log the TOTAL amount the player has IN THE POT for this round after the call
                action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}"

        # --- Bet (Opening Bet on a street) ---
        elif action_type == "bet":
            # Validate: Only allowed if no bet has been made yet this round (current_bet is 0 or negligible)
            if self.current_bet > 0.01:
                raise ValueError("Invalid bet: Use raise instead as there is a facing bet.")
            if amount < 0.01: # Amount here is the intended *size* of the bet
                 raise ValueError("Bet amount must be positive.")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                 raise ValueError("Max raises/bets reached for this street.")

            # Determine minimum legal bet size (usually BB, but can be less if stack is small post-flop)
            min_bet_amount = self.big_blind if self.betting_round == self.PREFLOP else 1.0
            # Bet size is capped by stack
            actual_bet_amount = min(amount, player_stack)
            is_all_in = abs(actual_bet_amount - player_stack) < 0.01

            # Check min bet size (unless going all-in for less than min bet)
            if actual_bet_amount < min_bet_amount - 0.01 and not is_all_in:
                raise ValueError(f"Bet {actual_bet_amount:.2f} is less than minimum {min_bet_amount:.2f}")

            # Apply bet (amount is the cost)
            self._deduct_bet(p_idx, actual_bet_amount)
            action_log_repr += f"b{int(round(actual_bet_amount))}" # Log the bet amount

            # Update betting state: New level is the bet amount, player is last raiser
            new_total_bet_level_this_round = self.player_bets_in_round[p_idx] # Total put in THIS ROUND after bet
            self.current_bet = new_total_bet_level_this_round # Set the level to call
            self.last_raise = new_total_bet_level_this_round # The size of this first aggression
            self.last_raiser = p_idx
            self.raise_count_this_street = 1 # This is the first bet/raise this street
            # Action re-opened, clear list of who acted *relative to this aggression*
            self.players_acted_this_round = {p_idx}
            # Note: is_all_in handled by _deduct_bet

        # --- Raise (Increasing a previous bet/raise) ---
        elif action_type == "raise":
            # Validate: Only allowed if there's a current bet level to raise over
            if self.current_bet <= 0.01:
                raise ValueError("Invalid raise: Use bet instead as there is no facing bet.")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                 raise ValueError("Max raises reached for this street.")

            # Amount here is the TOTAL target level player wants to bet TO in this round
            total_bet_target = amount
            # Cost for the player is the difference between the target and what they already have in
            cost_to_reach_target = total_bet_target - current_round_bet
            # Check if raise is valid (must increase the bet)
            if cost_to_reach_target <= 0.01:
                 raise ValueError(f"Raise target {total_bet_target:.2f} not greater than current bet in round {current_round_bet:.2f}")
            # Check affordability (allow tiny margin for float issues)
            if cost_to_reach_target > player_stack + 0.01:
                 raise ValueError(f"Player {p_idx} cannot afford raise cost {cost_to_reach_target:.2f} with stack {player_stack:.2f}")

            # Actual cost paid by player (capped by stack)
            actual_raise_cost = min(cost_to_reach_target, player_stack)
            # The total bet level player *actually* reaches after paying cost
            actual_total_bet_reached = current_round_bet + actual_raise_cost
            is_all_in = abs(actual_raise_cost - player_stack) < 0.01

            # Check legality of raise size increment:
            # Increment must be >= last raise size OR player is all-in
            min_legal_increment = max(self.last_raise, self.big_blind) # Min increase needed
            # The actual increase OVER the CURRENT BET LEVEL
            actual_increment_made = actual_total_bet_reached - self.current_bet

            # Check if increment is sufficient (unless all-in)
            if actual_increment_made < min_legal_increment - 0.01 and not is_all_in:
                # Raise is too small (and not an all-in)
                raise ValueError(f"Raise increment {actual_increment_made:.2f} is less than minimum legal increment {min_legal_increment:.2f} (Target={total_bet_target}, Current={self.current_bet}, LastRaise={self.last_raise})")

            # Apply the raise cost
            self._deduct_bet(p_idx, actual_raise_cost)
            # Log the TOTAL amount the player is raised TO in this round
            action_log_repr += f"r{int(round(actual_total_bet_reached))}"

            # --- Update betting state ---
            # Size of THIS raise increment (used for next min raise check)
            self.last_raise = actual_increment_made # Amount OVER the previous bet level
            # New total bet level that others must call TO
            self.current_bet = actual_total_bet_reached
            self.last_raiser = p_idx
            self.raise_count_this_street += 1
            # Action re-opened, clear who has acted *relative to this new aggression*
            self.players_acted_this_round = {p_idx}
            # Note: is_all_in handled by _deduct_bet

        # --- Unknown Action ---
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # --- Add action to history ---
        # Check length because p_idx might be > 9
        if len(action_log_repr) > len(f"P{p_idx}:"):
            self.action_sequence.append(action_log_repr)


    def get_betting_history(self):
         """ Returns the sequence of actions as a single semicolon-separated string. """
         return ";".join(self.action_sequence)

    def get_available_actions(self):
        """ Calculates and returns a list of legal actions for the current player. """
        actions = []
        player_idx = self.current_player_idx

        # --- Basic Validity Checks ---
        if player_idx == -1: return [] # No one's turn
        try:
            if not (0 <= player_idx < self.num_players and \
                    player_idx < len(self.player_folded) and \
                    player_idx < len(self.player_all_in) and \
                    player_idx < len(self.player_stacks) and \
                    player_idx < len(self.player_bets_in_round)):
                 print(f"WARN get_available_actions: Index {player_idx} out of bounds"); return []
            if self.player_folded[player_idx] or \
               self.player_all_in[player_idx] or \
               self.player_stacks[player_idx] < 0.01:
                return [] # Cannot act
        except IndexError:
            print(f"WARN get_available_actions: IndexError accessing state for P{player_idx}"); return []

        # --- Get Relevant State ---
        player_stack = self.player_stacks[player_idx]
        current_round_bet = self.player_bets_in_round[player_idx] # Already in pot this round
        current_bet_level = self.current_bet # Current highest total bet level in round

        # --- Fold Action ---
        actions.append(("fold", 0))

        # --- Check or Call Action ---
        amount_to_call = current_bet_level - current_round_bet # Additional amount needed
        can_check = amount_to_call < 0.01

        if can_check:
            actions.append(("check", 0))
        else: # Call is needed
            call_cost = min(amount_to_call, player_stack) # Can only call up to stack size
            if call_cost > 0.01: # Only add if it costs something significant
                # Action tuple format: (type, TOTAL_AMOUNT_IN_ROUND_AFTER_ACTION) - NO, GameState uses action cost or total TO
                # Let's be consistent with how GameState seems to use amounts.
                # Call takes the *cost* amount (relative) for get_available_actions output
                # Use ROUNDED INT for amount in tuple as per other action examples
                actions.append(("call", int(round(call_cost))))


        # --- Bet or Raise Actions (Aggression) ---
        # Check raise cap FIRST
        can_aggress = self.raise_count_this_street < self.MAX_RAISES_PER_STREET
        # Can only aggress if stack is enough to make a bet/raise *greater* than call cost
        effective_call_cost = max(0.0, amount_to_call) # Cost to just match
        can_afford_increase = player_stack > effective_call_cost + 0.01

        if can_aggress and can_afford_increase:
            # Define min/max possible TOTAL amounts TO bet/raise TO this round
            min_legal_target = 0.0
            max_legal_target = current_round_bet + player_stack # All-in target

            if current_bet_level < 0.01: # Situation is BETTING (no prior aggression this street)
                action_prefix = "bet"
                # Min bet COST is BB (or 1 postflop), capped by stack
                min_bet_cost = max(self.big_blind if self.betting_round == self.PREFLOP else 1.0, 1.0) # Ensure positive
                min_bet_cost_capped = min(player_stack, min_bet_cost)
                # Min TARGET amount is current bet (0) + min cost
                min_legal_target = current_round_bet + min_bet_cost_capped

            else: # Situation is RAISING
                action_prefix = "raise"
                # Minimum legal raise increment size
                min_legal_increment = max(self.last_raise, self.big_blind)
                # Minimum total amount player must raise TO
                min_raise_target_unbounded = current_bet_level + min_legal_increment
                # Ensure minimum target is AT LEAST the minimum raise, but capped by all-in amount
                min_legal_target = min(max_legal_target, min_raise_target_unbounded)

            # --- Add Specific Actions ---
            # Add MIN Legal Action: (must be greater than current bet level)
            if min_legal_target > current_bet_level + 0.01:
                # Action tuple: (type, TOTAL_AMOUNT_TO_RAISE_TO) -> Rounded Int
                actions.append((action_prefix, int(round(min_legal_target))))

            # Add ALL-IN Action (if it's aggressive and different from min raise)
            # Check if going all-in actually increases the bet level
            is_all_in_aggressive = (max_legal_target > current_bet_level + 0.01)
            # Check if all-in amount is distinct from min legal target (avoid duplicates)
            is_all_in_distinct = abs(max_legal_target - min_legal_target) > 0.01

            if is_all_in_aggressive and is_all_in_distinct:
                 actions.append((action_prefix, int(round(max_legal_target))))

        # --- Final Filtering and Sorting ---
        def sort_key(action_tuple):
            action_type, amount = action_tuple
            order = {"fold":0, "check":1, "call":2, "bet":3, "raise":4}
            sort_amount = amount if isinstance(amount, (int, float)) else 0
            return (order.get(action_type, 99), sort_amount)

        # Filter out duplicates based on (type, rounded_amount)
        final_actions_set = set()
        final_actions_list = []
        # Sort before filtering duplicates to potentially keep smallest call if multiple existed
        for act_tuple in sorted(actions, key=sort_key):
             act_type, act_amount = act_tuple
             action_key_repr = (act_type, int(round(act_amount)))

             if action_key_repr not in final_actions_set:
                 # Additional check: ensure player can afford the action represented by the tuple amount
                 can_afford = False
                 if act_type in ['fold', 'check']: can_afford = True
                 elif act_type == 'call': # Amount is the cost
                      cost = act_amount
                      can_afford = cost <= player_stack + 0.01
                 elif act_type in ['bet', 'raise']: # Amount is the target TO reach
                      cost = act_amount - current_round_bet
                      can_afford = cost <= player_stack + 0.01

                 if can_afford:
                     final_actions_list.append(act_tuple)
                     final_actions_set.add(action_key_repr)

        # Re-sort the final list after filtering affordability and duplicates
        return sorted(final_actions_list, key=sort_key)


    # --- MODIFIED METHOD: with Debug Logs ---
    def _is_betting_round_over(self):
        """ Checks if the current betting round has concluded based on player actions and bets. (V2 Add Debug Logs, Refine Case 3 logic) """
        # --- Add Entry Debug Log ---
        try:
            acted_set_str = "{" + ",".join(map(str, sorted(list(self.players_acted_this_round)))) + "}" if self.players_acted_this_round else "{}"
            print(f"DEBUG is_round_over?: ENTRY - Rnd={self.betting_round}, Turn={self.current_player_idx}, CurrentBet={self.current_bet:.2f}, Acted={acted_set_str}, LastRaiser={self.last_raiser}, RaiseCount={self.raise_count_this_street}")
        except Exception as log_e:
            print(f"DEBUG is_round_over? ENTRY Log Err: {log_e}")
        # --- End Entry Debug Log ---


        # --- Check for Immediate Hand End (Folds) ---
        # Get players currently not folded
        eligible_players = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(eligible_players) <= 1:
            print(f"DEBUG is_round_over?: True (<=1 eligible player left)")
            return True


        # --- Identify players who can still voluntarily act THIS ROUND ---
        # Must NOT be folded, NOT be all-in already, AND have stack > 0
        players_who_can_voluntarily_act = []
        for p_idx in eligible_players: # Iterate only over non-folded players
            # Safety check bounds before accessing lists
            if 0 <= p_idx < self.num_players and \
                p_idx < len(self.player_all_in) and not self.player_all_in[p_idx] and \
                p_idx < len(self.player_stacks) and self.player_stacks[p_idx] > 0.01:
                    players_who_can_voluntarily_act.append(p_idx)

        num_can_act = len(players_who_can_voluntarily_act)
        print(f"DEBUG is_round_over?: CanAct={players_who_can_voluntarily_act} (Count={num_can_act})")


        # --- Case 1: Zero players can voluntarily act ---
        # This means all players who aren't folded are already all-in. Betting must stop.
        if num_can_act == 0:
            # Exception: If *only one* player remained unfolded initially, Case 0 already caught it.
            # If there are >= 2 non-folded players, but none can act, betting stops.
            if len(eligible_players) >= 2:
                 print(f"DEBUG is_round_over?: True (>=2 unfolded, but 0 can act voluntarily -> All-in showdown)")
                 return True
            else:
                 # This case should be covered by the len(eligible_players) <= 1 check at the top
                 print(f"DEBUG is_round_over?: True (Edge Case: 0 can act, likely also <=1 eligible)")
                 return True


        # --- Case 2: Only one player can voluntarily act ---
        # Round ends *unless* it's the BB preflop facing no raise and hasn't acted yet (BB option).
        if num_can_act == 1:
            the_player = players_who_can_voluntarily_act[0]
            has_acted_this_round = the_player in self.players_acted_this_round
            # Check if they face a bet requiring action (allow for float tolerance)
            # Compare their bet in the round vs the current required bet level
            facing_bet = (self.current_bet - self.player_bets_in_round[the_player]) > 0.01

            # Check conditions for the special preflop Big Blind option
            is_preflop = (self.betting_round == self.PREFLOP)
            # Find BB robustly - depends on num_players
            bb_player_idx = None
            # Use self.num_players to calculate relative position reliably
            if self.num_players == 2: # HU Case: Player after dealer is BB
                # In HU, the dealer is SB, player 1 pos after dealer is BB
                bb_player_idx = (self.dealer_position + 1) % self.num_players
            elif self.num_players > 2: # 3+ players: Player 2 positions after dealer is BB
                bb_player_idx = (self.dealer_position + 2) % self.num_players

            is_bb_player = (the_player == bb_player_idx) if bb_player_idx is not None else False

            # Check if action has closed preflop without a re-raise:
            # This implies only the initial blind post happened (raise count <= 1)
            # AND the last raiser was the BB poster.
            initial_blind_raiser = bb_player_idx # Assumes BB always posts something or goes all-in

            # A raise happened if raise_count > 1 OR if raise_count==1 but last_raiser isn't the initial BB poster
            # Check is simpler: if current_bet is still just the BB amount, no raise happened beyond blinds.
            is_facing_only_initial_blind = (abs(self.current_bet - self.big_blind) <= 0.01 and self.raise_count_this_street <= 1)
            # More robust check might compare self.last_raiser == bb_player_idx, but needs care if BB was short

            # Log info for debugging Case 2
            print(f"DEBUG is_round_over? Case2 Info: P{the_player}, hasActed={has_acted_this_round}, facingBet={facing_bet}, isPreflop={is_preflop}, isBB={is_bb_player}({bb_player_idx}), facingOnlyBlinds={is_facing_only_initial_blind}(Raiser={self.last_raiser},Count={self.raise_count_this_street})")

            # BB Option Rule: If it is preflop, AND the player IS the big blind,
            # AND they are NOT facing a raise (only initial blinds effectively),
            # AND they haven't acted yet in this sequence of actions... then the round is NOT over.
            if is_preflop and is_bb_player and not facing_bet and is_facing_only_initial_blind and not has_acted_this_round:
                print(f"DEBUG is_round_over? Case2 Result=False (BB Option applies)")
                return False # BB still has the option to act

            # Otherwise (for all other Case 2 scenarios), the round ends.
            # This player is the only one left who could act. There's no one else to act after them.
            print(f"DEBUG is_round_over? Case2 Result=True (Only one player left who can act, action closes)")
            return True


        # --- Case 3: Multiple players can act ---
        # The round ends if **ALL** players who can still voluntarily act
        # have EITHER matched the current bet level OR are all-in,
        # AND have acted at least once *since the last aggressive action* (i.e., are in self.players_acted_this_round).

        all_acted_since_last_aggression = True
        all_bets_matched_or_all_in = True

        for p_idx in players_who_can_voluntarily_act:
            # Check 1: Has player acted since last aggression?
            # If players_acted_this_round is empty, this means the action is starting
            # or it's on the person who just made the last aggressive action.
            if not self.players_acted_this_round:
                # This state should not happen if multiple players can act and round didn't just start.
                # If round JUST started (postflop), players_acted is empty, so this needs care.
                # Let's assume if the set is empty, no one has acted yet relative to current state.
                 all_acted_since_last_aggression = False # Not everyone acted yet
                 print(f"DEBUG is_round_over? Case3 Check P{p_idx}: Acted=False (Set Empty)")
                 break # Exit loop early, round cannot be over
            elif p_idx not in self.players_acted_this_round:
                 all_acted_since_last_aggression = False
                 print(f"DEBUG is_round_over? Case3 Check P{p_idx}: Acted=False (Not in {self.players_acted_this_round})")
                 break # Exit loop early, round cannot be over


            # Check 2: Is player's bet matched OR are they all-in?
            player_bet_r = self.player_bets_in_round[p_idx]
            # Check if bet is essentially equal to current level
            is_bet_matched = abs(player_bet_r - self.current_bet) <= 0.01
            # If bet is not matched, check if they are all-in
            is_player_all_in = self.player_all_in[p_idx]

            # Player has effectively finished their action for this level if
            # their bet matches OR they are all-in (even if for less)
            is_action_complete = is_bet_matched or is_player_all_in

            print(f"DEBUG is_round_over? Case3 Check P{p_idx}: Bet={player_bet_r:.2f} vs Curr={self.current_bet:.2f}, isAllIn={is_player_all_in} -> ActionComplete={is_action_complete}, HasActed={p_idx in self.players_acted_this_round}")

            if not is_action_complete:
                all_bets_matched_or_all_in = False
                break # Exit loop early, round cannot be over

        # The round is over only if BOTH conditions are true for ALL players who can act
        result_c3 = all_acted_since_last_aggression and all_bets_matched_or_all_in
        print(f"DEBUG is_round_over? Case3 Final Check: all_acted={all_acted_since_last_aggression}, all_matched={all_bets_matched_or_all_in} -> Result={result_c3}")
        return result_c3


    # --- MODIFIED METHOD: with Debug Logs ---
    def _try_advance_round(self):
        """
        Attempts to deal next street OR start next betting round OR end hand.
        Called when a betting round concludes or should be skipped. MUTATES state. (Revised V2 - Add Debug Logs)
        """
        print(f"DEBUG try_advance: ENTER - CurrentRnd={self.betting_round}") # Log Entry

        # --- Check for Immediate Hand End (Folds) ---
        eligible_players = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(eligible_players) <= 1:
            print(f"DEBUG try_advance: Hand ended due to folds.")
            if self.betting_round != self.HAND_OVER:
                self.betting_round = self.HAND_OVER
            self.current_player_idx = -1 # Hand is over
            self.players_acted_this_round = set() # Clear acted set for safety
            return # Hand is over

        # --- Check if Further Betting Rounds Should Be Skipped (All-Ins) ---
        # This check is crucial *before* dealing the next street
        should_skip_further_betting = self._check_all_active_are_allin()
        print(f"DEBUG try_advance: Check all-in status -> SkipFurtherBetting={should_skip_further_betting}")

        current_round = self.betting_round
        next_round = -1 # Undetermined yet
        dealt_successfully = True # Assume success unless deal fails

        # --- Determine Next State (Deal or Showdown) ---
        if current_round == self.PREFLOP:
            print(f"DEBUG try_advance: Dealing Flop...")
            if not self.deal_flop(): dealt_successfully = False
            else: next_round = self.FLOP # Record expected next state if deal works

        elif current_round == self.FLOP:
            print(f"DEBUG try_advance: Dealing Turn...")
            if not self.deal_turn(): dealt_successfully = False
            else: next_round = self.TURN

        elif current_round == self.TURN:
            print(f"DEBUG try_advance: Dealing River...")
            if not self.deal_river(): dealt_successfully = False
            else: next_round = self.RIVER

        elif current_round == self.RIVER:
            # After River betting round FINISHES (or is skipped), always go to Showdown
            print(f"DEBUG try_advance: Advancing from River to Showdown.")
            self.betting_round = self.SHOWDOWN # Update state directly
            next_round = self.SHOWDOWN # Note the final state

        else: # Already in SHOWDOWN or HAND_OVER, do nothing more
            print(f"DEBUG try_advance: EXIT - Already in Showdown/HandOver ({self.betting_round}).")
            return


        # --- Handle Deal Failures ---
        if not dealt_successfully:
            print(f"DEBUG try_advance: EXIT - Deal Failed. Setting Hand Over.")
            if self.betting_round != self.HAND_OVER: # Avoid redundant set
                self.betting_round = self.HAND_OVER
            self.current_player_idx = -1
            self.players_acted_this_round = set()
            return


        # --- State Reached After Successful Deal (or directly from River) ---

        if self.betting_round == self.SHOWDOWN:
            # If we reached showdown (either normally or via skip)
            print(f"DEBUG try_advance: EXIT - Reached Showdown state. Setting Turn=-1.")
            self.current_player_idx = -1 # No more actions
            self.players_acted_this_round = set()
            return

        elif self.betting_round == self.HAND_OVER:
            # If dealing somehow failed and explicitly set HAND_OVER
            print(f"DEBUG try_advance: EXIT - Deal function set HAND_OVER.")
            self.current_player_idx = -1
            self.players_acted_this_round = set()
            return


        # --- Decide Whether to Start Next Betting Round ---
        # At this point, we've successfully dealt Flop/Turn/River (or determined Showdown)

        if not should_skip_further_betting:
            # Betting is needed for the street just dealt
            print(f"DEBUG try_advance: Attempting _start_betting_round for newly dealt Rnd={self.betting_round}...")
            self._start_betting_round() # Sets current_player_idx
            print(f"DEBUG try_advance: After _start_betting_round, next Turn is P{self.current_player_idx}")
            # If _start_betting_round fails (e.g., finds no one to act), it will set current_player_idx=-1
        else:
            # We dealt the next street, but betting should be skipped for it (all-in)
            print(f"DEBUG try_advance: SkipBetting=True for Rnd={self.betting_round}. Setting Turn=-1.")
            self.current_player_idx = -1 # Ensure no player is set to act
            self.players_acted_this_round = set()

            # If betting was skipped on Flop/Turn, we need to immediately proceed
            # to deal the *next* street (or go to showdown if river was skipped)
            if self.betting_round < self.RIVER:
                 print(f"DEBUG try_advance: SkipBetting=True, Rnd={self.betting_round} < River. RECURSING _try_advance_round...")
                 self._try_advance_round() # Recursive call to deal the next street
            elif self.betting_round == self.RIVER: # If River betting was skipped
                 print(f"DEBUG try_advance: SkipBetting=True after River dealt. Advancing to SHOWDOWN.")
                 self.betting_round = self.SHOWDOWN
                 self.current_player_idx = -1 # Ensure turn is off

        # --- END of try_advance ---


    # --- UNCHANGED METHODS BELOW (Keep from original file) ---

    def is_terminal(self):
        """ Checks if the game hand has reached a terminal state. """
        # Hand ends if only one player remains unfolded
        # Check against num_players for robustness if list indices fail
        unfolded_count = 0
        for i in range(self.num_players):
             try:
                 if i < len(self.player_folded) and not self.player_folded[i]:
                      unfolded_count += 1
             except IndexError: pass # Ignore players out of bounds of list
        if unfolded_count <= 1:
            return True
        # Hand ends if we have reached or passed the showdown stage
        if self.betting_round >= self.SHOWDOWN:
             return True
        return False


    def get_utility(self, player_idx, initial_stacks=None):
        """ Calculates the utility (profit/loss) for a player at the end of a terminal hand. """
        if not self.is_terminal():
             # print(f"WARN get_utility: Called on non-terminal state for P{player_idx}")
             return 0.0
        if initial_stacks is None:
             print(f"ERROR get_utility: initial_stacks missing for P{player_idx}. Returning 0.")
             return 0.0
        # Validate inputs
        if not (0 <= player_idx < self.num_players and \
                isinstance(initial_stacks, list) and \
                len(initial_stacks) == self.num_players and \
                player_idx < len(self.player_stacks)):
             print(f"WARN get_utility: Index or stack list mismatch for P{player_idx}")
             return 0.0

        # Get initial stack safely
        initial_stack = 0.0
        try:
            i_s = initial_stacks[player_idx]
            # Check type and for NaN/Inf
            if not isinstance(i_s, (int, float)) or np.isnan(i_s) or np.isinf(i_s):
                 raise ValueError("Invalid initial stack value")
            initial_stack = float(i_s)
        except (IndexError, TypeError, ValueError) as e:
            print(f"WARN get_utility: Invalid initial stack for P{player_idx}: {e}")
            return 0.0

        # --- Perform internal win determination ON A CLONE ---
        # This avoids mutating the state that might be needed elsewhere
        # Create a copy specifically for final calculation
        try:
             final_state_calculator = self.clone()
             # Ensure the calculator reflects the *current* state BEFORE distribution
             # Note: Determine winners *mutates* the stack on the object it's called on
             _ = final_state_calculator.determine_winners() # Call on the clone

             # Get the final stack *from the modified clone*
             final_effective_stack = final_state_calculator.player_stacks[player_idx]
             # Validate the final stack
             if not isinstance(final_effective_stack, (int,float)) or np.isnan(final_effective_stack) or np.isinf(final_effective_stack):
                  raise ValueError("Invalid final stack value after internal win determination")

        except Exception as win_err:
             print(f"ERROR get_utility: Internal win determination failed for P{player_idx}: {win_err}")
             # Fallback: Use the current stack *before* trying distribution if error occurred
             # Must read current stack from original 'self' object
             current_stack = 0.0
             try:
                 c_s = self.player_stacks[player_idx]
                 if not isinstance(c_s, (int,float)) or np.isnan(c_s) or np.isinf(c_s):
                      raise ValueError("Invalid current stack value (fallback)")
                 current_stack = float(c_s)
             except (IndexError, TypeError, ValueError):
                  # Cannot even get current stack reliably
                  print(f"WARN get_utility: Error getting fallback current stack for P{player_idx}")
                  return 0.0 # Give up
             final_effective_stack = current_stack # Use pre-distribution stack as best guess

        # Utility is the change in stack size
        utility = final_effective_stack - initial_stack

        # Final safety check for NaN/Inf
        if np.isnan(utility) or np.isinf(utility):
            # print(f"WARN get_utility: Calculated utility is NaN/Inf for P{player_idx}. Returning 0.")
            utility = 0.0

        return utility


    def determine_winners(self, player_names=None): # player_names is currently unused but kept for potential API use
        """
        Determines the winner(s) of the hand, calculates side pots, and updates player stacks.
        MUTATES the game state (self.player_stacks, self.pot).
        Returns a list summarizing pot distribution.
        """
        if not self.is_terminal():
             # print("WARN: determine_winners called on non-terminal state.")
             return [] # Cannot determine winners yet

        # If pot is negligible, nothing to distribute
        if self.pot < 0.01:
             self.pot = 0.0 # Ensure pot is zeroed
             return []

        # Make local copy of pot to distribute, zero out state pot
        total_pot_to_distribute = self.pot
        self.pot = 0.0
        pots_summary = [] # To store summary of each pot distribution

        # Identify players still eligible for the pot (not folded)
        eligible_for_pot = [p for p in range(self.num_players) if not self.player_folded[p]]

        # Case 1: Uncontested pot (everyone else folded)
        if len(eligible_for_pot) == 1:
            winner_idx = eligible_for_pot[0]
            amount_won = total_pot_to_distribute
            if 0 <= winner_idx < len(self.player_stacks):
                self.player_stacks[winner_idx] += amount_won
                pots_summary = [{'winners': [winner_idx], 'amount': amount_won, 'eligible': [winner_idx], 'desc': 'Uncontested'}]
            else:
                print(f"ERROR determine_winners: Uncontested winner index {winner_idx} out of bounds.")
            return pots_summary

        # Case 2: Showdown required
        evaluated_hands = {}
        # Identify players who need to show cards (not folded AND have cards)
        showdown_players = []
        for p_idx in eligible_for_pot:
            # Ensure index valid for hole_cards list
            if 0 <= p_idx < len(self.hole_cards) and len(self.hole_cards[p_idx]) == 2:
                 showdown_players.append(p_idx)
            # else: Player folded or has invalid cards - not included in showdown evaluation

        # Need at least 5 cards total for evaluation
        if len(self.community_cards) < 3: # Must have at least flop to evaluate
            print("ERROR determine_winners: Showdown required but not enough community cards dealt.")
            # Return pot? Likely an error state, perhaps leave stacks unchanged.
            return []

        # Evaluate hands for all showdown players
        valid_showdown_players = [] # Store players whose hands evaluated successfully
        for p_idx in showdown_players:
            all_cards_for_eval = self.hole_cards[p_idx] + self.community_cards
            if len(all_cards_for_eval) < 5: continue # Skip if somehow < 5 cards (should be >=5 post-flop)
            try:
                evaluated_hands[p_idx] = HandEvaluator.evaluate_hand(all_cards_for_eval)
                valid_showdown_players.append(p_idx)
            except Exception as eval_err:
                 print(f"WARN determine_winners: Hand evaluation failed for P{p_idx}: {eval_err}")
                 continue # Skip players whose hands cannot be evaluated

        # If no valid hands could be evaluated for showdown
        if not valid_showdown_players:
            print("WARN determine_winners: No valid hands found for showdown among eligible players.")
            # Return pot? Let it vanish? Add back to player_total_bets? Leave stacks as is for now.
            return []


        # Calculate side pots based on contributions of VALID showdown players
        # Contributions: list of (player_index, total_bet_this_hand) sorted by bet amount
        contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players], key=lambda x: x[1])

        side_pots = [] # Stores {'amount': float, 'eligible': list_of_player_indices}
        last_contribution_level = 0.0
        # Make a copy of players initially eligible for the first (potentially main) pot
        eligible_for_current_pot = valid_showdown_players[:]
        processed_amount = 0.0 # Track how much pot is assigned to side pots

        for p_idx_sp, total_contribution in contributions:
            # Calculate how much more this player contributed compared to the previous level
            contribution_increment = total_contribution - last_contribution_level
            # If this player contributed more...
            if contribution_increment > 0.01:
                 # Create a pot layer for this increment amount
                 num_eligible_at_this_level = len(eligible_for_current_pot)
                 # Pot amount for this layer = increment * number of players still active for this layer
                 pot_amount = contribution_increment * num_eligible_at_this_level
                 processed_amount += pot_amount

                 if pot_amount > 0.01:
                     # Add this side pot info (amount, and WHO is eligible for THIS specific pot layer)
                     side_pots.append({'amount': pot_amount, 'eligible': eligible_for_current_pot[:]}) # Use copy of list

                 last_contribution_level = total_contribution # Update the level for next comparison

            # This player, having contributed fully up to this level, is no longer eligible
            # for subsequent, *smaller* pot layers (that they over-contributed to).
            # Correct logic: Players are removed from *eligibility* for *future* pots
            # as we process contributions from lowest to highest.
            # Once a player's contribution level is processed, they are *removed* from the
            # eligibility list for the *next* contribution increment's pot.
            if p_idx_sp in eligible_for_current_pot:
                eligible_for_current_pot.remove(p_idx_sp)


        # --- Check for Discrepancy and Distribute ---
        if abs(processed_amount - total_pot_to_distribute) > 0.1: # Allow tolerance
           print(f"WARN determine_winners: Discrepancy Pot={total_pot_to_distribute:.2f} vs SidePotsSum={processed_amount:.2f}")
           # Optional: Adjust last pot or distribute remainder? For now, proceed with calculated side pots.

        # --- Award the calculated pots ---
        distributed_total = 0 # Track total distributed from side pots
        for i, pot_info in enumerate(side_pots):
            pot_amount = pot_info.get('amount', 0.0)
            eligible_players_this_pot = pot_info.get('eligible', [])

            if pot_amount < 0.01 or not eligible_players_this_pot:
                 continue # Skip empty pots or pots with no eligible players

            # Find the best hand among players eligible for THIS pot layer
            # Filter evaluated hands based on eligibility for *this pot*
            eligible_evaluated_hands = {p: evaluated_hands[p] for p in eligible_players_this_pot if p in evaluated_hands}

            if not eligible_evaluated_hands:
                 print(f"WARN determine_winners: No evaluated hands found for Pot {i+1} eligible: {eligible_players_this_pot}")
                 continue # Skip if no valid hands among eligible players

            # Find the best hand value within this eligible group
            best_hand_value = max(eligible_evaluated_hands.values())
            # Find all players *eligible for this pot* who have the best hand value
            pot_winners = [p for p, hand_val in eligible_evaluated_hands.items() if hand_val == best_hand_value]

            if pot_winners:
                 # Divide this specific pot amount among the winners for this pot
                 winner_share = pot_amount / len(pot_winners)
                 distributed_total += pot_amount # Track total distributed from these calculations

                 # Award winnings
                 for w_idx in pot_winners:
                     # Check bounds before adding winnings
                     if 0 <= w_idx < len(self.player_stacks):
                          self.player_stacks[w_idx] += winner_share
                     else: print(f"ERROR: Winner index {w_idx} out of bounds when awarding Pot {i+1}")


                 # Create summary entry for this pot
                 pot_desc = f"Side Pot {i+1}" if len(side_pots) > 1 and i < len(side_pots)-1 else "Main Pot" # Basic naming
                 # Note: More sophisticated naming could track which player created the side pot limit
                 pots_summary.append({'winners':pot_winners, 'amount':pot_amount, 'eligible':eligible_players_this_pot, 'desc': pot_desc})

            else: # Should not happen if eligible_evaluated_hands was not empty
                 print(f"WARN determine_winners: No winners found for Pot {i+1}")


        # --- Final check on distribution ---
        # Note: Small discrepancies are normal due to floating point arithmetic
        # If abs(distributed_total - total_pot_to_distribute) > 0.1:
        #     print(f"WARN determine_winners: Final Check: Distributed {distributed_total:.2f} != Total Pot {total_pot_to_distribute:.2f}")


        return pots_summary # Return summary of pot distributions


    def clone(self):
        """ Creates a deep copy of the game state. """
        # Using deepcopy is generally safest for complex objects with nested structures
        return deepcopy(self)

    def get_position(self, player_idx):
        """ Calculates the position relative to the dealer (0=dealer, 1=SB, etc.). """
        if not (0 <= player_idx < self.num_players) or self.num_players <= 1:
            return -1 # Invalid input or not meaningful
        # Position relative to dealer (dealer is pos 0)
        return (player_idx - self.dealer_position + self.num_players) % self.num_players

    def __str__(self):
        """ Provides a string representation of the current game state. """
        round_name = self.ROUND_NAMES.get(self.betting_round, f"R{self.betting_round}")
        turn = f"P{self.current_player_idx}" if self.current_player_idx != -1 else "None"
        board_list = self.community_cards if hasattr(self, 'community_cards') else []
        board = ' '.join(map(str, board_list)) if board_list else "-"
        # Limit history length for display
        hist_limit = 60
        hist = self.get_betting_history()
        hist_display = f"...{hist[-hist_limit:]}" if len(hist) > hist_limit else hist

        lines = []
        lines.append(f"Round: {round_name}({self.betting_round}), Turn: {turn}, Pot: {self.pot:.2f}, Board: [{board}]")
        lines.append(f"Dealer: P{self.dealer_position}, BetLevel: {self.current_bet:.2f}, LastRaiser: {self.last_raiser}, RaiseCnt: {self.raise_count_this_street}, Acted: {self.players_acted_this_round}")

        for i in range(self.num_players):
            # Check bounds before accessing player state
            state_flags = []
            if hasattr(self, 'player_stacks') and i < len(self.player_stacks) and \
               hasattr(self, 'player_folded') and i < len(self.player_folded) and \
               hasattr(self, 'player_all_in') and i < len(self.player_all_in) and \
               hasattr(self, 'player_bets_in_round') and i < len(self.player_bets_in_round):

                if i == self.dealer_position: state_flags.append("D")
                if self.player_folded[i]: state_flags.append("F")
                if self.player_all_in[i]: state_flags.append("A")

                state_str = "".join(state_flags) if state_flags else " "
                stack_str = f"{self.player_stacks[i]:.0f}"
                bet_rnd_str = f"{self.player_bets_in_round[i]:.0f}"
                bet_hand_str = f"{self.player_total_bets_in_hand[i]:.0f}"

                lines.append(f" P{i}[{state_str}]: Stack={stack_str:<5} Bet(Rnd)={bet_rnd_str:<4} Bet(Hand)={bet_hand_str:<5}")
            else:
                lines.append(f" P{i}: Invalid State Data")

        lines.append(f" History: {hist_display}")
        return "\n".join(lines)

# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
