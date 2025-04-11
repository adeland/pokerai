# --- START OF FILE organized_poker_bot/game_engine/game_state.py ---
"""
Game state implementation for poker games.
(Refactored V37: Fixed IndentationError in get_utility internal sim)
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
    from organized_poker_bot.game_engine.deck import Deck
    from organized_poker_bot.game_engine.card import Card
    from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
except ImportError as e:
    print(f"ERROR importing engine components in GameState: {e}")
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
        self.active_players = list(range(self.num_players))
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.deck = Deck()
        self.dealer_position = 0
        self.current_player_idx = -1
        self.current_bet = 0.0
        self.last_raiser = None
        self.last_raise = 0.0 # Store the size of the last raise increment
        self.players_acted_this_round = set()
        self.raise_count_this_street = 0
        self.action_sequence = []
        self.verbose_debug = False # Can be set externally if needed

    # --- Helper Methods ---
    def _get_next_active_player(self, start_idx):
        if not self.active_players or self.num_players == 0: return None
        valid_start = start_idx if 0 <= start_idx < self.num_players else -1
        current_idx = (valid_start + 1) % self.num_players
        search_start_idx = current_idx # Track start of search loop

        for _ in range(self.num_players * 2): # Limit loops to avoid infinite
             # Check if player index is valid and player is actually active and has chips
             if current_idx in self.active_players and \
                0 <= current_idx < len(self.player_stacks) and \
                self.player_stacks[current_idx] > 0.01 and \
                not self.player_folded[current_idx]: # Double check not folded
                 return current_idx # Found next active player

             current_idx = (current_idx + 1) % self.num_players
             if current_idx == search_start_idx: # Have we looped completely?
                  # print(f"DEBUG _get_next_active: Looped from {start_idx}, ended at {current_idx}") # DEBUG
                  break # Exit loop if we made a full circle

        # print(f"DEBUG _get_next_active: FAILED from {start_idx}, returning None") # DEBUG
        return None # No active player found

    def _find_player_relative_to_dealer(self, offset):
        if not self.active_players or self.num_players == 0: return None
        dealer = self.dealer_position % self.num_players
        start_idx = (dealer + offset) % self.num_players
        current_idx = start_idx
        search_start_idx = current_idx # Track start of search loop

        for _ in range(self.num_players * 2):
            # Check if player index valid, active, and has chips (NO folded check needed here, just find POTENTIAL seat)
            if current_idx in self.active_players and \
               0 <= current_idx < len(self.player_stacks) and \
               self.player_stacks[current_idx] > 0.01:
                 return current_idx # Found an active player at relative position

            current_idx = (current_idx + 1) % self.num_players
            if current_idx == search_start_idx: # Have we looped?
                 # print(f"DEBUG _find_relative: Looped for offset {offset}, Dlr={dealer}, start={start_idx}") # DEBUG
                 break

        # print(f"DEBUG _find_relative: FAILED for offset {offset}, Dlr={dealer}, returning None") # DEBUG
        return None # No suitable player found

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
        # Active players are those with chips at the start of the hand
        self.active_players = [i for i, s in enumerate(self.player_stacks) if s > 0.01]

        # Proceed with dealing and blinds if enough players
        if len(self.active_players) >= 2:
            self._deal_hole_cards()
            # Check if dealing failed (e.g., deck empty)
            if self.betting_round == self.HAND_OVER: return
            self._post_blinds()
            # Check if posting blinds failed (or only one player left)
            if self.betting_round == self.HAND_OVER: return
            self._start_betting_round()
        else: # Not enough active players
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1

    def _deal_hole_cards(self):
        """ Deals two cards to each active player. """
        if len(self.active_players) < 2: # Cannot deal if fewer than 2 players with stacks
            self.betting_round = self.HAND_OVER
            return

        # Determine starting player for dealing (player after dealer)
        # Use _find_player_relative_to_dealer to find first player with chips
        start_player = self._find_player_relative_to_dealer(1)
        if start_player is None: # Should not happen if len(active_players) >= 2
            print("ERROR: Could not find starting player for dealing hole cards.")
            self.betting_round = self.HAND_OVER
            return

        # Deal cards one at a time
        current_deal_idx = start_player
        for card_num in range(2): # Deal the first card, then the second
            players_dealt_this_pass = 0
            start_loop_idx = current_deal_idx
            attempts = 0
            # Loop until every active player gets their card for this pass
            while players_dealt_this_pass < len(self.active_players) and attempts < self.num_players * 2:
                # Check if current index is an active player
                if 0 <= current_deal_idx < self.num_players and current_deal_idx in self.active_players:
                    # Ensure hole_cards list exists (should be handled by __init__)
                    # Deal if player needs this card number
                    if len(self.hole_cards[current_deal_idx]) == card_num:
                        if not self.deck: # Check if deck is empty
                            print("ERROR: Deck empty during hole card deal.")
                            self.betting_round = self.HAND_OVER
                            return # Cannot deal
                        self.hole_cards[current_deal_idx].append(self.deck.deal())
                        players_dealt_this_pass += 1

                # Move to the next player index
                current_deal_idx = (current_deal_idx + 1) % self.num_players
                attempts += 1
                # Prevent infinite loop if logic fails
                if attempts >= self.num_players * 2:
                    print(f"ERROR: Stuck dealing hole card pass {card_num+1}")
                    self.betting_round = self.HAND_OVER
                    return

            # Check if correct number of players were dealt a card this pass
            if players_dealt_this_pass != len(self.active_players):
                 print(f"ERROR: Incorrect number of players dealt card pass {card_num+1}")
                 self.betting_round = self.HAND_OVER
                 return
            # After dealing one card to everyone, the next card starts at the same index

    def _deduct_bet(self, player_idx, amount_to_deduct):
        """ Internal helper to deduct bet, update pot/state. Returns actual amount deducted. """
        if not (0 <= player_idx < self.num_players and amount_to_deduct >= 0):
            # print(f"WARN _deduct_bet: Invalid input P{player_idx}, Amt={amount_to_deduct}")
            return 0.0 # Invalid input

        # Determine actual amount (capped by stack)
        actual_deduction = min(amount_to_deduct, self.player_stacks[player_idx])
        if actual_deduction < 0.01: # Don't process negligible amounts
            return 0.0

        # Apply deduction and update state
        self.player_stacks[player_idx] -= actual_deduction
        self.player_bets_in_round[player_idx] += actual_deduction
        self.player_total_bets_in_hand[player_idx] += actual_deduction
        self.pot += actual_deduction

        # Check if player went all-in
        # Use a small tolerance for floating point comparisons
        if abs(self.player_stacks[player_idx]) < 0.01:
            self.player_all_in[player_idx] = True

        return actual_deduction

    def _post_blinds(self):
        """ Posts small and big blinds based on dealer position and active players. """
        if len(self.active_players) < 2: # Need 2+ for blinds
            self.betting_round = self.HAND_OVER # Not enough players left
            return

        sb_player, bb_player = None, None
        # Determine SB and BB based on num players
        if len(self.active_players) == 2: # HU: Dealer=SB, other=BB (use relative finder)
            # In HU, _find_player_relative_to_dealer(0) finds dealer if they have chips
            # _find_player_relative_to_dealer(1) finds the other player if they have chips
            sb_player = self._find_player_relative_to_dealer(0)
            bb_player = self._find_player_relative_to_dealer(1)
            # If someone was busted exactly on previous hand, finder might return None
            if sb_player is None or bb_player is None:
                 # This implies only one player remains, handled by the initial check
                 self.betting_round = self.HAND_OVER
                 return
        else: # 3+ players: Normal positions relative to dealer
            sb_player = self._find_player_relative_to_dealer(1)
            bb_player = self._find_player_relative_to_dealer(2)
            # Handle cases where blinds might be missing (e.g., player busted)
            if sb_player is None or bb_player is None:
                 print("ERROR: Cannot find SB or BB position.")
                 self.betting_round = self.HAND_OVER
                 return
            # Ensure SB and BB are distinct if possible
            if sb_player == bb_player:
                 print("ERROR: SB and BB positions are the same.")
                 self.betting_round = self.HAND_OVER
                 return

        self.raise_count_this_street = 0 # Reset preflop raise count

        # Post Small Blind
        sb_posted_amount = 0.0
        if sb_player is not None: # Check if SB position was found
             sb_amount_to_post = min(self.small_blind, self.player_stacks[sb_player])
             sb_posted_amount = self._deduct_bet(sb_player, sb_amount_to_post)
             if sb_posted_amount > 0.01:
                  self.action_sequence.append(f"P{sb_player}:sb{int(round(sb_posted_amount))}")

        # Post Big Blind
        bb_posted_amount = 0.0
        if bb_player is not None: # Check if BB position was found
             bb_amount_to_post = min(self.big_blind, self.player_stacks[bb_player])
             bb_posted_amount = self._deduct_bet(bb_player, bb_amount_to_post)
             if bb_posted_amount > 0.01:
                 # Log total amount IN POT for round after BB post
                 log_bb_amt = self.player_bets_in_round[bb_player]
                 self.action_sequence.append(f"P{bb_player}:bb{int(round(log_bb_amt))}")

        # Set initial betting level and raiser state
        self.current_bet = self.big_blind # Minimum amount to call is BB level
        self.last_raise = self.big_blind # Reference for minimum raise size is initially BB size
        if bb_player is not None and bb_posted_amount >= self.big_blind - 0.01:
             # If BB posted full amount (or went all-in for at least BB)
             self.last_raiser = bb_player
             self.raise_count_this_street = 1 # BB post counts as the first 'raise'
        elif sb_player is not None and sb_posted_amount > 0.01: # Fallback if BB short/missing
            self.last_raiser = sb_player
            self.current_bet = sb_posted_amount # Current bet is only SB amount
            self.last_raise = sb_posted_amount # Next raise must be at least this much more
            self.raise_count_this_street = 1 # SB post counts as first 'raise'
        else: # No effective 'raise' if neither could post significant blind
            self.last_raiser = None
            self.current_bet = 0.0 # No bet to call yet
            self.last_raise = self.big_blind # Next aggression must be at least BB
            self.raise_count_this_street = 0


    # --- Round Progression ---
    def _start_betting_round(self):
        """ Initializes state for the start of a new betting round (post-flop or finds first actor pre-flop). """
        # Reset round-specific state post-flop
        if self.betting_round != self.PREFLOP:
            self.current_bet = 0.0
            self.last_raiser = None
            self.last_raise = self.big_blind # Minimum bet/raise size based on BB post-flop
            self.raise_count_this_street = 0
            # Clear bets *in this round*
            self.player_bets_in_round = [0.0] * self.num_players

        self.players_acted_this_round = set() # Clear who acted this round
        first_player_to_act = None

        if self.betting_round == self.PREFLOP:
             # Find player after BB to act first
             if len(self.active_players) == 2: # HU: Dealer/SB acts first preflop
                 first_player_to_act = self._find_player_relative_to_dealer(0)
             else: # 3+ players: Player after BB acts first (UTG)
                 bb_player = self._find_player_relative_to_dealer(2)
                 # Start search from player after BB
                 first_player_to_act = self._get_next_active_player(bb_player if bb_player is not None else self.dealer_position)
        else: # Postflop rounds: First active Player left of dealer acts first
            first_player_to_act = self._get_next_active_player(self.dealer_position)
        if first_player_to_act is None:
             print(f"!!! WARN _start_betting_round: FAILED to find first_player_to_act (Rnd={self.betting_round}, Dlr={self.dealer_position}, Active={self.active_players}, Stacks={self.player_stacks})")
             self.current_player_idx = -1
        else:
             # print(f"    DEBUG _start: Setting current_player_idx = {first_player_to_act}")
             self.current_player_idx = first_player_to_act
        # Set current player index
        self.current_player_idx = first_player_to_act if first_player_to_act is not None else -1

        # Check if betting can occur or should be skipped immediately
        if self._check_all_active_are_allin():
             # print(f"DEBUG _start_betting_round: Skipping betting, players all-in. Round: {self.betting_round}")
             self.current_player_idx = -1 # Skip betting if <=1 player can act


    def _deal_community_card(self, burn=True):
        """ Deals one card, optionally burning. Returns True if successful. """
        if burn:
            if not self.deck: return False # Cannot burn if empty
            self.deck.deal() # Burn card
        if not self.deck: return False # Cannot deal if empty
        self.community_cards.append(self.deck.deal())
        return True

    def deal_flop(self):
        """ Deals the flop cards. """
        # Need 4 cards: Burn + 3 Flop
        if len(self.community_cards) != 0 or len(self.deck) < 4:
            self.betting_round = self.HAND_OVER; return False
        try:
            self.deck.deal() # Burn card first
            # Deal 3 flop cards without burning between them
            if not all(self._deal_community_card(False) for _ in range(3)):
                raise RuntimeError("Deck ran out during flop deal")
            self.betting_round = self.FLOP # Advance round state
            self._start_betting_round() # Find next player, unless all-in check skips turn
            return True
        except Exception as e:
             print(f"ERROR dealing flop: {e}")
             self.betting_round = self.HAND_OVER; return False

    def deal_turn(self):
        """ Deals the turn card. """
        # Need 2 cards: Burn + Turn
        if len(self.community_cards) != 3 or len(self.deck) < 2:
            self.betting_round = self.HAND_OVER; return False
        try:
            if not self._deal_community_card(True): # Burn and Deal
                raise RuntimeError("Deck ran out during turn deal")
            self.betting_round = self.TURN # Advance round state
            self._start_betting_round()
            return True
        except Exception as e:
             print(f"ERROR dealing turn: {e}")
             self.betting_round = self.HAND_OVER; return False

    def deal_river(self):
        """ Deals the river card. """
         # Need 2 cards: Burn + River
        if len(self.community_cards) != 4 or len(self.deck) < 2:
            self.betting_round = self.HAND_OVER; return False
        try:
            if not self._deal_community_card(True): # Burn and Deal
                 raise RuntimeError("Deck ran out during river deal")
            self.betting_round = self.RIVER # Advance round state
            self._start_betting_round()
            return True
        except Exception as e:
             print(f"ERROR dealing river: {e}")
             self.betting_round = self.HAND_OVER; return False

    def _check_all_active_are_allin(self):
        """ Checks if <=1 player is NOT folded AND NOT all-in. """
        # Active players list now only contains non-folded players with chips at start_new_hand
        # Need to re-evaluate based on current folded/all_in status
        non_folded_players = [p for p in range(self.num_players) if not self.player_folded[p]]

        if len(non_folded_players) <= 1:
             # Hand is over or uncontested if 0 or 1 player is not folded
             return True

        # Count players who can still potentially bet/raise/fold voluntarily
        count_can_still_act_voluntarily = 0
        for p_idx in non_folded_players:
             # Check bounds just in case
             if 0 <= p_idx < self.num_players and \
                p_idx < len(self.player_all_in) and \
                p_idx < len(self.player_stacks):
                  if not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01:
                       count_can_still_act_voluntarily += 1

        # If 0 or 1 player can make a decision, betting should stop
        return count_can_still_act_voluntarily <= 1

    def _move_to_next_player(self):
        """ Finds next active player index who can act, handles no player found. Modifies self.current_player_idx """
        if self.current_player_idx != -1:
            next_p_idx = self._get_next_active_player(self.current_player_idx)
            # Set to -1 if no next active player is found (e.g., everyone else folded/all-in)
            self.current_player_idx = next_p_idx if next_p_idx is not None else -1
        # If current_player_idx was already -1, it stays -1


    # --- Action Handling ---
    def apply_action(self, action):
        """ Validates and applies action to a clone, returns new state. """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError(f"Action must be a tuple of length 2: {action}")
        action_type, amount_input = action

        # Validate amount format early
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

        # Validate player turn and state before cloning
        if acting_player_idx == -1:
            raise ValueError("Invalid action: No player's turn indicated")
        if not (0 <= acting_player_idx < self.num_players):
             raise ValueError(f"Invalid acting_player_idx: {acting_player_idx}")
        # Check list bounds for safety
        if acting_player_idx >= len(self.player_folded) or \
           acting_player_idx >= len(self.player_all_in) or \
           acting_player_idx >= len(self.player_stacks):
            raise ValueError(f"Player index {acting_player_idx} out of bounds for state lists")
        if self.player_folded[acting_player_idx]:
             raise ValueError(f"Invalid action: Player {acting_player_idx} has already folded")
        if self.player_all_in[acting_player_idx]:
             # If player is already all-in, they cannot act. Just advance turn check.
             new_state_skip = self.clone() # Clone state
             new_state_skip._move_to_next_player() # Move turn indicator
             # Check if round/hand ended after skipping the all-in player
             if new_state_skip._is_betting_round_over():
                  new_state_skip._try_advance_round()
             return new_state_skip # Return the advanced state

        # Clone the state BEFORE applying the action logic
        new_state = self.clone()

        try:
            # Apply the logic (MUTATES the clone 'new_state')
            new_state._apply_action_logic(acting_player_idx, action_type, amount)
        except ValueError as e:
            # Add more context to the error if needed
            # print(f"ERROR apply_action P{acting_player_idx} action={action}: {e}")
            raise # Re-raise validation error to stop bad processing

        # After action logic potentially modifies state (like setting hand_over),
        # check if betting round is over on the clone
        if new_state._is_betting_round_over():
            new_state._try_advance_round() # Attempts deal/showdown/end hand
        else:
            # If round not over, just move to the next player
            new_state._move_to_next_player()
            # Check again if moving turn ended the round (e.g., last player acted)
            # Avoid infinite loop if _move_to_next_player returns -1
            if new_state.current_player_idx != -1 and new_state._is_betting_round_over():
                 new_state._try_advance_round()

        return new_state # Return the successfully modified and advanced clone


    def _apply_action_logic(self, p_idx, action_type, amount):
        """ Internal logic, MUTATES self state based on validated action. """
        # Check bounds just in case internal state is inconsistent
        if not (0 <= p_idx < self.num_players and \
                p_idx < len(self.player_stacks) and \
                p_idx < len(self.player_bets_in_round)):
             raise IndexError(f"Invalid player index {p_idx} in _apply_action_logic")

        player_stack = self.player_stacks[p_idx]
        current_round_bet = self.player_bets_in_round[p_idx]
        self.players_acted_this_round.add(p_idx) # Mark as acted
        action_log_repr = f"P{p_idx}:" # Start building log string

        # --- Fold ---
        if action_type == "fold":
            self.player_folded[p_idx] = True
            # Remove from active_players list IF PRESENT (might have been removed earlier)
            if p_idx in self.active_players:
                self.active_players.remove(p_idx)
            action_log_repr += "f"
            # Check if hand ends now (only one player left not folded)
            if len([p for p in range(self.num_players) if not self.player_folded[p]]) <= 1:
                self.betting_round = self.HAND_OVER
                self.current_player_idx = -1 # Hand is over

        # --- Check ---
        elif action_type == "check":
            # Validate check: Current bet must be <= player's bet in round (allow for tolerance)
            if self.current_bet - current_round_bet > 0.01:
                raise ValueError(f"Invalid check P{p_idx}: Bet={self.current_bet}, HasBet={current_round_bet}")
            action_log_repr += "k" # Log as check

        # --- Call ---
        elif action_type == "call":
            amount_needed = self.current_bet - current_round_bet
            # If no amount needed, treat as check conceptually (but still log as call)
            if amount_needed <= 0.01:
                action_log_repr += "c0" # Log as call of 0
            else: # Actual call needed
                call_cost = min(amount_needed, player_stack)
                if call_cost < 0: call_cost = 0 # Safety
                self._deduct_bet(p_idx, call_cost)
                # Log total amount IN POT for round after call
                action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}"

        # --- Bet (Opening Bet) ---
        elif action_type == "bet":
            # Validate: Only allowed if current_bet is 0 (or negligible)
            if self.current_bet > 0.01:
                raise ValueError("Invalid bet: Use raise instead as there is a facing bet.")
            if amount < 0.01:
                 raise ValueError("Bet amount must be positive.")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                 raise ValueError("Max raises/bets reached for this street.")

            # Determine minimum legal bet size (usually BB, but can be less if stack is small)
            min_bet_amount = max(self.big_blind, 1.0) # Min bet is generally BB
            actual_bet_cost = min(amount, player_stack) # Cost is capped by stack
            is_all_in = abs(actual_bet_cost - player_stack) < 0.01

            # Check min bet size (unless all-in for less)
            if actual_bet_cost < min_bet_amount - 0.01 and not is_all_in:
                raise ValueError(f"Bet {actual_bet_cost:.2f} is less than minimum {min_bet_amount:.2f}")

            # Apply bet
            self._deduct_bet(p_idx, actual_bet_cost)
            action_log_repr += f"b{int(round(actual_bet_cost))}" # Log the bet cost

            # Update betting state: New level is the bet size, player is last raiser
            new_total_bet_level = self.player_bets_in_round[p_idx]
            self.current_bet = new_total_bet_level
            # First aggressive action sets baseline for next raise size
            self.last_raise = new_total_bet_level
            self.last_raiser = p_idx
            self.raise_count_this_street = 1 # This is the first bet/raise
            # Action re-opened, only this player has acted against the new level
            self.players_acted_this_round = {p_idx}
            if is_all_in:
                self.player_all_in[p_idx] = True # Mark all-in state

        # --- Raise ---
        elif action_type == "raise":
            # Validate: Only allowed if there's a current bet to raise over
            if self.current_bet <= 0.01:
                raise ValueError("Invalid raise: Use bet instead as there is no facing bet.")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                 raise ValueError("Max raises reached for this street.")

            total_bet_target = amount # Amount passed is the TOTAL target bet level for the round
            # Cost for the player to reach this target level
            cost_to_reach_target = total_bet_target - current_round_bet
            if cost_to_reach_target <= 0.01:
                 raise ValueError(f"Raise target {total_bet_target} not greater than current bet in round {current_round_bet}")
            if cost_to_reach_target > player_stack + 0.01:
                 raise ValueError(f"Player {p_idx} cannot afford raise cost {cost_to_reach_target:.2f} with stack {player_stack:.2f}")

            # Actual cost paid by player (capped by stack)
            actual_raise_cost = min(cost_to_reach_target, player_stack)
            # The total bet level player reaches after paying cost
            actual_total_bet_reached = current_round_bet + actual_raise_cost
            is_all_in = abs(actual_raise_cost - player_stack) < 0.01

            # Check legality of raise size: Increment must be >= last raise increment OR player is all-in
            # Minimum legal raise increment size (at least BB or the previous raise amount)
            min_legal_increment = max(self.last_raise, self.big_blind)
            # The actual increase over the *current facing bet level*
            actual_increment_made = actual_total_bet_reached - self.current_bet

            if actual_increment_made < min_legal_increment - 0.01 and not is_all_in:
                raise ValueError(f"Raise increment {actual_increment_made:.2f} is less than minimum legal increment {min_legal_increment:.2f}")

            # Apply the raise cost
            self._deduct_bet(p_idx, actual_raise_cost)
            # Log the TOTAL amount reached after the raise
            action_log_repr += f"r{int(round(actual_total_bet_reached))}"

            # Update state: new current bet level, last raiser, increment size
            new_bet_level = actual_total_bet_reached
             # Update size of *this* raise increment (used for next min raise check)
            self.last_raise = new_bet_level - self.current_bet
            self.current_bet = new_bet_level # Update level needed to call
            self.last_raiser = p_idx
            self.raise_count_this_street += 1
            # Action re-opened, only this player has acted against the new level
            self.players_acted_this_round = {p_idx}
            if is_all_in:
                 self.player_all_in[p_idx] = True # Mark all-in

        # --- Unknown Action ---
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # Add action to history if it wasn't just a marker (like Call 0)
        # Ensure action_log_repr has more than just "P#: "
        if len(action_log_repr) > len(f"P{p_idx}:"):
            self.action_sequence.append(action_log_repr)


    def get_betting_history(self):
         """ Returns the sequence of actions as a single semicolon-separated string. """
         return ";".join(self.action_sequence)

    def get_available_actions(self):
        """ Calculates and returns a list of legal actions for the current player. """
        actions = []
        player_idx = self.current_player_idx

        # Check if it's anyone's turn
        if player_idx == -1:
            return []

        # Basic validity checks for player index and state
        try:
             # Check bounds first
            if not (0 <= player_idx < self.num_players and \
                    player_idx < len(self.player_folded) and \
                    player_idx < len(self.player_all_in) and \
                    player_idx < len(self.player_stacks) and \
                    player_idx < len(self.player_bets_in_round)):
                 return [] # Invalid index or state lists not initialized correctly

             # Check if player can act
            if self.player_folded[player_idx] or \
               self.player_all_in[player_idx] or \
               self.player_stacks[player_idx] < 0.01:
                return []
        except IndexError:
            # This should not happen if bounds check passes, but for safety
            print(f"WARN get_available_actions: IndexError accessing state for P{player_idx}")
            return []

        # Get relevant state variables
        player_stack = self.player_stacks[player_idx]
        current_round_bet = self.player_bets_in_round[player_idx]
        current_bet_level = self.current_bet

        # --- Fold ---
        # Always possible if player can act
        actions.append(("fold", 0))

        # --- Check or Call ---
        amount_to_call = current_bet_level - current_round_bet # Amount needed to match
        can_check = amount_to_call < 0.01 # Check possible if no amount needed (or negligible)

        if can_check:
            actions.append(("check", 0))
        else: # Call is needed
            # Cost to call is capped by the player's stack
            call_cost = min(amount_to_call, player_stack)
            # Only add call if there's a significant cost (don't add "call 0")
            if call_cost > 0.01:
                # Store action tuple as (type, COST)
                actions.append(("call", int(round(call_cost))))

        # --- Bet or Raise (Aggression) ---
        # Check if max raises/bets limit reached
        can_aggress = self.raise_count_this_street < self.MAX_RAISES_PER_STREET
        # Can only aggress if stack is greater than cost to call (must be able to increase total bet)
        # Use max(0, amount_to_call) in case current_bet < current_round_bet somehow
        effective_call_cost = max(0.0, amount_to_call)
        if can_aggress and player_stack > effective_call_cost + 0.01:
            # Define min/max aggression amounts (as TOTAL bet target for the round)
            # Max possible target is going all-in
            max_legal_aggress_target_to = current_round_bet + player_stack
            min_legal_aggress_target_to = float('inf') # Smallest TOTAL bet allowed

            if current_bet_level < 0.01: # Current action is BET (no prior aggression)
                action_prefix = "bet"
                # Min bet COST is usually BB, capped by stack
                min_bet_cost = min(player_stack, max(self.big_blind, 1.0))
                # Min TARGET is current bet (0) + min cost
                min_legal_aggress_target_to = current_round_bet + min_bet_cost
            else: # Current action is RAISE
                action_prefix = "raise"
                # Minimum legal raise increment size
                min_legal_increment = max(self.last_raise, self.big_blind)
                # Minimum total amount to raise TO
                min_raise_target_to = current_bet_level + min_legal_increment
                # The actual minimum target might be capped by going all-in
                # Ensure min target is calculated correctly even if all-in is less than required increment
                min_legal_aggress_target_to = min(max_legal_aggress_target_to, min_raise_target_to)


            # Ensure the calculated minimum target is actually greater than the current bet level
            # (It might not be if going all-in is less than the required increment)
            is_min_target_aggressive = (min_legal_aggress_target_to > current_bet_level + 0.01)

            # Add Min Legal Aggressive Action (if possible & actually aggressive)
            if is_min_target_aggressive:
                # Store action tuple as (type, TARGET_TOTAL_AMOUNT)
                actions.append((action_prefix, int(round(min_legal_aggress_target_to))))

            # Add All-In Aggressive Action (if going all-in is possible, aggressive, and distinct from min legal)
            is_all_in_target_aggressive = (max_legal_aggress_target_to > current_bet_level + 0.01)
            is_all_in_distinct = abs(max_legal_aggress_target_to - min_legal_aggress_target_to) > 0.01

            if is_all_in_target_aggressive and (not is_min_target_aggressive or is_all_in_distinct) :
                # Add all-in if it's aggressive and either min wasn't added or all-in is different
                 actions.append((action_prefix, int(round(max_legal_aggress_target_to))))


        # --- Final Filtering and Sorting ---
        def sort_key(action_tuple):
            """ Defines sorting order: fold < check < call < bet < raise, then by amount. """
            action_type, amount = action_tuple
            order = {"fold":0, "check":1, "call":2, "bet":3, "raise":4}
            # Use amount directly for sorting, ensure type is numeric for comparison if needed
            sort_amount = amount if isinstance(amount, (int, float)) else 0
            return (order.get(action_type, 99), sort_amount)

        final_actions = []
        seen_actions_repr = set() # Use a representation string/tuple for uniqueness check

        # Sort the collected actions
        sorted_actions = sorted(actions, key=sort_key)

        for act_tuple in sorted_actions:
             act_type, act_amount = act_tuple
             # Create a canonical representation for checking uniqueness
             # Use rounded int amount for consistency
             action_key_repr = (act_type, int(round(act_amount)))

             # Calculate the actual COST of this action for the player
             cost = 0.0
             if act_type == 'call':
                  # Call amount IS the cost (already capped by stack if generated correctly)
                  cost = act_amount
             elif act_type == 'bet':
                  # Bet amount IS the cost (from 0), should be capped by stack
                  cost = act_amount
             elif act_type == 'raise':
                  # Raise cost is TARGET_TOTAL_AMOUNT - AlreadyInRound
                  cost = act_amount - current_round_bet
             # Ensure cost is not negative (e.g., if rounding causes issues)
             cost = max(0.0, cost)

             # Add if unique representation and player can afford the cost (with small tolerance)
             if action_key_repr not in seen_actions_repr and cost <= player_stack + 0.01:
                 final_actions.append(act_tuple) # Add the ORIGINAL tuple
                 seen_actions_repr.add(action_key_repr)

        return final_actions # Return the filtered, sorted list


    def _is_betting_round_over(self):
        """ Checks if the current betting round has concluded based on player actions and bets. """
        # Find players who are not folded
        eligible_players = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(eligible_players) <= 1:
            return True # Round (and hand) ends if <= 1 player remains

        # Find players who can still make a voluntary action (not folded, not all-in, has chips)
        players_who_can_voluntarily_act = []
        for p_idx in eligible_players:
             # Check bounds for safety
            if 0 <= p_idx < self.num_players and \
               p_idx < len(self.player_all_in) and \
               p_idx < len(self.player_stacks):
                 if not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01:
                      players_who_can_voluntarily_act.append(p_idx)

        num_can_act = len(players_who_can_voluntarily_act)

        # Case 1: Zero players can voluntarily act (everyone remaining is all-in)
        if num_can_act == 0:
            return True

        # Case 2: Only one player can voluntarily act
        if num_can_act == 1:
            the_player = players_who_can_voluntarily_act[0]
            has_acted = the_player in self.players_acted_this_round
            # Check if they face a bet requiring action
            facing_bet = (self.current_bet - self.player_bets_in_round[the_player]) > 0.01

            # Special Preflop BB Option Check:
            is_preflop = (self.betting_round == self.PREFLOP)
            # Determine BB player index robustly
            bb_player_idx = None
            if len(self.active_players) >= 2: # Need active_players list from start_new_hand
                 if len(self.active_players) == 2: bb_player_idx = self._find_player_relative_to_dealer(1)
                 else: bb_player_idx = self._find_player_relative_to_dealer(2)

            is_bb_player = (the_player == bb_player_idx)
            # Check if only blinds have been posted (no re-raise occurred)
            no_reraise_yet = (self.raise_count_this_street <= 1 and self.last_raiser == bb_player_idx)

            # If it's preflop, the player is BB, no re-raise occurred, they DON'T face a bet, AND they haven't acted yet...
            if is_preflop and is_bb_player and no_reraise_yet and not facing_bet and not has_acted:
                 return False # BB still has the option to act

            # Otherwise (for Case 2), the round ends if the player doesn't face a bet, OR if they do face a bet but have already acted this round.
            return not facing_bet or has_acted

        # Case 3: Multiple players can act
        # Round ends if ALL players who can act have matched the current bet AND have acted at least once this round.
        all_matched = True
        all_acted = True
        for p_idx in players_who_can_voluntarily_act:
            # Check if bet matches current level (allow tolerance)
            if abs(self.player_bets_in_round[p_idx] - self.current_bet) > 0.01:
                all_matched = False
             # Check if player has acted since the last aggressive action (or start of round)
            if p_idx not in self.players_acted_this_round:
                all_acted = False
            # If either condition fails for any player, we can stop checking
            if not all_matched or not all_acted:
                break

        return all_matched and all_acted


    def _try_advance_round(self):
        """ Attempts to deal next street or end hand if betting round finished. MUTATES state. """
        # Check if hand ended due to folds
        eligible_players = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(eligible_players) <= 1:
            if self.betting_round != self.HAND_OVER:
                self.betting_round = self.HAND_OVER
            self.current_player_idx = -1
            self.players_acted_this_round = set() # Clear acted set for safety
            return # Hand is over

        # Check if betting is skipped because players are all-in
        should_skip_betting = self._check_all_active_are_allin()

        if should_skip_betting and self.betting_round < self.SHOWDOWN:
            # Deal remaining streets without betting if players are all-in
            # Deals sequentially until river or error
            # Need to ensure we don't advance round state incorrectly within deal methods
            temp_round = self.betting_round # Store current round before dealing

            if temp_round < self.FLOP:
                 if not self.deal_flop(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return
                 # deal_flop sets round to FLOP, but we might need turn/river too
            if len(self.community_cards) < 4: # Check card count directly
                 if not self.deal_turn(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return
            if len(self.community_cards) < 5:
                 if not self.deal_river(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return

            # If we successfully dealt up to the river (or were already there)
            if self.betting_round != self.HAND_OVER:
                 self.betting_round = self.SHOWDOWN
                 self.current_player_idx = -1 # No more actions
                 self.players_acted_this_round = set()
            return # Reached showdown (or hand over due to error)

        # --- Normal round advancement after betting completes ---
        current_round = self.betting_round
        round_advanced_successfully = False

        if current_round == self.PREFLOP:
            round_advanced_successfully = self.deal_flop()
        elif current_round == self.FLOP:
            round_advanced_successfully = self.deal_turn()
        elif current_round == self.TURN:
            round_advanced_successfully = self.deal_river()
        elif current_round == self.RIVER:
             # After river betting, move to showdown
            self.betting_round = self.SHOWDOWN
            self.current_player_idx = -1 # No more actions
            self.players_acted_this_round = set() # Clear acted set
            round_advanced_successfully = True
        # If already in SHOWDOWN or HAND_OVER, do nothing

        # If dealing failed, set state to HAND_OVER
        if not round_advanced_successfully and self.betting_round < self.SHOWDOWN:
             if self.betting_round != self.HAND_OVER: # Avoid redundant set
                  self.betting_round = self.HAND_OVER
             self.current_player_idx = -1
             self.players_acted_this_round = set()


    def is_terminal(self):
        """ Checks if the game hand has reached a terminal state. """
        # Hand ends if only one player remains unfolded
        eligible_player_count = len([p for p in range(self.num_players) if not self.player_folded[p]])
        if eligible_player_count <= 1:
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

        # Get current stack safely (make a copy to avoid modifying original state)
        current_game_state_copy = self.clone()
        current_stack = 0.0
        try:
            c_s = current_game_state_copy.player_stacks[player_idx]
            if not isinstance(c_s, (int,float)) or np.isnan(c_s) or np.isinf(c_s):
                 raise ValueError("Invalid current stack value")
            current_stack = float(c_s)
        except (IndexError, TypeError, ValueError) as e:
             print(f"WARN get_utility: Invalid current stack for P{player_idx}: {e}")
             return 0.0


        # Determine winners and distribute pot internally (on the copy)
        # Use the *original* determine_winners logic for this internal calculation
        # This assumes determine_winners updates the stacks on the object it's called on.
        try:
             # Call determine_winners on the cloned state
             _ = current_game_state_copy.determine_winners()
             # Get the final stack from the *modified clone*
             final_effective_stack = current_game_state_copy.player_stacks[player_idx]
             # Validate the final stack
             if not isinstance(final_effective_stack, (int,float)) or np.isnan(final_effective_stack) or np.isinf(final_effective_stack):
                  raise ValueError("Invalid final stack value after internal win determination")

        except Exception as win_err:
             print(f"ERROR get_utility: Internal win determination failed for P{player_idx}: {win_err}")
             traceback.print_exc() # Print traceback for debugging
             # Fallback: return utility based on current stack before win determination attempt
             final_effective_stack = current_stack # Use pre-distribution stack

        # Utility is the change in stack size
        utility = final_effective_stack - initial_stack

        # Final safety check for NaN/Inf
        if np.isnan(utility) or np.isinf(utility):
            # print(f"WARN get_utility: Calculated utility is NaN/Inf for P{player_idx}. Returning 0.")
            utility = 0.0

        return utility


    def determine_winners(self, player_names=None):
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
            return pots_summary

        # Case 2: Showdown required
        evaluated_hands = {}
        valid_showdown_players = [] # Players who are eligible AND have valid hands for showdown
        for p_idx in eligible_for_pot:
            # Check list bounds and hand validity
            if p_idx >= len(self.hole_cards) or len(self.hole_cards[p_idx]) != 2:
                continue # Skip players with invalid hole cards
            all_cards_for_eval = self.hole_cards[p_idx] + self.community_cards
            # Need at least 5 cards total (2 hole + 3 community minimum) for evaluation
            if len(all_cards_for_eval) < 5:
                 continue # Skip if not enough cards dealt yet (shouldn't happen if terminal state is correct)
            try:
                # Evaluate hand using the HandEvaluator
                evaluated_hands[p_idx] = HandEvaluator.evaluate_hand(all_cards_for_eval)
                valid_showdown_players.append(p_idx)
            except Exception as eval_err:
                 print(f"WARN determine_winners: Hand evaluation failed for P{p_idx}: {eval_err}")
                 continue # Skip players whose hands cannot be evaluated

        # If no players have valid hands for showdown (e.g., error, insufficient cards), return empty
        if not valid_showdown_players:
            # print("WARN determine_winners: No valid hands found for showdown.")
            # Pot remains undistributed in this error case? Or return to players? Safest is maybe let it vanish.
            return []

        # Calculate side pots based on contributions
        # Get total contribution for each player eligible for showdown
        contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players], key=lambda x: x[1])

        side_pots = [] # List to store {'amount': float, 'eligible': list_of_player_indices}
        last_contribution_level = 0.0
        # Make a copy of players eligible for the *next* pot to be created
        eligible_for_next_pot = valid_showdown_players[:]

        for p_idx_sp, total_contribution in contributions:
             contribution_increment = total_contribution - last_contribution_level
             # If this player contributed more than the last level...
             if contribution_increment > 0.01:
                 num_eligible = len(eligible_for_next_pot)
                 # The amount for this side pot is the increment times the number of players eligible for it
                 pot_amount = contribution_increment * num_eligible
                 if pot_amount > 0.01:
                      # Add this side pot info (amount, and WHO is eligible for THIS pot)
                     side_pots.append({'amount': pot_amount, 'eligible': eligible_for_next_pot[:]}) # Use copy
                 last_contribution_level = total_contribution # Update the contribution level

             # This player is no longer eligible for subsequent, smaller side pots they didn't contribute fully to
             if p_idx_sp in eligible_for_next_pot:
                 eligible_for_next_pot.remove(p_idx_sp)

        # Add main pot if no side pots were needed (e.g. all contributed same)
        # Check if the calculated side pots cover the total pot
        calculated_pot_sum = sum(sp['amount'] for sp in side_pots)
        # If side pots were created but don't sum up, there might be an issue.
        # If NO side pots were created, the entire pot is the main pot.
        if not side_pots and total_pot_to_distribute > 0.01:
             side_pots.append({'amount': total_pot_to_distribute, 'eligible': valid_showdown_players[:]})
        # Optional check for discrepancy:
        # elif abs(calculated_pot_sum - total_pot_to_distribute) > 0.1: # Allow small tolerance
        #    print(f"WARN determine_winners: Discrepancy between total pot {total_pot_to_distribute} and sum of side pots {calculated_pot_sum}")


        # Award the pots
        distributed_total = 0
        pots_summary = [] # Re-initialize summary list
        for i, pot_info in enumerate(side_pots):
            pot_amount = pot_info.get('amount', 0.0)
            eligible_players_this_pot = pot_info.get('eligible', [])

            if pot_amount < 0.01 or not eligible_players_this_pot:
                 continue # Skip empty pots or pots with no eligible players

            # Find the best hand among players eligible for THIS pot
            eligible_hands = {p: evaluated_hands[p] for p in eligible_players_this_pot if p in evaluated_hands}
            if not eligible_hands:
                 continue # Skip if no valid hands among eligible players

            best_hand_value = max(eligible_hands.values())
            # Find all players eligible for this pot who have the best hand value
            pot_winners = [p for p, hand_val in eligible_hands.items() if hand_val == best_hand_value]

            if pot_winners:
                 winner_share = pot_amount / len(pot_winners)
                 distributed_total += pot_amount # Track total distributed
                 for w_idx in pot_winners:
                     # Check bounds before adding winnings
                     if 0 <= w_idx < len(self.player_stacks):
                          self.player_stacks[w_idx] += winner_share

                 # Create summary entry for this pot
                 pot_desc = f"Side Pot {i+1}" if len(side_pots) > 1 else "Main Pot"
                 pots_summary.append({'winners':pot_winners, 'amount':pot_amount, 'eligible':eligible_players_this_pot, 'desc': pot_desc})

        # Optional check if distributed amount matches total pot
        # if abs(distributed_total - total_pot_to_distribute) > 0.1:
        #     print(f"WARN determine_winners: Distributed {distributed_total} != Total Pot {total_pot_to_distribute}")

        return pots_summary


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
        board = ' '.join(map(str, self.community_cards)) if self.community_cards else "-"
        # Limit history length for display
        hist_limit = 60
        hist = self.get_betting_history()
        hist_display = f"...{hist[-hist_limit:]}" if len(hist) > hist_limit else hist

        lines = []
        lines.append(f"Round: {round_name}, Turn: {turn}, Pot: {self.pot:.2f}, Board: [{board}]")

        for i in range(self.num_players):
            # Check bounds before accessing player state
            if i < len(self.player_stacks) and i < len(self.player_folded) and i < len(self.player_all_in) and i < len(self.player_bets_in_round):
                state_flags = []
                if i == self.dealer_position: state_flags.append("D")
                if self.player_folded[i]: state_flags.append("F")
                if self.player_all_in[i]: state_flags.append("A")
                state_str = "".join(state_flags) if state_flags else " "
                # Format numbers nicely
                stack_str = f"{self.player_stacks[i]:.0f}"
                bet_str = f"{self.player_bets_in_round[i]:.0f}"
                lines.append(f" P{i}[{state_str}]: Stack={stack_str}, Bet(Round)={bet_str}")
            else:
                lines.append(f" P{i}: Invalid State Data")

        lines.append(f" History: {hist_display}")
        return "\n".join(lines)

# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
