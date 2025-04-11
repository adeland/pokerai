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
        self.active_players = list(range(self.num_players)) # Initial assumption
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
        if not self.active_players or self.num_players == 0:
            return None
        # Ensure start_idx is valid before using it
        valid_start = start_idx if 0 <= start_idx < self.num_players else -1
        current_idx = (valid_start + 1) % self.num_players
        search_start_idx = current_idx # Where the search begins in the loop

        for _ in range(self.num_players * 2): # Limit loops
             if current_idx in self.active_players and \
                0 <= current_idx < len(self.player_stacks) and \
                self.player_stacks[current_idx] > 0.01 and \
                not self.player_folded[current_idx]: # Check folded status too
                 return current_idx
             current_idx = (current_idx + 1) % self.num_players
             # Check if we wrapped around without finding anyone eligible
             if current_idx == search_start_idx:
                  break
        return None # No active player found

    def _find_player_relative_to_dealer(self, offset):
        if not self.active_players or self.num_players == 0:
            return None
        dealer = self.dealer_position % self.num_players
        start_idx = (dealer + offset) % self.num_players
        current_idx = start_idx

        for _ in range(self.num_players * 2): # Limit loops
            if current_idx in self.active_players and \
               0 <= current_idx < len(self.player_stacks) and \
               self.player_stacks[current_idx] > 0.01:
                return current_idx
            current_idx = (current_idx + 1) % self.num_players
            if current_idx == start_idx: # Wrapped around
                 break
        return None # No suitable player found

    # --- Hand Setup Methods ---
    def start_new_hand(self, dealer_pos, player_stacks):
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

        # Update player stacks and determine active players
        self.player_stacks = [float(s) for s in player_stacks]
        self.active_players = [i for i, s in enumerate(self.player_stacks) if s > 0.01]

        # Proceed with dealing and blinds if enough players
        if len(self.active_players) >= 2:
            self._deal_hole_cards()
            self._post_blinds()
            self._start_betting_round()
        else: # Not enough active players
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1

    def _deal_hole_cards(self):
        if not self.active_players:
            return
        start_player = self._find_player_relative_to_dealer(1) or \
                       (self.active_players[0] if self.active_players else None)
        if start_player is None: # No player found (e.g., 0 active)
            self.betting_round = self.HAND_OVER
            return

        current_deal_idx = start_player
        for round_num in range(2): # Deal two cards to each active player
            players_dealt_this_round = 0
            start_loop_idx = current_deal_idx # To detect infinite loop
            attempts = 0 # Safeguard against loops
            while players_dealt_this_round < len(self.active_players) and attempts < self.num_players * 2:
                if 0 <= current_deal_idx < self.num_players and current_deal_idx in self.active_players:
                    # Ensure hole_cards list is large enough dynamically (less likely needed with init)
                    while len(self.hole_cards) <= current_deal_idx:
                        self.hole_cards.append([])
                    # Deal if player needs a card for this round
                    if len(self.hole_cards[current_deal_idx]) == round_num:
                        if not self.deck: # Check if deck is empty
                            self.betting_round = self.HAND_OVER
                            return # Cannot deal
                        self.hole_cards[current_deal_idx].append(self.deck.deal())
                        players_dealt_this_round += 1

                current_deal_idx = (current_deal_idx + 1) % self.num_players
                attempts += 1
                if current_deal_idx == start_loop_idx and attempts > self.num_players:
                    # Looped without dealing all needed cards - error condition
                    print(f"ERROR: Stuck dealing hole card round {round_num+1}")
                    self.betting_round = self.HAND_OVER
                    return
            if players_dealt_this_round < len(self.active_players):
                 self.betting_round = self.HAND_OVER; return # Error dealing enough cards
            # Start next round from next player
            start_player = current_deal_idx

    def _deduct_bet(self, player_idx, amount_to_deduct):
        """ Internal helper to deduct bet, update pot/state. """
        if not (0 <= player_idx < self.num_players and amount_to_deduct >= 0):
            return # Invalid input

        # Determine actual amount (capped by stack)
        actual_deduction = min(amount_to_deduct, self.player_stacks[player_idx])
        if actual_deduction < 0.01: # Don't process negligible amounts
            return

        # Apply deduction and update state
        self.player_stacks[player_idx] -= actual_deduction
        self.player_bets_in_round[player_idx] += actual_deduction
        self.player_total_bets_in_hand[player_idx] += actual_deduction
        self.pot += actual_deduction

        # Check if player went all-in
        if abs(self.player_stacks[player_idx]) < 0.01:
            self.player_all_in[player_idx] = True

    def _post_blinds(self):
        if len(self.active_players) < 2: return # Need 2+ for blinds

        sb_player, bb_player = None, None
        # Determine SB and BB based on num players
        if self.num_players == 2: # HU: Dealer=SB
            sb_player = self._find_player_relative_to_dealer(0)
            bb_player = self._find_player_relative_to_dealer(1)
        else: # 3+ players: Normal positions
            sb_player = self._find_player_relative_to_dealer(1)
            bb_player = self._find_player_relative_to_dealer(2)

        self.raise_count_this_street = 0 # Reset preflop raise count

        # Post Small Blind
        sb_posted_amount = 0.0
        if sb_player is not None and 0 <= sb_player < len(self.player_stacks):
            amount_sb = min(self.small_blind, self.player_stacks[sb_player])
            if amount_sb > 0.01:
                self._deduct_bet(sb_player, amount_sb)
                sb_posted_amount = self.player_bets_in_round[sb_player]
                self.action_sequence.append(f"P{sb_player}:sb{int(round(sb_posted_amount))}")

        # Post Big Blind
        bb_posted_amount = 0.0
        if bb_player is not None and 0 <= bb_player < len(self.player_stacks):
            # Amount needed might be less than full BB if already bet (e.g. posted dead blind?)
            needed_bb = self.big_blind - self.player_bets_in_round[bb_player]
            amount_bb = min(max(0, needed_bb), self.player_stacks[bb_player])
            if amount_bb > 0.01:
                self._deduct_bet(bb_player, amount_bb)
                # Use total round bet for logging consistency
                log_bb_amt = self.player_bets_in_round[bb_player]
                self.action_sequence.append(f"P{bb_player}:bb{int(round(log_bb_amt))}")
                bb_posted_amount = log_bb_amt # Store amount posted

        # Set initial betting level and raiser state
        self.current_bet = self.big_blind # Minimum amount to call is BB
        self.last_raise = self.big_blind # Reference for minimum raise size
        if bb_player is not None and bb_posted_amount >= self.big_blind - 0.01:
            # If BB posted full amount (or went all-in for at least BB)
            self.last_raiser = bb_player
            self.raise_count_this_street = 1 # BB post counts as the first raise
        elif sb_player is not None and sb_posted_amount > 0.01: # Fallback if BB short/missing
            self.last_raiser = sb_player
            self.raise_count_this_street = 1
        else: # No effective 'raise' if neither could post significant blind
            self.last_raiser = None
            self.raise_count_this_street = 0


    # --- Round Progression ---
    def _start_betting_round(self):
        # Reset round-specific state post-flop
        if self.betting_round != self.PREFLOP:
            self.current_bet = 0.0
            self.last_raiser = None
            self.last_raise = self.big_blind # Minimum bet/raise size based on BB
            self.raise_count_this_street = 0
            # Clear bets *in this round*
            for i in range(self.num_players):
                self.player_bets_in_round[i] = 0.0

        self.players_acted_this_round = set() # Clear who acted this round
        first_player_to_act = None

        if self.betting_round == self.PREFLOP:
             # HU: Dealer/SB acts first preflop
            if self.num_players == 2: first_player_to_act = self._find_player_relative_to_dealer(0)
            # 3+ players: Player after BB acts first (UTG)
            else: bb_player = self._find_player_relative_to_dealer(2); first_player_to_act = self._get_next_active_player(bb_player if bb_player is not None else self.dealer_position)
        else: # Postflop rounds: Player left of dealer acts first
            first_player_to_act = self._get_next_active_player(self.dealer_position)

        # Set current player index
        self.current_player_idx = first_player_to_act if first_player_to_act is not None else -1

        # Check if betting can occur or should be skipped
        if self._check_all_active_are_allin():
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
        if len(self.community_cards) >= 3 or len(self.deck) < 4: # Pre-cond + Burn + 3 Flop cards
             self.betting_round = self.HAND_OVER; return False
        self.deck.deal() # Burn card first
        # Deal 3 flop cards without burning between them
        if not all(self._deal_community_card(False) for _ in range(3)):
             self.betting_round = self.HAND_OVER; return False # End hand if deck ran out
        self.betting_round = self.FLOP # Advance round state
        self._start_betting_round() # Find next player, unless all-in check skips turn
        return True

    def deal_turn(self):
        if len(self.community_cards) >= 4 or len(self.deck) < 2: # Pre-cond + Burn + Turn
            self.betting_round = self.HAND_OVER; return False
        if not self._deal_community_card(True): # Burn and Deal
            self.betting_round = self.HAND_OVER; return False
        self.betting_round = self.TURN # Advance round state
        self._start_betting_round()
        return True

    def deal_river(self):
        if len(self.community_cards) >= 5 or len(self.deck) < 2: # Pre-cond + Burn + River
            self.betting_round = self.HAND_OVER; return False
        if not self._deal_community_card(True): # Burn and Deal
             self.betting_round = self.HAND_OVER; return False
        self.betting_round = self.RIVER # Advance round state
        self._start_betting_round()
        return True

    def _check_all_active_are_allin(self):
        """ Checks if <=1 player is NOT folded AND NOT all-in. """
        non_folded_players = [p for p in self.active_players if not self.player_folded[p]]
        if len(non_folded_players) <= 1: return True # Hand over or uncontested
        # Count players who can still potentially bet/raise/fold
        count_can_still_act_voluntarily = sum(1 for p_idx in non_folded_players
                        if not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01)
        # If 0 or 1 player can make a decision, the rest are effectively all-in (relative to pot size)
        return count_can_still_act_voluntarily <= 1

    def _move_to_next_player(self):
        """ Finds next active player index, handles no player found. """
        if self.current_player_idx != -1:
            next_p_idx = self._get_next_active_player(self.current_player_idx)
            # Set to -1 if no next active player is found (e.g., everyone folded/all-in)
            self.current_player_idx = next_p_idx if next_p_idx is not None else -1

    # --- Action Handling ---
    def apply_action(self, action):
        """ Validates and applies action to a clone, returns new state. """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError(f"Action !tuple(2): {action}")
        action_type, amount_input = action
        try: # Validate amount format
             amount = float(amount_input); assert amount >= 0
        except (ValueError, TypeError, AssertionError):
             raise ValueError(f"Invalid action amount: {amount_input}")

        acting_player_idx = self.current_player_idx
        # Validate player state before cloning
        if acting_player_idx == -1: raise ValueError("Invalid action: No player's turn")
        if not (0 <= acting_player_idx < self.num_players and \
                acting_player_idx < len(self.player_folded) and \
                acting_player_idx < len(self.player_all_in)): raise ValueError("Invalid player index/state")
        if self.player_folded[acting_player_idx]: raise ValueError(f"Invalid action: P{acting_player_idx} folded")

        # If player is already all-in, they cannot act further this hand. Advance state check.
        if self.player_all_in[acting_player_idx]:
             new_state_skip = self.clone() # Clone state
             new_state_skip._move_to_next_player() # Move turn indicator
             if new_state_skip._is_betting_round_over(): # Check if round/hand ended
                  new_state_skip._try_advance_round()
             return new_state_skip # Return the advanced state

        # Clone the state BEFORE applying the action logic
        new_state = self.clone()
        try:
             # Apply the logic (MUTATES the clone 'new_state')
             new_state._apply_action_logic(acting_player_idx, action_type, amount)
        except ValueError as e:
             print(f"ERROR ApplyAction P{acting_player_idx} {action}: {e}")
             raise # Re-raise validation error to stop bad processing

        # After action, check if round/hand ended on the clone
        if new_state._is_betting_round_over():
             new_state._try_advance_round() # Attempts deal/showdown/end
        else:
             # If round not over, just move to the next player
             new_state._move_to_next_player()
             # One final check: did moving the player actually end the round?
             # (e.g. BB checks option preflop, or only one player left after moving turn)
             if new_state.current_player_idx != -1 and new_state._is_betting_round_over():
                  new_state._try_advance_round()

        return new_state # Return the successfully modified and advanced clone


    def _apply_action_logic(self, p_idx, action_type, amount):
        """ Internal logic, MUTATES self state based on validated action. """
        player_stack = self.player_stacks[p_idx]
        current_round_bet = self.player_bets_in_round[p_idx]
        self.players_acted_this_round.add(p_idx) # Mark as acted
        action_log_repr = f"P{p_idx}:"

        if action_type == "fold":
            self.player_folded[p_idx] = True
            if p_idx in self.active_players:
                self.active_players.remove(p_idx)
            action_log_repr += "f"
            # Check if hand ends now
            if len(self.active_players) <= 1:
                 self.betting_round = self.HAND_OVER
                 self.current_player_idx = -1 # Hand is over

        elif action_type == "check":
            # Validate check based on current bet level
            if self.current_bet - current_round_bet > 0.01:
                 raise ValueError(f"Invalid check P{p_idx}: Bet={self.current_bet}, HasBet={current_round_bet}")
            action_log_repr += "k" # Log as check

        elif action_type == "call":
            amount_needed = self.current_bet - current_round_bet
            if amount_needed <= 0.01: # Calling 0 is like a check if check was possible
                action_log_repr += "k(c0)"
            else: # Actual call needed
                call_cost = min(amount_needed, player_stack)
                if call_cost < 0: call_cost = 0 # Safety
                self._deduct_bet(p_idx, call_cost)
                # Log total amount IN POT for round after call
                action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}"

        elif action_type == "bet": # Opening bet (current_bet is 0)
            if self.current_bet > 0.01: raise ValueError("Invalid bet: Use raise instead")
            if amount < 0.01: raise ValueError("Bet amount must be positive")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET: raise ValueError("Max raises reached")
            min_bet_amount = max(self.big_blind, 1.0) # Min bet is usually BB
            actual_bet_cost = min(amount, player_stack) # Amount is the cost here
            is_all_in = abs(actual_bet_cost - player_stack) < 0.01
            # Check min bet size (unless all-in for less)
            if actual_bet_cost < min_bet_amount - 0.01 and not is_all_in:
                raise ValueError(f"Bet {actual_bet_cost:.2f} < min {min_bet_amount:.2f}")
            self._deduct_bet(p_idx, actual_bet_cost)
            action_log_repr += f"b{int(round(actual_bet_cost))}" # Log the bet cost
            # Update betting state: New level is the bet size, player is last raiser
            new_total_bet_level = self.player_bets_in_round[p_idx]
            self.current_bet = new_total_bet_level
            self.last_raise = new_total_bet_level # First aggressive action sets baseline for next raise
            self.last_raiser = p_idx
            self.raise_count_this_street = 1
            self.players_acted_this_round = {p_idx} # Action re-opened, only this player acted
            if is_all_in: self.player_all_in[p_idx] = True # Mark all-in state

        elif action_type == "raise": # Re-raising over a prior bet/raise
            if self.current_bet <= 0.01: raise ValueError("Invalid raise: Use bet instead")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET: raise ValueError("Max raises reached")
            total_bet_target = amount # Amount passed is the TOTAL target bet level
            raise_increment_needed = total_bet_target - self.current_bet # Raise must be *over* current bet level
            if raise_increment_needed <= 0.01: raise ValueError("Raise target must be > current bet")
            # Cost to reach the target
            cost_to_reach_target = total_bet_target - current_round_bet
            if cost_to_reach_target > player_stack + 0.01: raise ValueError("Cannot afford raise cost")
            actual_raise_cost = min(cost_to_reach_target, player_stack) # Cost paid by player
            actual_total_bet_reached = current_round_bet + actual_raise_cost
            is_all_in = abs(actual_raise_cost - player_stack) < 0.01
            # Check legality: Raise increment must be >= last raise increment size OR player is all-in
            min_legal_increment = max(self.last_raise, self.big_blind) # Min size increase is last raise amount (or BB if first raise)
            actual_increment_made = actual_total_bet_reached - self.current_bet # The actual increase over the facing bet
            if actual_increment_made < min_legal_increment - 0.01 and not is_all_in:
                raise ValueError(f"Raise incr {actual_increment_made:.2f} < min {min_legal_increment:.2f}")
            # Apply the raise cost
            self._deduct_bet(p_idx, actual_raise_cost)
            action_log_repr += f"r{int(round(actual_total_bet_reached))}" # Log TOTAL amount reached
            # Update state: new current bet level, last raiser, increment size
            new_bet_level = actual_total_bet_reached
            self.last_raise = new_bet_level - self.current_bet # Update size of *this* raise increment
            self.current_bet = new_bet_level # Update level needed to call
            self.last_raiser = p_idx
            self.raise_count_this_street += 1
            self.players_acted_this_round = {p_idx} # Action re-opened
            if is_all_in: self.player_all_in[p_idx] = True # Mark all-in

        else: raise ValueError(f"Unknown action type: {action_type}")

        # Add action to history only if it wasn't just a marker like P#k(c0)
        # Keep only type and amount for history clarity
        if len(action_log_repr) > len(f"P{p_idx}:"):
             self.action_sequence.append(action_log_repr)


    def get_betting_history(self): return ";".join(self.action_sequence)

    # --- Corrected get_available_actions from V17 ---
    def get_available_actions(self):
        actions = []
        player_idx = self.current_player_idx
        if player_idx == -1:
            return [] # No player's turn
        try:
             # Basic check if player can act
             if player_idx >= self.num_players or self.player_folded[player_idx] or \
                self.player_all_in[player_idx] or self.player_stacks[player_idx] < 0.01:
                  return []
        except IndexError:
             return [] # Should not happen if checks above pass

        player_stack = self.player_stacks[player_idx]
        # Crucial Fix: Get current_round_bet HERE before using it below
        current_round_bet = self.player_bets_in_round[player_idx]
        current_bet_level = self.current_bet

        # Always possible to Fold
        actions.append(("fold", 0))

        # Check or Call?
        amount_to_call = current_bet_level - current_round_bet # Amount needed to match
        can_check = amount_to_call < 0.01 # Check possible if no amount needed

        if can_check:
            actions.append(("check", 0))
        else: # Call is needed
            call_cost = min(amount_to_call, player_stack) # Cost capped by stack
            # Only add call if there's a significant cost
            if call_cost > 0.01: # Using small threshold
                actions.append(("call", int(round(call_cost)))) # Tuple: (type, COST)

        # Bet or Raise? (Aggression)
        can_aggress = self.raise_count_this_street < self.MAX_RAISES_PER_STREET
        # Can only aggress if stack is greater than cost to call (must be able to increase total bet)
        if can_aggress and player_stack > max(0.01, amount_to_call):
            # Define min/max aggression amounts based on rules
            min_legal_aggress_target_to = float('inf') # Smallest TOTAL bet allowed
            max_legal_aggress_target_to = current_round_bet + player_stack # All-in TOTAL target

            if current_bet_level < 0.01: # Current action is BET (no prior aggression)
                 action_prefix = "bet"
                 min_bet_cost = min(player_stack, max(self.big_blind, 1.0))
                 min_legal_aggress_target_to = current_round_bet + min_bet_cost # Min total bet = current + cost
            else: # Current action is RAISE
                 action_prefix = "raise"
                 # Minimum legal raise increment size
                 min_legal_increment = max(self.last_raise, self.big_blind)
                 # Minimum total amount to raise TO
                 min_raise_target_to = current_bet_level + min_legal_increment
                 # The actual minimum might be capped by going all-in
                 min_legal_aggress_target_to = min(max_legal_aggress_target_to, min_raise_target_to)

            # Add Min Legal Aggressive Action (if possible & actually aggressive)
            if min_legal_aggress_target_to > current_bet_level + 0.01:
                actions.append((action_prefix, int(round(min_legal_aggress_target_to)))) # Tuple: (type, TARGET_TOTAL_AMOUNT)

            # Add All-In Aggressive Action (if possible, aggressive, and distinct from min legal)
            if max_legal_aggress_target_to > current_bet_level + 0.01 and \
               abs(max_legal_aggress_target_to - min_legal_aggress_target_to) > 0.01:
               actions.append((action_prefix, int(round(max_legal_aggress_target_to)))) # Tuple: (type, TARGET_TOTAL_AMOUNT)


        # --- Final Filtering and Sorting ---
        def sort_key(a):
            t, amt = a
            o = {"fold":0,"check":1,"call":2,"bet":3,"raise":4}
            return (o.get(t, 99), amt) # Sort by type, then amount

        final_actions = []
        seen_actions = set()
        for act_tuple in sorted(actions, key=sort_key): # Sort intermediate list
             act_type, act_amount = act_tuple
             # Standardize action tuple for key checking
             action_key = (act_type, int(round(act_amount)))

             # Calculate cost of this action
             cost = 0.0
             if action_key[0] == 'call': cost = action_key[1] # Call amount IS cost
             elif action_key[0] == 'bet': cost = action_key[1] # Bet amount IS cost (from 0)
             elif action_key[0] == 'raise': cost = action_key[1] - current_round_bet # Raise cost is Target - AlreadyIn
             # Handle potential negative cost if something strange happens
             cost = max(0.0, cost)

             # Add if unique and player can afford the cost (with small tolerance)
             if action_key not in seen_actions and cost <= player_stack + 0.01:
                 final_actions.append(act_tuple) # Add the ORIGINAL tuple
                 seen_actions.add(action_key)

        return final_actions # Return the filtered, sorted list


    def _is_betting_round_over(self):
        """ Checks if the current betting round has concluded. """
        if len(self.active_players) < 2: return True
        players_who_can_voluntarily_act = []
        for p_idx in self.active_players:
            if not self.player_folded[p_idx] and not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01:
                players_who_can_voluntarily_act.append(p_idx)
        num_can_act = len(players_who_can_voluntarily_act)
        if num_can_act == 0: return True
        if num_can_act == 1:
            the_player = players_who_can_voluntarily_act[0]; has_acted = the_player in self.players_acted_this_round
            facing_bet = (self.current_bet - self.player_bets_in_round[the_player]) > 0.01
            is_preflop = (self.betting_round == self.PREFLOP)
            bb_player = self._find_player_relative_to_dealer(2 if self.num_players > 2 else 1); is_bb_player = (the_player == bb_player)
            no_raise_yet = (self.raise_count_this_street <= 1) # Only blinds posted
            if is_preflop and is_bb_player and no_raise_yet and not facing_bet and not has_acted: return False # BB option
            return (not facing_bet or has_acted) # Otherwise, round over if only one player unless they face bet & haven't acted
        # Case 3: Multiple players can act
        all_matched = True; all_acted = True
        for p_idx in players_who_can_voluntarily_act:
             if abs(self.player_bets_in_round[p_idx] - self.current_bet) > 0.01: all_matched = False; break
             if p_idx not in self.players_acted_this_round: all_acted = False; break
        return all_matched and all_acted

    # --- Corrected _try_advance_round ---
    def _try_advance_round(self):
        """ Attempts to deal next street or end hand if betting round finished. V4 Corrected """
        if len(self.active_players) <= 1:
            if self.betting_round != self.HAND_OVER: self.betting_round = self.HAND_OVER
            self.current_player_idx = -1; return

        if self._check_all_active_are_allin() and self.betting_round < self.SHOWDOWN:
            # Deals sequentially until river or error
            card_count = len(self.community_cards)
            if card_count < 3:
                 if not self.deal_flop(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return
            if len(self.community_cards) < 4:
                 if not self.deal_turn(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return
            if len(self.community_cards) < 5:
                 if not self.deal_river(): self.betting_round = self.HAND_OVER; self.current_player_idx = -1; return

            if self.betting_round != self.HAND_OVER:
                self.betting_round = self.SHOWDOWN; self.current_player_idx = -1
                self.players_acted_this_round = set()
            return

        # --- Normal round advancement ---
        current_round = self.betting_round; round_advanced_successfully = False
        if current_round == self.PREFLOP: round_advanced_successfully = self.deal_flop()
        elif current_round == self.FLOP: round_advanced_successfully = self.deal_turn()
        elif current_round == self.TURN: round_advanced_successfully = self.deal_river()
        elif current_round == self.RIVER: self.betting_round = self.SHOWDOWN; self.current_player_idx = -1; self.players_acted_this_round = set(); round_advanced_successfully = True

        if not round_advanced_successfully and self.betting_round < self.SHOWDOWN:
             if self.betting_round != self.HAND_OVER: self.betting_round = self.HAND_OVER
             self.current_player_idx = -1; self.players_acted_this_round = set()

    def is_terminal(self):
        active_count = len(self.active_players) if hasattr(self, 'active_players') else 0
        return (active_count <= 1) or (self.betting_round >= self.SHOWDOWN)

    # --- Final get_utility (Internal Calc) ---
    def get_utility(self, player_idx, initial_stacks=None):
        if not self.is_terminal(): return 0.0
        if initial_stacks is None: print(f"ERROR get_utility: initial_stacks missing P{player_idx}. Ret 0."); return 0.0
        if not (0 <= player_idx < self.num_players and isinstance(initial_stacks, list) and player_idx < len(initial_stacks) and player_idx < len(self.player_stacks)): print(f"WARN get_utility: Index mismatch P{player_idx}"); return 0.0

        initial_stack = 0.0; current_stack = 0.0
        try: i_s = initial_stacks[player_idx]; assert isinstance(i_s, (int, float)) and not np.isnan(i_s) and not np.isinf(i_s); initial_stack = float(i_s)
        except Exception: print(f"WARN get_utility: Invalid init stack P{player_idx}"); return 0.0
        try: c_s = self.player_stacks[player_idx]; assert isinstance(c_s, (int,float)) and not np.isnan(c_s) and not np.isinf(c_s); current_stack = float(c_s)
        except Exception: print(f"WARN get_utility: Invalid curr stack P{player_idx}"); return 0.0

        final_effective_stack = current_stack; pot_to_distribute = self.pot if isinstance(self.pot, (int, float)) else 0.0
        eligible_for_pot = [p for p in range(self.num_players) if not self.player_folded[p]]

        if pot_to_distribute > 0.01 and len(eligible_for_pot) > 0:
            if len(eligible_for_pot) == 1:
                 if eligible_for_pot[0] == player_idx: final_effective_stack += pot_to_distribute
            else: # Simulate Showdown/Side Pots internally
                 evaluated_hands = {}; valid_showdown_players = []
                 for p_idx_eval in eligible_for_pot:
                      if p_idx_eval >= len(self.hole_cards) or len(self.hole_cards[p_idx_eval]) != 2: continue
                      all_cards_for_eval = self.hole_cards[p_idx_eval] + self.community_cards
                      if len(all_cards_for_eval) < 5: continue
                      try: evaluated_hands[p_idx_eval] = HandEvaluator.evaluate_hand(all_cards_for_eval); valid_showdown_players.append(p_idx_eval)
                      except Exception as eval_e: print(f"WARN get_utility: HandEval fail P{p_idx_eval}: {eval_e}"); continue

                 if player_idx in valid_showdown_players:
                      player_winnings = 0.0
                      contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players], key=lambda x: x[1])
                      side_pots = []; last_contribution_level = 0.0; eligible_for_next_pot = valid_showdown_players[:]
                      for p_idx_sp, total_contribution in contributions:
                           contribution_increment = total_contribution - last_contribution_level
                           if contribution_increment > 0.01:
                               num_eligible = len(eligible_for_next_pot); pot_amount = contribution_increment * num_eligible
                               if pot_amount > 0.01: side_pots.append({'amount': pot_amount, 'eligible': eligible_for_next_pot[:]})
                               last_contribution_level = total_contribution
                           if p_idx_sp in eligible_for_next_pot: eligible_for_next_pot.remove(p_idx_sp)
                      if not side_pots and pot_to_distribute > 0.01: # Handle main pot if no side pots created
                           side_pots.append({'amount': pot_to_distribute, 'eligible': valid_showdown_players[:]})

                      for i, pot_info in enumerate(side_pots): # Award side pots virtually
                           pot_amount = pot_info.get('amount', 0.0); eligible_sp = pot_info.get('eligible', [])
                           if pot_amount < 0.01 or not eligible_sp or player_idx not in eligible_sp: continue
                           eligible_hands_this_pot = {p: evaluated_hands[p] for p in eligible_sp if p in evaluated_hands}
                           if not eligible_hands_this_pot: continue
                           best_hand_value = max(eligible_hands_this_pot.values())
                           if evaluated_hands.get(player_idx) == best_hand_value:
                                pot_winners = [p for p, hand_val in eligible_hands_this_pot.items() if hand_val == best_hand_value]
                                if pot_winners: player_winnings += pot_amount / len(pot_winners)
                      final_effective_stack += player_winnings

        utility = final_effective_stack - initial_stack
        # print(f"    DEBUG get_utility FINAL: P{player_idx}, EffFinal={final_effective_stack:.1f}, Init={initial_stack:.1f}, Util={utility:.1f}") # Keep Commented
        if np.isnan(utility) or np.isinf(utility): utility = 0.0
        return utility


    # --- Restore ORIGINAL determine_winners ---
    def determine_winners(self, player_names=None):
        # [ ... Paste ORIGINAL determine_winners logic here ... ]
        if not self.is_terminal(): return []
        if not self.active_players and self.pot < 0.01: return []
        if not self.active_players and self.pot >= 0.01: self.pot = 0.0; return []

        total_pot_to_distribute = self.pot; self.pot = 0.0; pots_summary = []
        eligible_for_pot = [p for p in range(self.num_players) if not self.player_folded[p]]

        if len(eligible_for_pot) == 1: # Uncontested
            winner_idx = eligible_for_pot[0]; amount_won = total_pot_to_distribute
            if 0 <= winner_idx < len(self.player_stacks):
                 self.player_stacks[winner_idx] += amount_won
                 pots_summary = [{'winners': [winner_idx], 'amount': amount_won, 'eligible': [winner_idx], 'desc': 'Uncontested'}]
            return pots_summary

        evaluated_hands = {}; valid_showdown_players = [] # Showdown
        for p_idx in eligible_for_pot:
            if p_idx >= len(self.hole_cards) or len(self.hole_cards[p_idx]) != 2: continue
            all_cards_for_eval = self.hole_cards[p_idx] + self.community_cards
            if len(all_cards_for_eval) < 5: continue
            try: evaluated_hands[p_idx] = HandEvaluator.evaluate_hand(all_cards_for_eval); valid_showdown_players.append(p_idx)
            except Exception: continue
        if not valid_showdown_players: return []

        contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players], key=lambda x: x[1]) # Side Pots
        side_pots = []; last_contribution_level = 0.0; eligible_for_next_pot = valid_showdown_players[:]
        for p_idx_sp, total_contribution in contributions:
             contribution_increment = total_contribution - last_contribution_level
             if contribution_increment > 0.01:
                 num_eligible = len(eligible_for_next_pot); pot_amount = contribution_increment * num_eligible
                 if pot_amount > 0.01: side_pots.append({'amount': pot_amount, 'eligible': eligible_for_next_pot[:]})
                 last_contribution_level = total_contribution
             if p_idx_sp in eligible_for_next_pot: eligible_for_next_pot.remove(p_idx_sp)
        # Add main pot if no side pots were needed (e.g. all contributed same)
        if not side_pots and total_pot_to_distribute > 0.01:
             side_pots.append({'amount': total_pot_to_distribute, 'eligible': valid_showdown_players[:]})

        distributed_total = 0; pots_summary = [] # Award Pots
        for i, pot_info in enumerate(side_pots):
             pot_amount = pot_info['amount']; eligible_players = pot_info['eligible']
             if pot_amount < 0.01 or not eligible_players: continue
             eligible_hands = {p: evaluated_hands[p] for p in eligible_players if p in evaluated_hands}
             if not eligible_hands: continue
             best_hand_value = max(eligible_hands.values())
             pot_winners = [p for p, hand_val in eligible_hands.items() if hand_val == best_hand_value]
             if pot_winners:
                 winner_share = pot_amount / len(pot_winners); distributed_total += pot_amount
                 for w_idx in pot_winners:
                      if 0 <= w_idx < len(self.player_stacks): self.player_stacks[w_idx] += winner_share
                 pot_desc = f"Side Pot {i+1}" if len(side_pots)>1 else "Main Pot"
                 pots_summary.append({'winners':pot_winners, 'amount':pot_amount, 'eligible':eligible_players, 'desc': pot_desc})
        return pots_summary


    def clone(self): return deepcopy(self) # Simplified

    def get_position(self, player_idx):
        if not (0 <= player_idx < self.num_players) or self.num_players <= 1: return -1
        return (player_idx - self.dealer_position + self.num_players) % self.num_players

    def __str__(self): # Simplified String
        round_name = self.ROUND_NAMES.get(self.betting_round, f"R{self.betting_round}")
        turn = f"P{self.current_player_idx}" if self.current_player_idx != -1 else "N"
        board = ' '.join(map(str, self.community_cards)) if self.community_cards else "-"
        hist = self.get_betting_history()[-60:]
        s = f"Rnd:{round_name},Turn:{turn},Pot:{self.pot:.0f},Board:[{board}]\n"
        for i in range(self.num_players):
            state = "F" if self.player_folded[i] else ("A" if self.player_all_in[i] else " ")
            s += f" P{i}{state}:Stk={self.player_stacks[i]:.0f},BetRnd={self.player_bets_in_round[i]:.0f}\n"
        s += f" Hist:...{hist}"
        return s

# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
