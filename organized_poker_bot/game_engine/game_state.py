# --- START OF FILE organized_poker_bot/game_engine/game_state.py ---
"""
Game state implementation for poker games.
(Refactored V36: Comprehensive semicolon removal and formatting)
"""

import random
import math
import sys
import os
import traceback
from collections import defaultdict
from copy import deepcopy # Use deepcopy for cloning

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute imports
try:
    from organized_poker_bot.game_engine.deck import Deck
    from organized_poker_bot.game_engine.card import Card
    from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
except ImportError as e:
    print(f"ERROR importing engine components in GameState: {e}")
    sys.exit(1)


class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {
        0: "Preflop", 1: "Flop", 2: "Turn", 3: "River",
        4: "Showdown", 5: "Hand Over"
    }
    MAX_RAISES_PER_STREET = 7

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        if not (2 <= num_players <= 9):
            raise ValueError("Num players must be between 2 and 9")
        self.num_players = int(num_players)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)

        # State Initialization
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
        self.verbose_debug = False

    # --- Helper Methods ---
    def _get_next_active_player(self, start_idx):
        """ Finds the next active player index in rotation, skipping inactive/busted players. """
        if not self.active_players or self.num_players == 0:
            return None
        valid_start = start_idx if 0 <= start_idx < self.num_players else -1
        current_idx = (valid_start + 1) % self.num_players

        for _ in range(self.num_players * 2): # Limit loop to prevent infinite loops
             if current_idx in self.active_players and \
                0 <= current_idx < len(self.player_stacks) and \
                self.player_stacks[current_idx] > 0.01 and \
                not self.player_folded[current_idx]: # Check folded status too
                 return current_idx
             current_idx = (current_idx + 1) % self.num_players
             # Check if we wrapped around without finding anyone
             if current_idx == (valid_start + 1) % self.num_players and _ > self.num_players:
                  break
        return None # No active player found

    def _find_player_relative_to_dealer(self, offset):
        """ Finds an active player at a specific offset from the dealer button. """
        if not self.active_players or self.num_players == 0:
            return None
        dealer = getattr(self, 'dealer_position', 0) % self.num_players
        start_idx = (dealer + offset) % self.num_players
        current_idx = start_idx

        for _ in range(self.num_players * 2): # Limit loop
            if current_idx in self.active_players and \
               0 <= current_idx < len(self.player_stacks) and \
               self.player_stacks[current_idx] > 0.01:
                return current_idx
            current_idx = (current_idx + 1) % self.num_players
            # Check if we wrapped around
            if current_idx == start_idx and _ > self.num_players:
                 break
        return None # No suitable player found

    # --- Hand Setup Methods ---
    def start_new_hand(self, dealer_pos, player_stacks):
        """ Initializes state for a new hand. """
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
        else:
            # Not enough active players to start a hand
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1

    def _deal_hole_cards(self):
        """ Deals two hole cards to each active player. """
        if not self.active_players:
            return

        # Find the first player to deal to (typically player left of dealer)
        start_player = self._find_player_relative_to_dealer(1)
        if start_player is None: # Fallback if Relative search fails (e.g. only 2 players, one is dealer)
             if self.active_players: start_player = self.active_players[0]
             else: self.betting_round = self.HAND_OVER; return

        current_deal_idx = start_player
        # Deal two rounds of cards
        for round_num in range(2):
            players_dealt_this_round = 0
            attempts = 0
            start_loop_idx = current_deal_idx
            while players_dealt_this_round < len(self.active_players) and attempts < self.num_players * 2:
                attempts += 1
                if 0 <= current_deal_idx < self.num_players:
                    if current_deal_idx in self.active_players:
                        # Ensure hole_cards list is large enough
                        while len(self.hole_cards) <= current_deal_idx:
                             self.hole_cards.append([])
                        # Deal if player needs a card for this round
                        if len(self.hole_cards[current_deal_idx]) == round_num:
                            if len(self.deck) > 0:
                                self.hole_cards[current_deal_idx].append(self.deck.deal())
                                players_dealt_this_round += 1
                            else:
                                print("ERROR: Deck empty during hole card deal.")
                                self.betting_round = self.HAND_OVER # End hand immediately
                                return
                # Move to the next player index
                current_deal_idx = (current_deal_idx + 1) % self.num_players
                # Check if we looped without dealing enough cards
                if current_deal_idx == start_loop_idx and players_dealt_this_round < len(self.active_players) and attempts > self.num_players:
                     print(f"ERROR: Could not deal card round {round_num + 1} - Stuck in loop?")
                     self.betting_round = self.HAND_OVER
                     return

            # Update starting index for next round of dealing (continues rotation)
            start_player = current_deal_idx # Next round starts from next player


            if players_dealt_this_round < len(self.active_players):
                print(f"ERROR: Failed to deal hole cards for round {round_num + 1}.")
                self.betting_round = self.HAND_OVER
                return

    def _deduct_bet(self, player_idx, amount_to_deduct):
        """ Internal helper to deduct bet from stack and update pot/state. """
        if not (0 <= player_idx < self.num_players and amount_to_deduct >= 0):
            print(f"WARN: Invalid call to _deduct_bet P:{player_idx}, Amt:{amount_to_deduct}")
            return # Invalid parameters

        actual_deduction = min(amount_to_deduct, self.player_stacks[player_idx])
        if actual_deduction <= 0.01 and amount_to_deduct > 0: # Don't process negligible amounts unless it's intended 0
             actual_deduction = 0.0

        if actual_deduction == 0 and amount_to_deduct > 0: return # Skip if effectively no bet made

        self.player_stacks[player_idx] -= actual_deduction
        self.player_bets_in_round[player_idx] += actual_deduction
        self.player_total_bets_in_hand[player_idx] += actual_deduction
        self.pot += actual_deduction

        # Check for all-in
        if abs(self.player_stacks[player_idx]) < 0.01:
            self.player_all_in[player_idx] = True

    def _post_blinds(self):
        """ Posts Small and Big Blinds. """
        if len(self.active_players) < 2:
            return # Not enough players for blinds

        sb_player, bb_player = None, None

        # Determine SB and BB positions
        if self.num_players == 2: # Heads Up logic
            sb_player = self._find_player_relative_to_dealer(0) # Dealer posts SB
            bb_player = self._find_player_relative_to_dealer(1) # Non-dealer posts BB
        else: # 3+ players
            sb_player = self._find_player_relative_to_dealer(1) # Player left of dealer is SB
            bb_player = self._find_player_relative_to_dealer(2) # Player 2 left of dealer is BB

        self.raise_count_this_street = 0 # Reset raise count for preflop

        # Post Small Blind
        if sb_player is not None and 0 <= sb_player < len(self.player_stacks):
            amount_sb = min(self.small_blind, self.player_stacks[sb_player])
            if amount_sb > 0:
                self._deduct_bet(sb_player, amount_sb)
                self.action_sequence.append(f"P{sb_player}:sb{int(round(amount_sb))}")
        else: print("WARN: Could not find SB player.")

        # Post Big Blind
        bb_posted_amount = 0.0
        if bb_player is not None and 0 <= bb_player < len(self.player_stacks):
            needed_bb = self.big_blind - self.player_bets_in_round[bb_player] # Check if BB already posted by straddle etc (rare)
            amount_bb = min(needed_bb, self.player_stacks[bb_player])
            if amount_bb > 0:
                self._deduct_bet(bb_player, amount_bb)
                # Use the total amount in round for logging, consistent with raises/calls
                log_bb_amt = int(round(self.player_bets_in_round[bb_player]))
                self.action_sequence.append(f"P{bb_player}:bb{log_bb_amt}")
                bb_posted_amount = self.player_bets_in_round[bb_player]
        else: print("WARN: Could not find BB player.")


        # Set initial betting level and last raiser info
        self.current_bet = self.big_blind # The level to call is the BB
        self.last_raise = self.big_blind # Initial 'raise' is the BB itself
        if bb_player is not None and bb_posted_amount >= self.big_blind - 0.01:
            # If BB successfully posted full amount (or went all-in for >= BB)
            self.last_raiser = bb_player
            self.raise_count_this_street = 1 # The BB post counts as the first 'raise' conceptually
        else:
            # If BB couldn't post full blind, the SB action might be first 'raise'
            self.last_raiser = None
            self.raise_count_this_street = 0


    # --- Round Progression ---
    def _start_betting_round(self):
        """ Resets round state and finds the first player to act. """
        # Reset round-specific state (bets, last raiser) if not Preflop
        if self.betting_round != self.PREFLOP:
            self.current_bet = 0.0
            self.last_raiser = None
            self.last_raise = self.big_blind # Minimum raise reference point (use BB post-flop too)
            self.raise_count_this_street = 0
            # Reset bets contributed THIS ROUND
            for i in range(self.num_players):
                if 0 <= i < len(self.player_bets_in_round):
                     self.player_bets_in_round[i] = 0.0

        self.players_acted_this_round = set()
        first_player_to_act = None

        if self.betting_round == self.PREFLOP:
             # HU: SB (pos 0) acts first. Standard deal P1 is BB/D, P0 is SB. SB offset=0 from P1.
             if self.num_players == 2:
                 # Find player who is NOT the dealer (should be SB)
                 sb_p = self._find_player_relative_to_dealer(0)
                 first_player_to_act = sb_p
             else: # 3+ players: UTG (player after BB) acts first
                 bb_player = self._find_player_relative_to_dealer(2)
                 start_search = bb_player if bb_player is not None else self._find_player_relative_to_dealer(1) # Start search from BB or SB if BB not found
                 first_player_to_act = self._get_next_active_player(start_search) if start_search is not None else self._get_next_active_player(self.dealer_position)
        else: # Postflop: Player left of dealer acts first
             first_player_to_act = self._get_next_active_player(self.dealer_position)

        self.current_player_idx = first_player_to_act if first_player_to_act is not None else -1

        # Check if betting is possible or should be skipped
        players_who_can_voluntarily_act = [
             p for p in self.active_players
             if 0 <= p < len(self.player_all_in) and not self.player_all_in[p] and \
                0 <= p < len(self.player_folded) and not self.player_folded[p] and \
                0 <= p < len(self.player_stacks) and self.player_stacks[p] > 0.01
        ]

        if len(players_who_can_voluntarily_act) <= 1:
             # If 0 or 1 player can act, no betting round needed. Current player is set to -1 or remains invalid.
             # Exception: Preflop BB option - check _is_betting_round_over handles this.
             if self.betting_round != self.PREFLOP: # Postflop, skip betting round
                  self.current_player_idx = -1 # Signal no action needed this round

        # Also check if only players left are all-in (can occur if side pots exist)
        # If all remaining active players are all-in, no betting.
        if self._check_all_active_are_allin():
             self.current_player_idx = -1


    def _deal_community_card(self, burn=True):
        """ Deals one community card, optionally burning one first. """
        if burn:
             if len(self.deck) > 0:
                 self.deck.deal() # Burn card
             else: return False # Cannot burn from empty deck
        if len(self.deck) > 0:
            self.community_cards.append(self.deck.deal())
            return True
        else:
            return False # Cannot deal from empty deck

    def deal_flop(self):
        """ Deals the flop (3 cards). """
        if self.betting_round != self.PREFLOP or len(self.active_players) <= 1 or len(self.community_cards) >= 3:
             return False # Cannot deal flop in wrong round or state
        if len(self.deck) < 4: # 1 burn + 3 flop
             self.betting_round = self.HAND_OVER
             return False
        self.deck.deal() # Burn card
        # Deal 3 flop cards without burning between them
        flop_deal_success = all(self._deal_community_card(False) for _ in range(3))
        if not flop_deal_success:
            self.betting_round = self.HAND_OVER # End hand if deck ran out mid-flop
            return False

        self.betting_round = self.FLOP
        # Determine if betting happens or we fast-forward
        if self._check_all_active_are_allin():
             self.current_player_idx = -1 # All-in, skip betting
        else:
             self._start_betting_round() # Start flop betting round
        return True

    def deal_turn(self):
        """ Deals the turn card (1 card). """
        if self.betting_round != self.FLOP or len(self.active_players) <= 1 or len(self.community_cards) >= 4:
            return False
        if len(self.deck) < 2: # 1 burn + 1 turn
             self.betting_round = self.HAND_OVER
             return False
        if not self._deal_community_card(True): # Deal turn card with burn
             self.betting_round = self.HAND_OVER
             return False

        self.betting_round = self.TURN
        if self._check_all_active_are_allin():
            self.current_player_idx = -1
        else:
            self._start_betting_round()
        return True

    def deal_river(self):
        """ Deals the river card (1 card). """
        if self.betting_round != self.TURN or len(self.active_players) <= 1 or len(self.community_cards) >= 5:
            return False
        if len(self.deck) < 2: # 1 burn + 1 river
             self.betting_round = self.HAND_OVER
             return False
        if not self._deal_community_card(True): # Deal river card with burn
             self.betting_round = self.HAND_OVER
             return False

        self.betting_round = self.RIVER
        # Check if final betting round occurs or go straight to showdown
        if self._check_all_active_are_allin():
             self.betting_round = self.SHOWDOWN # Advance directly if all-in
             self.current_player_idx = -1
             self.players_acted_this_round = set()
        else:
            self._start_betting_round() # Start river betting round
        return True

    def _check_all_active_are_allin(self):
        """
        Checks if betting action should stop because all players eligible
        to continue the hand are either all-in or only one player remains
        who is not all-in.
        """
        # Get players who haven't folded yet
        non_folded_players = [p for p in self.active_players if not self.player_folded[p]]

        # If 0 or 1 players haven't folded, no more betting possible
        if len(non_folded_players) <= 1:
            return True

        # Count how many of the non-folded players are NOT all-in and have chips
        count_can_still_act = 0
        for p_idx in non_folded_players:
            if not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01:
                count_can_still_act += 1

        # If there is 0 or 1 player who can still make a betting decision,
        # then everyone else is effectively all-in (relative to this round),
        # and the board should be dealt out.
        return count_can_still_act <= 1

    def _move_to_next_player(self):
        """ Finds the next player in rotation and updates current_player_idx. """
        if self.current_player_idx != -1:
            next_p_idx = self._get_next_active_player(self.current_player_idx)
            self.current_player_idx = next_p_idx if next_p_idx is not None else -1

    # --- Action Handling ---
    def apply_action(self, action):
        """
        Applies a player action to a *clone* of the game state and advances state.

        Args:
            action (tuple): The action tuple (action_type, amount).

        Returns:
            GameState: The new game state after the action.

        Raises:
            ValueError: If the action is invalid or formatting is wrong.
        """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple of (action_type, amount)")

        action_type, amount_input = action
        try:
            amount = float(amount_input) # Ensure amount is numeric
        except (ValueError, TypeError):
            raise ValueError(f"Invalid action amount: {amount_input}")

        acting_player_idx = self.current_player_idx

        # --- Perform Basic State Validations ---
        if acting_player_idx == -1:
            raise ValueError("Invalid action: No player's turn")
        if not (0 <= acting_player_idx < self.num_players):
            raise ValueError(f"Invalid action: Player index {acting_player_idx} out of range")
        if not (acting_player_idx < len(self.player_folded) and
                acting_player_idx < len(self.player_all_in) and
                acting_player_idx < len(self.player_stacks)):
             raise ValueError(f"Invalid action: Player {acting_player_idx} lists out of sync")
        # Check if player is active in the hand (redundant with active_players?)
        # Check folded / all-in status
        if self.player_folded[acting_player_idx]:
            raise ValueError(f"Invalid action: Player {acting_player_idx} has folded")
        if self.player_all_in[acting_player_idx]:
            # All-in players technically can't act, but this might be called recursively? Check needed.
            # If already all-in, moving to next player is likely correct. Don't raise error?
             print(f"INFO: apply_action called for already all-in player {acting_player_idx}, skipping logic.")
             # We still need to advance state if round should end etc. Clone first.
             new_state_skip = self.clone()
             new_state_skip._move_to_next_player() # Attempt to move turn
             if new_state_skip._is_betting_round_over():
                  new_state_skip._try_advance_round()
             return new_state_skip

        # --- Apply Action to a Clone ---
        new_state = self.clone()
        try:
            new_state._apply_action_logic(acting_player_idx, action_type, amount) # Mutates the clone
        except ValueError as e:
            print(f"ERROR Applying action P{acting_player_idx} {action}: {e}")
            traceback.print_exc(limit=1)
            raise # Re-raise the validation error

        # --- Advance Game State (Turn or Round) on the clone ---
        if new_state._is_betting_round_over():
            new_state._try_advance_round() # Advance round if betting over
        else:
            # Betting not over, just move to next player
            new_state._move_to_next_player()
            # Second check: Did moving player END the round? (e.g. BB checks preflop option)
            if new_state.current_player_idx != -1 and new_state._is_betting_round_over():
                 new_state._try_advance_round()

        return new_state # Return the modified clone


    def _apply_action_logic(self, p_idx, action_type, amount):
        """ Internal logic to modify state based on a validated action. Mutates self. """
        player_stack = self.player_stacks[p_idx]
        current_round_bet = self.player_bets_in_round[p_idx]

        self.players_acted_this_round.add(p_idx) # Mark player as acted
        action_log_repr = f"P{p_idx}:"

        # Handle Fold
        if action_type == "fold":
            self.player_folded[p_idx] = True
            if p_idx in self.active_players:
                self.active_players.remove(p_idx)
            action_log_repr += "f"
            # Check if hand ends immediately due to fold
            if len(self.active_players) <= 1:
                 self.betting_round = self.HAND_OVER
                 self.current_player_idx = -1 # Hand over, no next player

        # Handle Check
        elif action_type == "check":
            # Validate check: Can only check if current bet matches player's bet in round
            if self.current_bet - current_round_bet > 0.01:
                raise ValueError("Invalid check: Must call or raise")
            action_log_repr += "k" # Use 'k' for check

        # Handle Call
        elif action_type == "call":
            amount_needed = self.current_bet - current_round_bet
            if amount_needed <= 0.01:
                 # Treat calling 0 (or negligible amount) like a check if allowed
                 # Could raise ValueError if a check wasn't an option but call 0 was attempted
                 action_log_repr += "k(c0)" # Log as check due to call 0
                 return # No state change needed besides marking acted

            # Amount parameter for call usually represents the COST
            call_cost = min(amount_needed, player_stack)
            if call_cost < 0: call_cost = 0 # Should not happen
            self._deduct_bet(p_idx, call_cost)
            action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}" # Log total bet AFTER call

        # Handle Bet (only valid if self.current_bet is ~0)
        elif action_type == "bet":
            if self.current_bet > 0.01:
                raise ValueError("Invalid bet: Raise required instead")
            if amount < 0.01:
                raise ValueError("Invalid bet: Amount must be positive")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                raise ValueError("Invalid bet: Maximum number of raises reached")

            min_bet_amount = max(self.big_blind, 1.0) # Usually BB, or 1 if BB is less
            actual_bet = min(amount, player_stack) # Amount is the bet cost

            # Check if bet is valid size (>= min bet OR all-in)
            is_all_in = abs(actual_bet - player_stack) < 0.01
            if actual_bet < min_bet_amount - 0.01 and not is_all_in:
                raise ValueError(f"Invalid bet: Amount {actual_bet:.2f} is less than minimum {min_bet_amount:.2f}")

            self._deduct_bet(p_idx, actual_bet)
            action_log_repr += f"b{int(round(actual_bet))}" # Log bet amount

            # Update betting state after a bet
            new_total_bet_level = self.player_bets_in_round[p_idx]
            self.current_bet = new_total_bet_level
            self.last_raise = new_total_bet_level # Bet acts as the first 'raise' size reference
            self.last_raiser = p_idx
            self.raise_count_this_street = 1
            self.players_acted_this_round = {p_idx} # Reset players acted for this new level

        # Handle Raise (only valid if self.current_bet > 0)
        elif action_type == "raise":
            if self.current_bet <= 0.01:
                raise ValueError("Invalid raise: Bet required instead")
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET:
                raise ValueError("Invalid raise: Maximum number of raises reached")

            # Amount here is the TOTAL bet amount TO raise to
            total_bet_target = amount
            raise_increment = total_bet_target - current_round_bet

            if raise_increment <= 0.01:
                raise ValueError("Invalid raise: Must increase total bet amount")
            if raise_increment > player_stack + 0.01: # Check affordability
                raise ValueError(f"Invalid raise: Cost {raise_increment:.2f} > stack {player_stack:.2f}")

            # Validate raise size based on last raise
            min_raise_increment = max(self.last_raise, self.big_blind) # Min increase is last raise amount or BB
            min_legal_total_target = self.current_bet + min_raise_increment

            actual_raise_cost = min(raise_increment, player_stack) # The amount to deduct
            actual_total_bet_reached = current_round_bet + actual_raise_cost
            is_all_in = abs(actual_raise_cost - player_stack) < 0.01

            # Check if the raise meets the minimum legal size requirement OR is all-in
            if actual_total_bet_reached < min_legal_total_target - 0.01 and not is_all_in:
                raise ValueError(f"Invalid raise: Total bet {actual_total_bet_reached:.2f} < minimum required {min_legal_total_target:.2f}")

            self._deduct_bet(p_idx, actual_raise_cost)
            action_log_repr += f"r{int(round(actual_total_bet_reached))}" # Log total bet amount AFTER raise

            # Update betting state after a raise
            new_bet_level = self.player_bets_in_round[p_idx]
            # last_raise should be the SIZE of the increment over the previous bet level
            self.last_raise = new_bet_level - self.current_bet
            self.current_bet = new_bet_level # New level to call
            self.last_raiser = p_idx
            self.raise_count_this_street += 1
            self.players_acted_this_round = {p_idx} # Reset players acted for new level

            if is_all_in:
                 self.player_all_in[p_idx] = True # Mark player explicitly if raise was all-in

        # Unknown action type
        else:
            raise ValueError(f"Unknown action type received: {action_type}")

        # Add action to history if it wasn't just a log placeholder
        if action_log_repr != f"P{p_idx}:":
            self.action_sequence.append(action_log_repr)


    def get_betting_history(self):
        """ Returns a string representation of the betting sequence. """
        return ";".join(self.action_sequence)

    def get_available_actions(self):
        """
        Returns a list of valid action tuples (type, amount) for the current player.
        Amount for call/bet/raise is typically the cost or target total bet amount. Check specific implementation.
        Based on V33 code structure, seems to be:
         - fold: ('fold', 0)
         - check: ('check', 0)
         - call: ('call', COST_TO_CALL)
         - bet: ('bet', BET_AMOUNT_COST) -> **Careful: Check if applies wants cost or total**
         - raise: ('raise', TARGET_TOTAL_BET_AMOUNT) -> **Careful: Check if applies wants cost or total**

        Let's align `get_available_actions` with how `_apply_action_logic` expects amounts:
         - call: needs COST
         - bet: needs COST
         - raise: needs TARGET_TOTAL_AMOUNT
        """
        actions = []
        player_idx = self.current_player_idx
        if player_idx == -1:
            return []

        try: # Safety checks for player state access
            if player_idx >= len(self.player_folded) or self.player_folded[player_idx] or \
               player_idx >= len(self.player_all_in) or self.player_all_in[player_idx] or \
               player_idx >= len(self.player_stacks) or self.player_stacks[player_idx] < 0.01:
                return [] # Player cannot act
        except IndexError:
            print(f"WARN: get_available_actions IndexError for P{player_idx}")
            return []

        player_stack = self.player_stacks[player_idx]
        current_round_bet = self.player_bets_in_round[player_idx]
        current_bet_level = self.current_bet

        # 1. Fold
        actions.append(("fold", 0))

        # 2. Check or Call
        amount_to_call = current_bet_level - current_round_bet
        can_check = amount_to_call < 0.01

        if can_check:
            actions.append(("check", 0))
        else:
            # Can Call? Cost is amount_to_call capped by stack
            call_cost = min(amount_to_call, player_stack)
            if call_cost > 0.01: # Only add call if cost is significant
                actions.append(("call", int(round(call_cost)))) # Add CALL with COST

        # 3. Bet or Raise (Aggression)
        # Remaining stack available for aggression = current stack - cost to call current level
        remaining_stack_for_aggression = player_stack - max(0, amount_to_call)
        can_make_aggression = (self.raise_count_this_street < self.MAX_RAISES_PER_STREET)

        if remaining_stack_for_aggression > 0.01 and can_make_aggression:
             # Define min raise increment (based on last raise or BB) and max action (all-in)
             min_raise_inc = max(self.last_raise if self.last_raise > 0 else self.big_blind, self.big_blind)
             max_total_bet_possible = current_round_bet + player_stack # The amount if player goes all-in

             if current_bet_level < 0.01: # BET Scenario (no previous bet in round)
                 action_prefix = "bet"
                 # Min bet cost = BB or remaining stack if less
                 min_bet_cost = max(1.0, min(self.big_blind, player_stack))

                 # Add Min Bet (using COST) if possible
                 if player_stack >= min_bet_cost - 0.01:
                     actions.append((action_prefix, int(round(min_bet_cost))))

                 # Add All-in Bet (using COST = stack) if distinct from min bet
                 all_in_cost = player_stack
                 if all_in_cost > 0.01 and abs(all_in_cost - min_bet_cost) > 0.01:
                     # Check if already added by amount logic later
                     is_already_present = any(a[0]==action_prefix and abs(a[1]-all_in_cost)<0.01 for a in actions)
                     if not is_already_present:
                           actions.append((action_prefix, int(round(all_in_cost))))

             else: # RAISE Scenario (facing a bet/raise)
                 action_prefix = "raise"
                 # Min legal raise target = current level + min increment
                 min_legal_target_total = current_bet_level + min_raise_inc
                 # Cost to make min legal raise
                 min_raise_cost = min_legal_target_total - current_round_bet

                 # Add Min Legal Raise (using TARGET TOTAL amount) if possible
                 # Can afford if stack >= cost of min raise
                 if player_stack >= min_raise_cost - 0.01:
                      # Actual target might be capped by player going all-in
                      actual_min_legal_target = min(min_legal_target_total, max_total_bet_possible)
                      # Ensure target is actually higher than current bet
                      if actual_min_legal_target > current_bet_level + 0.01:
                           actions.append((action_prefix, int(round(actual_min_legal_target))))

                 # Add All-in Raise (using TARGET TOTAL amount = all-in amount)
                 all_in_target_total = max_total_bet_possible
                 # Check if all-in is a valid raise (>= min raise OR is a short all-in)
                 is_valid_all_in_raise = (all_in_target_total >= min_legal_target_total - 0.01) or \
                                         (player_stack < min_raise_cost - 0.01)
                 # Check if all-in target is distinct from min legal raise target already added
                 is_distinct_from_min = True
                 last_action_added = actions[-1] if actions else None
                 if last_action_added and last_action_added[0]==action_prefix and abs(last_action_added[1]-all_in_target_total)<0.01:
                       is_distinct_from_min = False

                 if is_valid_all_in_raise and all_in_target_total > current_bet_level + 0.01 and is_distinct_from_min:
                       actions.append((action_prefix, int(round(all_in_target_total))))


        # --- Filter duplicates and sort ---
        # Use dictionary to filter by (action_type, rounded_amount) key
        final_action_dict = {}
        for act, amt_float in actions:
             amount = max(0, int(round(float(amt_float))))
             action_key = (act, amount)
             if action_key in final_action_dict: continue

             # Sanity check affordability again based on final amount representation
             cost = 0.0
             if act == 'call': cost = amount # Amount for call IS the cost
             elif act == 'bet': cost = amount # Amount for bet IS the cost
             elif act == 'raise': cost = amount - current_round_bet # Cost is TARGET - current round bet

             # Allow small tolerance for floating point issues
             if cost <= player_stack + 0.01:
                 final_action_dict[action_key] = action_key

        # Define sorting order: fold < check < call < bet < raise, then by amount
        def sort_key(a):
            act_type, amount = a
            order = {"fold":0, "check":1, "call":2, "bet":3, "raise":4}
            return (order.get(act_type, 99), amount)

        return sorted(list(final_action_dict.values()), key=sort_key)

# --- In organized_poker_bot/game_engine/game_state.py ---

    def _is_betting_round_over(self):
        """ Checks if the current betting round has concluded. """
        if len(self.active_players) < 2:
            return True

        players_who_can_act = [
            p for p in self.active_players
            if 0 <= p < len(self.player_all_in) and not self.player_all_in[p] and \
               0 <= p < len(self.player_folded) and not self.player_folded[p] and \
               0 <= p < len(self.player_stacks) and self.player_stacks[p] > 0.01
        ]

        # If 0 players can act, betting is over.
        if len(players_who_can_act) == 0:
             return True

        # If 1 player can act, check specific conditions
        if len(players_who_can_act) == 1:
             p_idx = players_who_can_act[0]
             has_acted = p_idx in self.players_acted_this_round
             facing_bet = (self.current_bet - self.player_bets_in_round[p_idx]) > 0.01

             # Condition 1: Player is facing a bet they haven't acted on yet. Round is NOT over.
             if facing_bet and not has_acted:
                 # This covers the BB facing an initial push scenario, or any player facing a raise/bet before acting.
                 return False

             # Condition 2: Preflop BB Option (already handled in previous version, re-verify)
             is_preflop = (self.betting_round == self.PREFLOP)
             bb_player_idx = self._find_player_relative_to_dealer(2 if self.num_players > 2 else 1) # Find BB
             is_bb = (p_idx == bb_player_idx)
             bet_is_just_bb = abs(self.current_bet - self.big_blind) < 0.01
             last_aggressor_was_bb_or_none = (self.last_raiser == p_idx or self.last_raiser is None)

             if is_preflop and is_bb and bet_is_just_bb and not has_acted and last_aggressor_was_bb_or_none:
                 return False # BB still has option

             # Otherwise (not facing bet OR already acted), round IS over if only one can act.
             return True

        # Case 3: Multiple players can act. Check if all have matched current bet AND acted.
        all_matched_current_bet = all(
             abs(self.player_bets_in_round[p] - self.current_bet) < 0.01
             for p in players_who_can_act
        )
        all_have_acted = all(p in self.players_acted_this_round for p in players_who_can_act)

        return all_matched_current_bet and all_have_acted

# --- Inside game_state.py ---

    def _try_advance_round(self):
        """ Attempts to deal next street or end hand if betting round finished. """
        # Check if hand should already be over before proceeding
        if len(self.active_players) <= 1:
            if self.betting_round != self.HAND_OVER: self.betting_round = self.HAND_OVER
            self.current_player_idx = -1
            return

        # If all remaining active players are all-in, deal remaining board and go to Showdown/End
        # Crucial: Check if _check_all_active_are_allin() includes check for <=1 player who can act. Yes it does.
        # Crucial: Make sure betting_round < SHOWDOWN because we might enter here from RIVER betting round completion
        if self._check_all_active_are_allin() and self.betting_round < self.SHOWDOWN:
            # Temporarily store the round we entered this block from
            entry_round = self.betting_round
            # Deal remaining streets automatically if needed
            if entry_round <= self.PREFLOP and len(self.community_cards) < 3: self.deal_flop()
            if entry_round <= self.FLOP and len(self.community_cards) < 4: self.deal_turn()
            if entry_round <= self.TURN and len(self.community_cards) < 5: self.deal_river()

            # Now check again if board dealing failed / finished
            if self.betting_round == self.HAND_OVER: # Deal failed?
                pass # Keep HAND_OVER state
            elif len(self.community_cards) >= 5: # Board complete?
                self.betting_round = self.SHOWDOWN # Go to Showdown
            # If somehow board dealing ended early but hand not over, something is wrong, but SHOWDOWN/HAND_OVER is likely correct.

            self.current_player_idx = -1 # No more betting turns
            self.players_acted_this_round = set() # Clear acted set
            return # Finished fast-forwarding state

        # --- Normal round advancement logic (only if NOT all-in scenario handled above) ---
        current_round = self.betting_round
        round_advanced_successfully = False

        if current_round == self.PREFLOP:
            round_advanced_successfully = self.deal_flop()
        elif current_round == self.FLOP:
            round_advanced_successfully = self.deal_turn()
        elif current_round == self.TURN:
            round_advanced_successfully = self.deal_river()
        elif current_round == self.RIVER:
            # River betting completed normally -> Showdown
            self.betting_round = self.SHOWDOWN
            self.current_player_idx = -1
            self.players_acted_this_round = set()
            round_advanced_successfully = True
        # No further action needed if already in SHOWDOWN or HAND_OVER

        # If dealing failed or didn't happen for other reasons, ensure hand ends if appropriate
        if not round_advanced_successfully and self.betting_round < self.SHOWDOWN:
             if self.betting_round != self.HAND_OVER:
                  self.betting_round = self.HAND_OVER
             self.current_player_idx = -1
             self.players_acted_this_round = set()

    def is_terminal(self):
        """ Returns True if the hand is over (<=1 active player or Showdown/HandOver reached). """
        return len(self.active_players) <= 1 or self.betting_round >= self.SHOWDOWN

    def get_utility(self, player_idx, initial_stacks=None):
        """
        Calculates utility for a player in a terminal state relative to initial stacks.
        Note: Requires `initial_stacks` if called mid-game simulation (like in CFR).
              If called after `determine_winners`, stacks reflect final outcome,
              and comparison to starting stack shows profit/loss for the hand.
        """
        if not self.is_terminal():
             # print("WARN: get_utility called on non-terminal state. Returning 0.")
             return 0.0

        if initial_stacks is None:
             print("WARN: get_utility needs initial_stacks for meaningful comparison.")
             # Fallback: Return current stack? Might not be useful for CFR regret calc.
             if 0 <= player_idx < len(self.player_stacks): return self.player_stacks[player_idx]
             else: return 0.0

        if not (0 <= player_idx < self.num_players and
                player_idx < len(initial_stacks) and
                player_idx < len(self.player_stacks)):
            print(f"ERROR: get_utility index mismatch for P{player_idx}")
            return 0.0

        # Utility is the change in stack size
        return self.player_stacks[player_idx] - initial_stacks[player_idx]


    def determine_winners(self, player_names=None):
        """
        Determines winner(s) at showdown or end of hand, updates stacks, returns summary.
        Modifies self.player_stacks and self.pot.
        """
        if not self.is_terminal():
            print("WARN: determine_winners called on non-terminal state.")
            return []

        if not self.active_players and self.pot < 0.01:
             # Hand ended with no active players and no pot - nothing to award
             return []
        if not self.active_players and self.pot >= 0.01:
             # Pot exists but no players left - this shouldn't happen, log warning
             print(f"WARN: Pot {self.pot:.2f} exists but no active players at hand end.")
             self.pot = 0.0 # Pot is lost? Or refund logic needed? Clearing pot for now.
             return []

        total_pot_to_distribute = self.pot
        self.pot = 0.0 # Pot is now being distributed
        pots_summary = []
        players_in_showdown = self.active_players.copy() # Players involved at the end

        # --- Case 1: Uncontested Pot ---
        if len(players_in_showdown) == 1:
            winner_idx = players_in_showdown[0]
            amount_won = total_pot_to_distribute
            if 0 <= winner_idx < len(self.player_stacks):
                self.player_stacks[winner_idx] += amount_won
                winner_name = player_names[winner_idx] if player_names and winner_idx < len(player_names) else f"Player {winner_idx}"
                print(f"{winner_name} wins uncontested pot of {amount_won:.0f}")
                pots_summary = [{'winners': [winner_idx], 'amount': amount_won, 'eligible': [winner_idx], 'desc': 'Uncontested'}]
            else:
                print(f"ERROR: Uncontested winner index {winner_idx} out of range.")
            return pots_summary # Return summary

        # --- Case 2: Showdown ---
        evaluated_hands = {}
        valid_showdown_players = []
        print("\n--- Showdown ---")
        for p_idx in players_in_showdown:
            if p_idx >= len(self.hole_cards) or len(self.hole_cards[p_idx]) != 2:
                print(f" P{p_idx}: No hole cards for showdown.")
                continue # Skip player if no cards

            all_cards_for_eval = self.hole_cards[p_idx] + self.community_cards
            if len(all_cards_for_eval) < 5:
                 print(f" P{p_idx}: Not enough cards ({len(all_cards_for_eval)}) for hand evaluation.")
                 continue # Need 5+ cards

            try:
                hand_rank_tuple = HandEvaluator.evaluate_hand(all_cards_for_eval)
                evaluated_hands[p_idx] = hand_rank_tuple
                valid_showdown_players.append(p_idx)
                # Optionally print player hands at showdown
                hand_type_str, hand_desc_str = HandEvaluator.describe_hand(hand_rank_tuple) # Assumes describe_hand exists
                p_name = player_names[p_idx] if player_names and p_idx < len(player_names) else f"P{p_idx}"
                hole_str = ' '.join(map(str, self.hole_cards[p_idx]))
                print(f" {p_name} ({hole_str}): {hand_desc_str}")

            except Exception as e:
                print(f"ERROR evaluating hand for P{p_idx}: {e}")

        if not valid_showdown_players:
            print("WARN: Showdown occurred but no valid hands found to compare.")
            # Need logic for what happens to the pot here - split equally? Refund?
            # For now, return empty summary, pot remains 0.
            return []

        # --- Calculate Side Pots ---
        # 1. Get total contribution of each player eligible for showdown
        contributions = sorted(
             [(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players],
             key=lambda x: x[1] # Sort by amount contributed, lowest first
        )

        side_pots = [] # List of dicts: {'amount': float, 'eligible': list_of_player_indices}
        last_contribution_level = 0.0
        eligible_for_next_pot = valid_showdown_players[:] # Start with all showdown players

        for p_idx, total_contribution in contributions:
             contribution_increment = total_contribution - last_contribution_level
             if contribution_increment > 0.01:
                 # Create a pot for this increment, funded by players still eligible
                 num_eligible = len(eligible_for_next_pot)
                 pot_amount = contribution_increment * num_eligible
                 if pot_amount > 0.01:
                     side_pots.append({'amount': pot_amount, 'eligible': eligible_for_next_pot[:]}) # Store current eligible players
                 last_contribution_level = total_contribution # Update level

             # Player who defined this contribution level is no longer eligible for *further* pots
             if p_idx in eligible_for_next_pot:
                 eligible_for_next_pot.remove(p_idx)

        # Sanity check: Total amount in calculated side pots vs total_pot_to_distribute
        calculated_pot_total = sum(p['amount'] for p in side_pots)
        if abs(calculated_pot_total - total_pot_to_distribute) > 0.1 * self.num_players: # Allow small tolerance per player
            print(f"WARN: Side pot calculation discrepancy! Total Side Pots={calculated_pot_total:.2f}, Original Pot={total_pot_to_distribute:.2f}")
            # Attempt to correct? Difficult. Proceeding with calculated pots.

        # --- Award Side Pots ---
        distributed_total = 0
        pots_summary = [] # Rebuild summary from awarded pots
        print("--- Pot Distribution ---")
        for i, pot_info in enumerate(side_pots):
             pot_amount = pot_info['amount']
             eligible_players = pot_info['eligible']

             if pot_amount < 0.01 or not eligible_players: continue

             # Find best hand among eligible players for this pot
             eligible_hands = {p: evaluated_hands[p] for p in eligible_players if p in evaluated_hands}
             if not eligible_hands:
                  print(f"WARN: No eligible hands found for side pot {i+1}. Pot amount {pot_amount:.2f} unawarded?")
                  continue # Skip this pot if no one can win it

             best_hand_value = max(eligible_hands.values())
             pot_winners = [p for p, hand_val in eligible_hands.items() if hand_val == best_hand_value]

             if pot_winners:
                 winner_share = pot_amount / len(pot_winners)
                 distributed_total += pot_amount

                 # Update winner stacks
                 for w_idx in pot_winners:
                      if 0 <= w_idx < len(self.player_stacks):
                           self.player_stacks[w_idx] += winner_share

                 # Logging and Summary
                 winner_names_str = ', '.join(player_names[w] if player_names and w<len(player_names) else f"P{w}" for w in pot_winners)
                 pot_desc = f"Side Pot {i+1}" if len(side_pots)>1 else "Main Pot"
                 print(f" {pot_desc} ({pot_amount:.0f}) awarded to {winner_names_str} (Share: {winner_share:.0f})")
                 pots_summary.append({'winners':pot_winners, 'amount':pot_amount, 'eligible':eligible_players, 'desc': pot_desc})

             else:
                  print(f"WARN: No winners determined for side pot {i+1}. Pot amount {pot_amount:.2f} unawarded?")


        # Final check for undistributed pot remainders (usually due to rounding or complex edge cases)
        remainder = total_pot_to_distribute - distributed_total
        if abs(remainder) > 0.01 * self.num_players: # Allow small tolerance
            print(f"WARN: Pot distribution remainder: {remainder:.2f}. This amount is currently unawarded.")
            # Logic to distribute remainder needed? (e.g., to player left of dealer) - Complex.

        return pots_summary

    # --- Utility Methods ---
    def clone(self):
        """ Creates a deep copy of the game state. """
        new_state = deepcopy(self)
        # Ensure deck is also cloned properly if it has internal state beyond list
        if hasattr(self.deck, 'clone') and callable(self.deck.clone):
             new_state.deck = self.deck.clone()
        elif isinstance(self.deck, Deck): # Basic deck clone if no method
             new_state.deck = Deck()
             new_state.deck.cards = [card.clone() for card in self.deck.cards] # Ensure cards are cloned

        return new_state

    def get_position(self, player_idx):
        """ Returns player position relative to the dealer (0=Dealer, 1=SB, etc.). """
        if not (0 <= player_idx < self.num_players):
             return -1 # Invalid index
        if self.num_players <= 1:
             return 0 # Only one player, always position 0?

        dealer_pos = getattr(self, 'dealer_position', 0) % self.num_players
        # Calculate clockwise distance from dealer
        relative_pos = (player_idx - dealer_pos + self.num_players) % self.num_players
        return relative_pos

    # --- String Representation ---
    def __str__(self):
        """ Provides a human-readable summary of the current game state. """
        round_name = self.ROUND_NAMES.get(self.betting_round, f"Round {self.betting_round}")
        current_b = f"{self.current_bet:.0f}"
        last_r_size = f"{self.last_raise:.0f}"
        pot_size = f"{self.pot:.0f}"
        raises_this_street = str(self.raise_count_this_street)

        header = [
            f"-- State (Round: {round_name} | Bet: {current_b} | LastRaise: {last_r_size} | Raises: {raises_this_street}) Pot: {pot_size} --",
            f"Board: {' '.join(map(str, self.community_cards)) if self.community_cards else '(none)'}"
        ]

        turn_player = f"P{self.current_player_idx}" if self.current_player_idx is not None and self.current_player_idx != -1 else "None"
        last_aggressor = f"P{self.last_raiser}" if self.last_raiser is not None else "None"
        acted_in_round = f"{sorted(list(self.players_acted_this_round))}"
        header.append(f"Dealer: P{self.dealer_position}, Turn: {turn_player}, Last Aggressor: {last_aggressor}, Acted This Round: {acted_in_round}")

        player_lines = []
        for i in range(self.num_players):
            line = f" P{i}"
            try:
                stack_str = f"{self.player_stacks[i]:.0f}"
                pos_str = f"Pos{self.get_position(i)}"
                bet_round_str = f"{self.player_bets_in_round[i]:.0f}"
                bet_hand_str = f"{self.player_total_bets_in_hand[i]:.0f}"
                folded = self.player_folded[i]
                all_in = self.player_all_in[i]
                active_hand = i in self.active_players
                # Determine player status symbol
                status_symbol = "F" if folded else ("!" if all_in else ("*" if active_hand else "-"))

                # Get hole cards string safely
                cards_str = "-"
                if self.hole_cards and i < len(self.hole_cards) and self.hole_cards[i]:
                     cards_str = ' '.join(map(str, self.hole_cards[i]))

                line += f" {status_symbol} ({pos_str}): Stack={stack_str}, RoundBet={bet_round_str}, HandBet={bet_hand_str}, Cards=[{cards_str}]"
            except IndexError:
                 line += ": Error - Index Out of Bounds"
            except Exception as e:
                 line += f": Error Generating String - {e}"
            player_lines.append(line)

        history_str = self.get_betting_history()
        deck_count = len(self.deck) if self.deck else 0
        footer = [
            f"History: {history_str}",
            f"Deck Cards Remaining: {deck_count}",
            "-" * 60 # Increased separator width
        ]

        return "\n".join(header + player_lines + footer) + "\n"

# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
