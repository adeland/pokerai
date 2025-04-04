# --- START OF FILE organized_poker_bot/game_engine/game_state.py ---
"""
Game state implementation for poker games.
(Refactored V25: Removed debug prints from start_new_hand)
"""

# ... (Imports and class header V24/Corrected) ...
from organized_poker_bot.game_engine.deck import Deck
#... etc
import random, math, sys, os, traceback
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organized_poker_bot.game_engine.card import Card
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
from organized_poker_bot.game_engine.player import Player

class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {0:"Preflop", 1:"Flop", 2:"Turn", 3:"River", 4:"Showdown", 5:"Hand Over"}
    MAX_RAISES_PER_STREET = 7

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        # ... (init variables from V24/Corrected) ...
        if not (2 <= num_players <= 9): raise ValueError("Num players must be 2-9")
        self.num_players = num_players; self.small_blind = float(small_blind); self.big_blind = float(big_blind)
        self.player_stacks = [float(starting_stack)] * num_players; self.hole_cards = [[] for _ in range(num_players)]
        self.player_total_bets_in_hand = [0.0] * num_players; self.player_bets_in_round = [0.0] * num_players
        self.player_folded = [False] * num_players; self.player_all_in = [False] * num_players
        self.active_players = list(range(num_players)); self.community_cards = []; self.pot = 0.0
        self.betting_round = self.PREFLOP; self.deck = Deck(); self.dealer_position = 0
        self.current_player_idx = -1; self.current_bet = 0.0; self.last_raiser = None; self.last_raise = 0.0
        self.players_acted_this_round = set(); self.verbose_debug = False; self.raise_count_this_street = 0

    # --- Helpers (_get_next_active_player, _find_player_relative_to_dealer - V24/Corrected) ---
    def _get_next_active_player(self, start_idx):
        if not self.active_players or self.num_players == 0: return None
        valid_start_idx = start_idx if 0 <= start_idx < self.num_players else 0
        current_idx = (valid_start_idx + 1) % self.num_players; loop_count = 0
        while loop_count < self.num_players * 2:
            if current_idx in self.active_players:
                 if 0 <= current_idx < len(self.player_stacks) and self.player_stacks[current_idx] > 0.01: return current_idx
            current_idx = (current_idx + 1) % self.num_players
            if current_idx == (valid_start_idx + 1) % self.num_players and loop_count > self.num_players: break
            loop_count += 1
        if self.verbose_debug: print(f"WARN _get_next_active_player: No active player found from {start_idx}.")
        return None
    def _find_player_relative_to_dealer(self, offset):
        if not self.active_players or self.num_players == 0: return None
        current_dealer_pos = getattr(self, 'dealer_position', 0) % self.num_players
        loop_idx = (current_dealer_pos + offset) % self.num_players; loop_count = 0
        while loop_count < self.num_players * 2:
             if loop_idx in self.active_players:
                  if loop_idx < len(self.player_stacks) and self.player_stacks[loop_idx] > 0.01: return loop_idx
             loop_idx = (loop_idx + 1) % self.num_players
             if loop_idx == (current_dealer_pos + offset) % self.num_players and loop_count > self.num_players: break
             loop_count += 1
        return None

    # *** Modified start_new_hand (CLEANED DEBUG PRINTS) ***
    def start_new_hand(self, dealer_pos, player_stacks):
        self.hole_cards=[[] for _ in range(self.num_players)]; self.community_cards=[]; self.pot=0.0; self.betting_round=self.PREFLOP
        self.player_bets_in_round=[0.0]*self.num_players
        self.player_total_bets_in_hand=[0.0]*self.num_players; self.player_folded=[False]*self.num_players
        self.player_all_in=[False]*self.num_players; self.current_player_idx=-1; self.current_bet=0.0; self.last_raiser=None; self.last_raise=0.0
        self.players_acted_this_round=set(); self.dealer_position=dealer_pos % self.num_players; self.deck=Deck(); self.deck.shuffle()
        self.raise_count_this_street = 0

        # --- Correct assignment based on arg ---
        self.player_stacks=[float(s) for s in player_stacks];

        # Active players based on *newly assigned* stacks
        self.active_players=[i for i, stack in enumerate(self.player_stacks) if stack > 0.01]

        if len(self.active_players) < 2:
             self.betting_round = self.HAND_OVER;
             self.current_player_idx = -1 # Explicitly set -1 if hand can't start
             # Removed debug print
             return

        self._deal_hole_cards();
        self._post_blinds();
        self._start_betting_round(); # This sets current_player_idx

        # Removed final debug print
        if self.verbose_debug: print(f"DEBUG GS: Hand Started. D={self.dealer_position}. State:\n{self}")


    # --- (All other methods _deal_hole_cards through __str__ should use V24 corrected versions) ---
    def _deal_hole_cards(self): # Keep V21 corrected syntax
        if not self.active_players: return; start_deal_idx = -1
        potential_start_idx = self._find_player_relative_to_dealer(1);
        if potential_start_idx is not None: start_deal_idx = potential_start_idx
        else:
            idx_to_check = (self.dealer_position + 1) % self.num_players
            for _ in range(self.num_players):
                 if idx_to_check in self.active_players: start_deal_idx = idx_to_check; break
                 idx_to_check = (idx_to_check + 1) % self.num_players
            if start_deal_idx == -1:
                 if self.active_players: start_deal_idx = self.active_players[0]
                 else: print("ERROR: Cannot find start deal index."); self.betting_round=self.HAND_OVER; return
        for card_round in range(2):
            current_deal_idx=start_deal_idx; dealt_in_round=0; attempts=0
            while dealt_in_round<len(self.active_players) and attempts < self.num_players*2:
                attempts+=1;
                if not (0 <= current_deal_idx < self.num_players): current_deal_idx=(current_deal_idx+1)%self.num_players; continue
                if current_deal_idx in self.active_players:
                    while len(self.hole_cards) <= current_deal_idx: self.hole_cards.append([])
                    if len(self.hole_cards[current_deal_idx]) == card_round:
                        if len(self.deck)>0: self.hole_cards[current_deal_idx].append(self.deck.deal()); dealt_in_round+=1
                        else: print(f"ERR: Deck empty R{card_round+1}"); self.betting_round=self.HAND_OVER; return
                current_deal_idx=(current_deal_idx+1)%self.num_players
                if current_deal_idx==start_deal_idx and dealt_in_round<len(self.active_players) and attempts>self.num_players: print(f"WARN: Deal stuck R{card_round+1}"); break
            if attempts >= self.num_players * 2 and dealt_in_round < len(self.active_players): print(f"ERR: Failed deal R{card_round+1}"); self.betting_round=self.HAND_OVER; return
    def _post_blinds(self): # Keep V23 logic (Cleaned)
        if len(self.active_players) < 2: return; sb_idx=None; bb_idx=None;
        if self.num_players==2: sb_idx=self._find_player_relative_to_dealer(0); bb_idx=self._find_player_relative_to_dealer(1);
        else: sb_idx=self._find_player_relative_to_dealer(1); bb_idx=self._find_player_relative_to_dealer(2);
        posted_bb=0.0; self.raise_count_this_street = 0
        if sb_idx is not None and sb_idx < len(self.player_stacks):
            amt=min(self.small_blind, self.player_stacks[sb_idx]);
            if amt>0: self.player_stacks[sb_idx]-=amt; self.player_bets_in_round[sb_idx]=amt; self.player_total_bets_in_hand[sb_idx]+=amt; self.pot+=amt;
            if abs(self.player_stacks[sb_idx])<0.01 : self.player_all_in[sb_idx]=True
        else:
            if self.verbose_debug: print("WARN post_blinds: Could not find/post SB.")
        if bb_idx is not None and bb_idx < len(self.player_stacks):
            alr = self.player_bets_in_round[bb_idx] if bb_idx < len(self.player_bets_in_round) else 0.0
            need = self.big_blind - alr; amt = min(need, self.player_stacks[bb_idx]);
            if amt>0: self.player_stacks[bb_idx]-=amt; self.player_bets_in_round[bb_idx]+=amt; self.player_total_bets_in_hand[bb_idx]+=amt; self.pot+=amt;
            if abs(self.player_stacks[bb_idx])<0.01: self.player_all_in[bb_idx]=True
            posted_bb = self.player_bets_in_round[bb_idx]
        else:
            if self.verbose_debug: print("WARN post_blinds: Could not find/post BB.")
        self.current_bet = self.big_blind; self.last_raise = self.big_blind
        if posted_bb >= self.big_blind - 0.01: self.last_raiser = bb_idx; self.raise_count_this_street = 1;
        else: self.last_raiser = None; self.raise_count_this_street = 0
        if self.verbose_debug: print(f"DEBUG GS: Blinds Posted. SB={sb_idx}, BB={bb_idx}. Pot={self.pot}. CurrBet={self.current_bet}. LastRaiser={self.last_raiser}. Raises={self.raise_count_this_street}")
    def _start_betting_round(self): # Keep V24 logic (Corrected Loop Indentation)
        self.players_acted_this_round = set(); first_player = None
        if self.betting_round != self.PREFLOP:
            self.current_bet=0.0; self.last_raiser=None; self.last_raise=self.big_blind; self.raise_count_this_street = 0;
            for i in range(self.num_players):
                if i < len(self.player_bets_in_round): self.player_bets_in_round[i]=0.0
                else:
                    if self.verbose_debug: print(f"WARN _start_round: Index {i} OOB reset bets.")
        if self.betting_round == self.PREFLOP:
             if self.num_players == 2: first_player = self._find_player_relative_to_dealer(0)
             else: bb = self._find_player_relative_to_dealer(2); start_search_idx = bb if bb is not None else self._find_player_relative_to_dealer(1); first_player = self._get_next_active_player(start_search_idx) if start_search_idx is not None else None;
             if first_player is None: first_player = self._get_next_active_player(self.dealer_position)
        else: first_player = self._get_next_active_player(self.dealer_position)
        self.current_player_idx = first_player if first_player is not None else -1
        active_with_stack = [p for p in self.active_players if p < len(self.player_stacks) and self.player_stacks[p] > 0.01]
        if len(active_with_stack) <= 1: self.current_player_idx = -1;
        elif self._check_all_active_are_allin(): self.current_player_idx = -1
        round_name = self.ROUND_NAMES.get(self.betting_round, '?');
        if self.verbose_debug: print(f"DEBUG GS: Start Rnd {round_name} Ready. Actor: P{self.current_player_idx}. Raises={self.raise_count_this_street}")
    def deal_flop(self): # V24 correct logic
        if self.verbose_debug: print("DEBUG GS: Try Deal Flop...")
        if self.betting_round!=self.PREFLOP:
             if self.verbose_debug: print(f"DEBUG GS: Cannot deal flop, not in PREFLOP");
             return False
        if len(self.active_players) <= 1:
             if self.verbose_debug: print(f"DEBUG GS: Cannot deal flop, <=1 active player.");
             self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
            if self.verbose_debug: print("DEBUG GS: All active are all-in pre-flop, dealing board.");
            if len(self.community_cards)<3:
                if len(self.deck) <= 3: self.betting_round=self.HAND_OVER; return False
                if len(self.deck)>0: self.deck.deal();
                else: self.betting_round=self.HAND_OVER; return False
                for _ in range(3):
                    if len(self.deck)>0: self.community_cards.append(self.deck.deal());
                    else: self.betting_round=self.HAND_OVER; return False
            self.betting_round = self.FLOP; return self.deal_turn()
        if len(self.deck) <= 3: self.betting_round=self.HAND_OVER; return False
        if len(self.deck)>0: self.deck.deal();
        else: self.betting_round=self.HAND_OVER; return False
        for _ in range(3):
             if len(self.deck)>0: self.community_cards.append(self.deck.deal());
             else: self.betting_round=self.HAND_OVER; return False
        self.betting_round = self.FLOP; self._start_betting_round(); return True
    def deal_turn(self): # V24 correct logic
        if self.verbose_debug: print("DEBUG GS: Try Deal Turn...")
        if self.betting_round!=self.FLOP:
             if self.verbose_debug: print(f"DEBUG GS: Cannot deal turn, not in FLOP");
             return False
        if len(self.active_players)<=1:
             if self.verbose_debug: print(f"DEBUG GS: Cannot deal turn, <=1 active player.");
             self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
             if self.verbose_debug: print("DEBUG GS: All active are all-in pre-turn, dealing card.")
             if len(self.community_cards)<4:
                 if len(self.deck) <= 1: self.betting_round=self.HAND_OVER; return False
                 if len(self.deck)>0: self.deck.deal();
                 else: self.betting_round=self.HAND_OVER; return False
                 if len(self.deck)>0: self.community_cards.append(self.deck.deal());
                 else: self.betting_round=self.HAND_OVER; return False
             self.betting_round = self.TURN; return self.deal_river()
        if len(self.deck) <= 1: self.betting_round=self.HAND_OVER; return False
        if len(self.deck)>0: self.deck.deal();
        else: self.betting_round=self.HAND_OVER; return False
        if len(self.deck)>0: self.community_cards.append(self.deck.deal());
        else: self.betting_round=self.HAND_OVER; return False
        self.betting_round = self.TURN; self._start_betting_round(); return True
    def deal_river(self): # V24 correct logic
        if self.verbose_debug: print("DEBUG GS: Try Deal River...")
        if self.betting_round!=self.TURN:
            if self.verbose_debug: print(f"DEBUG GS: Cannot deal river, not in TURN");
            return False
        if len(self.active_players)<=1:
            if self.verbose_debug: print(f"DEBUG GS: Cannot deal river, <=1 active player.");
            self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
             if self.verbose_debug: print("DEBUG GS: All active are all-in pre-river, dealing card.")
             if len(self.community_cards)<5:
                 if len(self.deck) <= 1: self.betting_round=self.HAND_OVER; return False
                 if len(self.deck)>0: self.deck.deal();
                 else: self.betting_round=self.HAND_OVER; return False
                 if len(self.deck)>0: self.community_cards.append(self.deck.deal());
                 else: self.betting_round=self.HAND_OVER; return False
             self.betting_round = self.SHOWDOWN; self.current_player_idx=-1; self.players_acted_this_round=set(); return True
        if len(self.deck) <= 1: self.betting_round=self.HAND_OVER; return False
        if len(self.deck)>0: self.deck.deal();
        else: self.betting_round=self.HAND_OVER; return False
        if len(self.deck)>0: self.community_cards.append(self.deck.deal());
        else: self.betting_round=self.HAND_OVER; return False

    def _check_all_active_are_allin(self):
        if not self.active_players:
            return True # No active players = effectively "all-in" / out

        num_can_act_voluntarily = 0

        for p_idx in self.active_players:
             try:
                 # Check player exists in lists before accessing status
                 if p_idx < len(self.player_folded) and \
                    p_idx < len(self.player_all_in) and \
                    p_idx < len(self.player_stacks):

                     # Check conditions for being able to act voluntarily
                     if not self.player_folded[p_idx] and \
                        not self.player_all_in[p_idx] and \
                        self.player_stacks[p_idx] > 0.01:
                            num_can_act_voluntarily += 1
                            if num_can_act_voluntarily > 1:
                                 return False # More than one can act
                 else:
                     # Index out of bounds is an error state, log and skip
                     if self.verbose_debug: print(f"WARN _check_all_active_allin: Index {p_idx} OOB checking lists.")
                     continue # Skip this player index

             # --- CORRECTED Indentation for Except block ---
             except IndexError:
                 # Catch potential errors just in case during list access
                 if self.verbose_debug:
                      print(f"WARN _check_all_active_allin: IndexError checking P{p_idx}.")
                 continue # Skip this player on error
             # --- END CORRECTION ---

        # Loop finished: means 0 or 1 players could act voluntarily.
        return True

    def _move_to_next_player(self): # V24 logic
        if self.current_player_idx == -1: return
        start_search_idx = self.current_player_idx; next_player_idx = self._get_next_active_player(start_search_idx)
        if next_player_idx == start_search_idx: pass
        self.current_player_idx = next_player_idx if next_player_idx is not None else -1
    def apply_action(self, action): # V24 logic
        if not isinstance(action, tuple) or len(action)!=2: raise ValueError(f"Invalid action:{action}.");
        action_type, amount_input = action; amount = float(amount_input); player_idx = self.current_player_idx
        if player_idx==-1: raise ValueError("No current player.");
        if not (0 <= player_idx < self.num_players): raise ValueError(f"Invalid P idx {player_idx}")
        if player_idx >= len(self.player_folded) or player_idx >= len(self.player_all_in) or player_idx >= len(self.player_stacks) or player_idx >= len(self.player_bets_in_round): raise ValueError(f"P idx {player_idx} OOB Lists")
        is_active=player_idx in self.active_players; is_folded=self.player_folded[player_idx]; is_all_in=self.player_all_in[player_idx];
        if not is_active: raise ValueError(f"P{player_idx} not active in hand.");
        if is_folded: raise ValueError(f"P{player_idx} already folded.");
        if is_all_in: raise ValueError(f"P{player_idx} all-in, cannot act.");
        new_state = self.clone()
        try: new_state._apply_action_logic(player_idx, action_type, amount)
        except ValueError as e:
             if self.verbose_debug: print(f"ERROR GS apply P{player_idx} {action}: {e}\nState BEFORE action:\n{self}")
             raise
        round_over = new_state._is_betting_round_over()
        if round_over: new_state._try_advance_round()
        else:
            new_state._move_to_next_player()
            round_over_after_move = new_state._is_betting_round_over()
            if new_state.current_player_idx != -1 and round_over_after_move: new_state._try_advance_round()
        return new_state
    def _apply_action_logic(self, player_idx, action_type, amount): # V24 logic
        player_stack=self.player_stacks[player_idx]; current_round_bet=self.player_bets_in_round[player_idx]; self.players_acted_this_round.add(player_idx)
        if action_type=="fold": self.player_folded[player_idx]=True; self.active_players.remove(player_idx) if player_idx in self.active_players else None;
        if len(self.active_players) <= 1: self.betting_round = self.HAND_OVER; self.current_player_idx = -1;
        elif action_type=="check":
             bet_to_call = self.current_bet - current_round_bet;
             if bet_to_call > 0.01: raise ValueError(f"Check invalid: Call {bet_to_call} needed")
        elif action_type=="call":
            bet_to_call = self.current_bet - current_round_bet;
            if bet_to_call <= 0.01: return
            actual_call = min(bet_to_call, player_stack);
            self.player_stacks[player_idx]-=actual_call; self.player_bets_in_round[player_idx]+=actual_call; self.player_total_bets_in_hand[player_idx]+=actual_call; self.pot+=actual_call
            if abs(self.player_stacks[player_idx]) < 0.01: self.player_all_in[player_idx] = True
        elif action_type=="bet":
            if self.current_bet > 0.01: raise ValueError("Bet invalid: Must raise facing bet.");
            if amount < 0.01: raise ValueError("Bet must be positive.");
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET: raise ValueError(f"Bet invalid: Max raises ({self.MAX_RAISES_PER_STREET}) reached.")
            min_bet=max(self.big_blind,1.0); actual_bet_amount=min(amount, player_stack); is_all_in_for_less = abs(actual_bet_amount - player_stack) < 0.01 and actual_bet_amount < min_bet
            if actual_bet_amount < min_bet - 0.01 and not is_all_in_for_less: raise ValueError(f"Bet {actual_bet_amount} < min {min_bet}");
            self.player_stacks[player_idx]-=actual_bet_amount; self.player_bets_in_round[player_idx]+=actual_bet_amount; self.player_total_bets_in_hand[player_idx]+=actual_bet_amount; self.pot+=actual_bet_amount
            new_total_bet_this_round = self.player_bets_in_round[player_idx]; self.current_bet=new_total_bet_this_round;
            self.last_raise=new_total_bet_this_round; self.last_raiser=player_idx; self.raise_count_this_street = 1;
            if abs(self.player_stacks[player_idx]) < 0.01: self.player_all_in[player_idx] = True
            self.players_acted_this_round={player_idx}
        elif action_type=="raise":
            if self.current_bet <= 0.01: raise ValueError("Raise invalid: Must bet.");
            if self.raise_count_this_street >= self.MAX_RAISES_PER_STREET: raise ValueError(f"Raise invalid: Max raises ({self.MAX_RAISES_PER_STREET}) reached.")
            total_bet_intended = amount; raise_increase = total_bet_intended - current_round_bet;
            if raise_increase <= 0.01: raise ValueError(f"Raise must increase bet");
            if raise_increase > player_stack + 0.01: raise ValueError(f"Raise increase > stack");
            min_raise_inc = max(self.last_raise if self.last_raise > 0 else self.big_blind, self.big_blind); min_legal_raise_to = self.current_bet + min_raise_inc; actual_increase_needed=min(raise_increase, player_stack); actual_total_bet = current_round_bet + actual_increase_needed; is_all_in = abs(actual_increase_needed - player_stack) < 0.01
            if actual_total_bet < min_legal_raise_to - 0.01 and not is_all_in: raise ValueError(f"Raise to {actual_total_bet:.2f} < min {min_legal_raise_to:.2f}");
            self.player_stacks[player_idx]-=actual_increase_needed; self.player_bets_in_round[player_idx]+=actual_increase_needed; self.player_total_bets_in_hand[player_idx]+=actual_increase_needed; self.pot+=actual_increase_needed;
            new_bet_level = self.player_bets_in_round[player_idx]; self.last_raise = new_bet_level - self.current_bet; self.current_bet = new_bet_level; self.last_raiser = player_idx; self.raise_count_this_street += 1;
            if is_all_in: self.player_all_in[player_idx] = True
            self.players_acted_this_round = {player_idx}
        else: raise ValueError(f"Unknown action: {action_type}")
    def get_available_actions(self): # V24 logic
        actions = []; player_idx = self.current_player_idx
        if player_idx == -1: return []
        try:
            if player_idx >= len(self.player_folded) or self.player_folded[player_idx] or \
               player_idx >= len(self.player_all_in) or self.player_all_in[player_idx]: return []
        except IndexError: return []
        player_stack = self.player_stacks[player_idx]; player_bet_this_round = self.player_bets_in_round[player_idx]; current_bet_level = self.current_bet
        actions.append(("fold", 0))
        bet_to_call = current_bet_level - player_bet_this_round; can_check = bet_to_call < 0.01
        if can_check: actions.append(("check", 0))
        else:
            call_amount = min(bet_to_call, player_stack)
            if call_amount > 0.01: actions.append(("call", int(round(call_amount))))
        effective_stack_if_call = player_stack - min(bet_to_call, player_stack) if bet_to_call > 0 else player_stack
        can_make_aggressive_action = (self.raise_count_this_street < self.MAX_RAISES_PER_STREET)
        if effective_stack_if_call > 0.01 and can_make_aggressive_action:
            min_raise_inc = max(self.last_raise if self.last_raise > 0 else self.big_blind, self.big_blind)
            if current_bet_level < 0.01: # Can BET
                 prefix="bet"; min_bet_amount = max(1.0, min(self.big_blind, player_stack))
                 if player_stack >= min_bet_amount - 0.01: actions.append((prefix, int(round(min_bet_amount))))
                 pot_bet_amount = min(player_stack, self.pot); pot_bet_amount = max(min_bet_amount, pot_bet_amount)
                 if abs(pot_bet_amount - min_bet_amount) > 0.01 and pot_bet_amount < player_stack - 0.01: actions.append((prefix, int(round(pot_bet_amount))))
                 all_in_amount_this_action = player_stack
                 if all_in_amount_this_action > 0.01:
                     already = any(abs(a[1] - all_in_amount_this_action) < 0.01 and a[0] == prefix for a in actions)
                     if not already: actions.append((prefix, int(round(all_in_amount_this_action))))
            else: # Can RAISE
                 prefix = "raise"; min_legal_raise_to=current_bet_level+min_raise_inc; max_possible_total_bet = player_bet_this_round + player_stack
                 min_raise_increase_needed = min_legal_raise_to - player_bet_this_round
                 if player_stack >= min_raise_increase_needed - 0.01:
                      actual_min_raise_to_amount = min(min_legal_raise_to, max_possible_total_bet)
                      if actual_min_raise_to_amount - player_bet_this_round <= player_stack + 0.01: actions.append((prefix, int(round(actual_min_raise_to_amount))))
                      call_amount_raise = min(bet_to_call, player_stack); pot_after_call = self.pot + call_amount_raise;
                      desired_pot_raise_increase = pot_after_call; desired_pot_raise_total = current_bet_level + desired_pot_raise_increase;
                      actual_pot_raise_total = max(actual_min_raise_to_amount, desired_pot_raise_total); actual_pot_raise_total = min(actual_pot_raise_total, max_possible_total_bet)
                      if abs(actual_pot_raise_total - actual_min_raise_to_amount) > 0.01 and actual_pot_raise_total < max_possible_total_bet - 0.01 and actual_pot_raise_total - player_bet_this_round <= player_stack + 0.01: actions.append((prefix, int(round(actual_pot_raise_total))))
                 all_in_total_bet = max_possible_total_bet
                 is_valid_all_in = (all_in_total_bet >= min_legal_raise_to - 0.01) or (player_stack < min_raise_increase_needed - 0.01)
                 if is_valid_all_in and all_in_total_bet > current_bet_level + 0.01:
                     already = any(abs(a[1] - all_in_total_bet) < 0.01 and a[0] == prefix for a in actions)
                     if not already: actions.append((prefix, int(round(all_in_total_bet))))
        final = {}; cost=0.0
        for act, amt_f in actions:
            amt = max(0, int(round(float(amt_f)))); key = (act, amt);
            if key in final: continue
            cost = 0.0; local_bet_to_call = current_bet_level - player_bet_this_round
            if act=='call': cost=min(max(0, local_bet_to_call), player_stack)
            elif act=='bet': cost=amt
            elif act=='raise': cost=amt - player_bet_this_round
            if cost <= player_stack + 0.01: final[key] = key
            elif self.verbose_debug: print(f"WARN get_actions: Action {key} removed, cost {cost:.2f} > stack {player_stack:.2f}")
        def sort_key(a): t,amt=a; o={"fold":0,"check":1,"call":2,"bet":3,"raise":4}; return (o.get(t,99), amt)
        return sorted(list(final.values()), key=sort_key)
    def _is_betting_round_over(self): # Keep V24 logic
        active_can_act_voluntarily = [ p for p in self.active_players if p < len(self.player_all_in) and not self.player_all_in[p] and p < len(self.player_stacks) and self.player_stacks[p] > 0.01 and p < len(self.player_folded) and not self.player_folded[p] ]
        if len(active_can_act_voluntarily) < 2: return True
        significant_action_occurred = self.current_bet > (self.big_blind if self.betting_round == self.PREFLOP else 0.01); all_eligible_acted_at_least_once = True; bets_match = True
        for p_idx in self.active_players:
             if p_idx >= len(self.player_folded) or self.player_folded[p_idx]: continue
             if p_idx >= len(self.player_all_in) or self.player_all_in[p_idx]: continue
             if p_idx >= len(self.player_bets_in_round) or abs(self.player_bets_in_round[p_idx] - self.current_bet) > 0.01: bets_match = False
             if p_idx not in self.players_acted_this_round:
                 is_bb = (self._find_player_relative_to_dealer(2 if self.num_players > 2 else 1) == p_idx); initial_bb_only = (abs(self.current_bet - self.big_blind) < 0.01) and (self.last_raiser == p_idx or self.last_raiser is None)
                 if self.betting_round == self.PREFLOP and is_bb and initial_bb_only: all_eligible_acted_at_least_once = False
                 else: all_eligible_acted_at_least_once = False
        if bets_match and all_eligible_acted_at_least_once:
             if self.betting_round > self.PREFLOP and not significant_action_occurred: return True
             if significant_action_occurred: return True
             if self.betting_round == self.PREFLOP:
                  is_bb_check = (self._find_player_relative_to_dealer(2 if self.num_players > 2 else 1) == self.current_player_idx); initial_bb_only_check = (abs(self.current_bet - self.big_blind) < 0.01) and (self.last_raiser == self.current_player_idx or self.last_raiser is None)
                  if is_bb_check and initial_bb_only_check: return False
                  else: return True
        return False
    def _try_advance_round(self): # Keep V24 logic
        if self.verbose_debug: print(f"DEBUG GS _try_advance_round (Current: {self.betting_round})");
        if len(self.active_players)<=1:
            if self.betting_round < self.HAND_OVER: self.betting_round=self.HAND_OVER
            self.current_player_idx = -1; self.players_acted_this_round = set(); return
        if self._check_all_active_are_allin() and self.betting_round < self.RIVER:
            if self.verbose_debug: print(f"DEBUG GS: Advancing round via all-in. Current round {self.betting_round}")
            current_round_before_dealing = self.betting_round
            if current_round_before_dealing == self.PREFLOP: self.deal_flop()
            elif current_round_before_dealing == self.FLOP: self.deal_turn()
            elif current_round_before_dealing == self.TURN: self.deal_river()
            if len(self.community_cards) >= 5 and self.betting_round < self.SHOWDOWN: self.betting_round = self.SHOWDOWN
            self.current_player_idx = -1; self.players_acted_this_round = set(); return
        rnd=self.betting_round; successful_deal = False; new_rnd_name="?"
        if rnd==self.PREFLOP: successful_deal=self.deal_flop(); new_rnd_name="Flop"
        elif rnd==self.FLOP: successful_deal=self.deal_turn(); new_rnd_name="Turn"
        elif rnd==self.TURN: successful_deal=self.deal_river(); new_rnd_name="River"
        elif rnd==self.RIVER: self.betting_round=self.SHOWDOWN; self.current_player_idx = -1; self.players_acted_this_round = set(); successful_deal = True; new_rnd_name="Showdown";
        if not successful_deal and self.betting_round < self.SHOWDOWN :
            if self.verbose_debug: print(f"DEBUG GS: Dealing {new_rnd_name} failed. Setting Hand Over.")
            if self.betting_round != self.HAND_OVER: self.betting_round=self.HAND_OVER
            self.current_player_idx = -1; self.players_acted_this_round = set();
    def is_terminal(self): return len(self.active_players)<=1 or self.betting_round>=self.SHOWDOWN or self.betting_round == self.HAND_OVER
    def get_utility(self, player_idx): return 0.0 # Placeholder
    def determine_winners(self, player_names=None): # V24 logic (correct syntax)
        if not self.is_terminal():
             if self._is_betting_round_over(): self._try_advance_round()
             if not self.is_terminal(): print("WARN: determine_winners called on non-terminal state."); return []
        if not self.active_players and self.pot < 0.01 : return []
        elif not self.active_players and self.pot > 0.01:
             if self.verbose_debug: print(f"WARN dw: Pot {self.pot} but no active players.");
             self.pot = 0; return []
        total_pot=self.pot; pots_info=[]; showdown_player_indices = self.active_players.copy()
        if len(showdown_player_indices) == 1:
            winner_idx = showdown_player_indices[0]; won = total_pot
            if winner_idx < len(self.player_stacks): self.player_stacks[winner_idx] += won
            else: print(f"ERROR: Uncontested winner index {winner_idx} out of stack range.")
            self.pot = 0; pots_info.append(({'winners': [winner_idx], 'amount': won, 'eligible': [winner_idx]})); return pots_info
        hands={}; valid_showdown_players=[];
        for p in showdown_player_indices:
            if p >= len(self.hole_cards) or len(self.hole_cards[p]) != 2: continue
            all_c = self.hole_cards[p] + self.community_cards;
            if len(all_c)<5: continue
            try: hands[p]=HandEvaluator.evaluate_hand(all_c); valid_showdown_players.append(p);
            except Exception as e: print(f"ERR hand eval P{p}:{e}")
        if not hands or not valid_showdown_players:
             if self.verbose_debug: print(f"WARN dw: No valid hands at showdown. Pot {total_pot} remains?.");
             self.pot=0; return[]
        player_contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_showdown_players], key=lambda x: x[1])
        created_pots=[]; last_bet_level = 0.0; current_eligible_for_pot = valid_showdown_players[:]
        for player_idx, player_total_bet in player_contributions:
             bet_increase_at_this_level = player_total_bet - last_bet_level
             if bet_increase_at_this_level > 0.01:
                  num_contributing = len(current_eligible_for_pot); pot_amount_at_this_level = bet_increase_at_this_level * num_contributing
                  if pot_amount_at_this_level > 0.01: created_pots.append({'amount': pot_amount_at_this_level, 'eligible': current_eligible_for_pot[:]})
                  last_bet_level = player_total_bet
             if player_idx in current_eligible_for_pot: current_eligible_for_pot.remove(player_idx)
        distributed_total=0; pots_summary = []
        for pot_data in created_pots:
            pot_amt = pot_data['amount']; eligible_players = pot_data['eligible'];
            if pot_amt < 0.01: continue
            eligible_hands = {p: hands[p] for p in eligible_players if p in hands};
            if not eligible_hands: continue
            best_hand_value = max(eligible_hands.values()); winners = [p for p, val in eligible_hands.items() if val == best_hand_value];
            if winners:
                share = pot_amt / len(winners);
                if self.verbose_debug: print(f"DEBUG winners: Pot {pot_amt:.0f} elig {eligible_players}, Winners: {winners} (Share: {share:.0f})")
                for w in winners:
                     if w < len(self.player_stacks): self.player_stacks[w]+=share
                     else: print(f"ERROR: Side pot winner index {w} out of stack range.")
                distributed_total += pot_amt; pots_summary.append({'winners': winners, 'amount': pot_amt, 'eligible': eligible_players})
            else:
                if self.verbose_debug: print(f"WARN dw: No winners for pot {pot_amt}, elig {eligible_players}?")
        remaining_pot = total_pot - distributed_total
        if abs(remaining_pot) > 0.01:
            if self.verbose_debug: print(f"WARN dw: Pot discrepancy {remaining_pot:.2f}.")
            if pots_summary: last_pot = pots_summary[-1]; last_winners = last_pot['winners'];
            else: last_winners = []
            if last_winners: share = remaining_pot / len(last_winners)
            else:
                 share = 0
                 if self.verbose_debug: print(f"WARN dw: Could not distribute discrepancy {remaining_pot}")
            for w in last_winners:
                 if w < len(self.player_stacks): self.player_stacks[w]+=share
            if pots_summary:
                if last_winners: last_pot['amount'] += remaining_pot
            self.pot = 0
        else: self.pot = 0
        return pots_summary
    def clone(self): # V24 logic
        new=GameState(self.num_players,0,self.small_blind,self.big_blind); new.num_players=self.num_players; new.pot=self.pot; new.current_player_idx=self.current_player_idx; new.betting_round=self.betting_round; new.current_bet=self.current_bet; new.last_raiser=self.last_raiser; new.last_raise=self.last_raise; new.dealer_position=self.dealer_position; new.small_blind=self.small_blind; new.big_blind=self.big_blind; new.player_stacks=self.player_stacks[:]; new.hole_cards=[c[:] if isinstance(c, list) else [] for c in self.hole_cards]; new.community_cards=self.community_cards[:]; new.player_total_bets_in_hand=self.player_total_bets_in_hand[:]; new.player_bets_in_round=self.player_bets_in_round[:]; new.player_folded=self.player_folded[:]; new.player_all_in=self.player_all_in[:]; new.active_players=self.active_players[:]; new.players_acted_this_round=self.players_acted_this_round.copy(); new.deck=self.deck.clone(); new.verbose_debug = self.verbose_debug; new.raise_count_this_street = self.raise_count_this_street;
        return new
    def get_betting_history(self): # V24 logic
        pot_bbs = int(round(self.pot/self.big_blind)) if self.big_blind > 0 else 0; cb_bbs = int(round(self.current_bet/self.big_blind)) if self.big_blind > 0 else 0; num_active = len(self.active_players); actor_idx = self.current_player_idx; bets_tuple = tuple(int(b) for b in self.player_bets_in_round); bets_hash = hash(bets_tuple); rc = self.raise_count_this_street
        return f"R{self.betting_round}|P{pot_bbs}|CB{cb_bbs}|N{num_active}|Act{actor_idx}|RC{rc}|BH{bets_hash}"
    def get_position(self, player_idx): # V24 corrected logic
        if not (0 <= player_idx < self.num_players): return -1
        if self.num_players <= 1: return 0
        dealer_pos_attr = getattr(self, 'dealer_position', 0)
        if not (0 <= dealer_pos_attr < self.num_players): dealer_pos_attr = 0
        dealer_pos = dealer_pos_attr % self.num_players
        relative_position = (player_idx - dealer_pos + self.num_players) % self.num_players
        return relative_position
    def __str__(self): # V24 logic
        rnd=self.ROUND_NAMES.get(self.betting_round, f"Unk({self.betting_round})"); cb_str = f"{self.current_bet:.0f}" if self.current_bet is not None else "N/A"; lr_str = f"{self.last_raise:.0f}" if self.last_raise is not None else "N/A"; pot_str = f"{self.pot:.0f}" if self.pot is not None else "N/A";
        s = f"-- State (Rnd:{rnd}|Bet:{cb_str}|LastRaise:{lr_str}|Raises:{self.raise_count_this_street}) Pot:{pot_str} --\n"; board_str = ' '.join(str(c) for c in self.community_cards) if self.community_cards else "(none)"; s += f"Board: {board_str}\n"; turn_str = f"{self.current_player_idx}" if self.current_player_idx is not None else "N/A"; last_raiser_str = f"{self.last_raiser}" if self.last_raiser is not None else "None"; acted_str = f"{sorted(list(self.players_acted_this_round))}" if self.players_acted_this_round is not None else "{}"; s += f"D:{self.dealer_position}, Turn:{turn_str}, LastRaiser:{last_raiser_str}, Acted:{acted_str}\n"
        for i in range(self.num_players):
            stk_val="ERR";pos_val="ERR";rB_val="ERR";hB_val="ERR";fld_val=False;ain_val=False;act_val=False;cards_str="--"
            try:
                if i < len(self.player_stacks): stk_val = f"{self.player_stacks[i]:.0f}"
                pos_val = f"Pos{self.get_position(i)}";
                if i < len(self.player_bets_in_round): rB_val = f"{self.player_bets_in_round[i]:.0f}"
                if i < len(self.player_total_bets_in_hand): hB_val = f"{self.player_total_bets_in_hand[i]:.0f}"
                if i < len(self.player_folded): fld_val = self.player_folded[i]
                if i < len(self.player_all_in): ain_val = self.player_all_in[i]
                act_val = i in self.active_players if self.active_players is not None else False
                if i < len(self.hole_cards) and isinstance(self.hole_cards[i], list) and self.hole_cards[i]: cards_str = ' '.join(str(c) for c in self.hole_cards[i])
                stat_char = ("F" if fld_val else "!" if ain_val else "*" if act_val else "-"); s += f"  P{i}{stat_char} {pos_val}: S={stk_val}, RndB={rB_val}, HndB={hB_val}, Cards=[{cards_str}]\n"
            except Exception as e: s += f"  P{i}: Error generating string - {e}\n"
        deck_len = len(self.deck) if self.deck else 0; s += f"Deck:{deck_len} cards\n"+"-"*20+"\n"; return s

# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
