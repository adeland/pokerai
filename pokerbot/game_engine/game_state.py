# --- START OF FILE organized_poker_bot/game_engine/game_state.py ---
"""
Game state implementation for poker games.
(Refactored V13: Added extensive DEBUG prints for tracing)
"""

import random
from collections import defaultdict
import os
import sys
import traceback
import math

# Add the parent directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute imports
from organized_poker_bot.game_engine.deck import Deck
from organized_poker_bot.game_engine.card import Card
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator
from organized_poker_bot.game_engine.player import Player

class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {0:"Preflop", 1:"Flop", 2:"Turn", 3:"River", 4:"Showdown", 5:"Hand Over"}

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        if not (2 <= num_players <= 9): raise ValueError("Num players must be 2-9")
        self.num_players = num_players; self.small_blind = float(small_blind); self.big_blind = float(big_blind)
        self.player_stacks = [float(starting_stack)] * num_players; self.hole_cards = [[] for _ in range(num_players)]
        self.player_total_bets_in_hand = [0.0] * num_players; self.player_bets_in_round = [0.0] * num_players
        self.player_folded = [False] * num_players; self.player_all_in = [False] * num_players
        self.active_players = list(range(num_players)); self.community_cards = []; self.pot = 0.0
        self.betting_round = self.PREFLOP; self.deck = Deck(); self.dealer_position = 0
        self.current_player_idx = -1; self.current_bet = 0.0; self.last_raiser = None; self.last_raise = 0.0
        self.players_acted_this_round = set()
        # --- DEBUGGING FLAG ---
        self.verbose_debug = False # Set to True manually or via runner for detailed logs

    # --- Helper Methods ---
    def _get_next_active_player(self, start_idx):
        if not self.active_players or self.num_players == 0: return None
        current_idx = (start_idx + 1) % self.num_players
        for _ in range(self.num_players + 1):
            if 0 <= current_idx < self.num_players:
                is_active = current_idx in self.active_players
                has_stack = current_idx < len(self.player_stacks) and self.player_stacks[current_idx] > 0.01
                if is_active and has_stack: return current_idx
            current_idx = (current_idx + 1) % self.num_players
        return None
    def _find_player_relative_to_dealer(self, offset):
        if not self.active_players or self.num_players == 0: return None
        current_dealer_pos = getattr(self, 'dealer_position', 0) % self.num_players
        start_index_calc = (current_dealer_pos + offset) % self.num_players
        loop_idx = start_index_calc
        for _ in range(self.num_players + 1):
            if loop_idx in self.active_players:
                 if loop_idx < len(self.player_stacks) and self.player_stacks[loop_idx] > 0.01: return loop_idx
            loop_idx = (loop_idx + 1) % self.num_players
        return None

    # --- Hand Setup Methods ---
    def start_new_hand(self, dealer_pos, player_stacks):
        # Reset logic... (keep as before)
        self.hole_cards=[[] for _ in range(self.num_players)]; self.community_cards=[]; self.pot=0.0; self.betting_round=self.PREFLOP
        self.player_stacks=[float(s) for s in player_stacks]; self.player_bets_in_round=[0.0]*self.num_players
        self.player_total_bets_in_hand=[0.0]*self.num_players; self.player_folded=[False]*self.num_players
        self.player_all_in=[False]*self.num_players; self.current_player_idx=-1; self.current_bet=0.0; self.last_raiser=None; self.last_raise=0.0
        self.players_acted_this_round=set(); self.dealer_position=dealer_pos % self.num_players; self.deck=Deck(); self.deck.shuffle()
        self.active_players=[i for i, stack in enumerate(self.player_stacks) if stack > 0.01]
        if len(self.active_players) < 2: self.betting_round = self.HAND_OVER; return
        self._deal_hole_cards(); self._post_blinds(); self._start_betting_round()
        if self.verbose_debug: print(f"DEBUG GS: Hand Started. Dealer={self.dealer_position}. Players={self.active_players}. State:\n{self}")
    def _deal_hole_cards(self):
        # Deal logic... (keep as before)
        if not self.active_players: return
        start_deal_idx=-1; potential_start_idx=self._find_player_relative_to_dealer(1)
        if potential_start_idx is not None: start_deal_idx = potential_start_idx
        else: current_check=(self.dealer_position+1)%self.num_players;
        for _ in range(self.num_players):
             if current_check in self.active_players: start_deal_idx=current_check; break; current_check=(current_check+1)%self.num_players
        if start_deal_idx==-1 and self.active_players: start_deal_idx=self.active_players[0]
        elif start_deal_idx==-1: print("ERR: No start deal idx"); return
        for r in range(2):
            cur=start_deal_idx; dealt=0; att=0;
            while dealt<len(self.active_players) and att<self.num_players*2:
                att+=1;
                if not(0<=cur<self.num_players): cur=(cur+1)%self.num_players; continue
                if cur in self.active_players:
                    while len(self.hole_cards)<=cur: self.hole_cards.append([])
                    if len(self.hole_cards[cur])==r:
                        if len(self.deck)>0: self.hole_cards[cur].append(self.deck.deal()); dealt+=1
                        else: print(f"ERR Deck empty R{r+1}"); self.betting_round=self.HAND_OVER; return
                cur=(cur+1)%self.num_players
                if cur==start_deal_idx and dealt<len(self.active_players) and att>self.num_players: print(f"WARN Deal stuck R{r+1}"); break
            if att>=self.num_players*2 and dealt<len(self.active_players): print(f"ERR Fail deal R{r+1}"); self.betting_round=self.HAND_OVER; return
    def _post_blinds(self):
        # Blind posting logic... (keep as before)
        if len(self.active_players)<2: return; sb_idx=None; bb_idx=None;
        if self.num_players==2: sb_idx=self.dealer_position if self.dealer_position in self.active_players else None; bb_idx=(self.dealer_position+1)%2 if (self.dealer_position+1)%2 in self.active_players else None;
        else: sb_idx=self._find_player_relative_to_dealer(1); bb_idx=self._find_player_relative_to_dealer(2)
        posted_bb=0.0;
        if sb_idx is not None: amt=min(self.small_blind, self.player_stacks[sb_idx]);
        if amt>0: self.player_stacks[sb_idx]-=amt; self.player_bets_in_round[sb_idx]=amt; self.player_total_bets_in_hand[sb_idx]+=amt; self.pot+=amt;
        if abs(self.player_stacks[sb_idx])<0.01: self.player_all_in[sb_idx]=True
        if bb_idx is not None: alr=self.player_bets_in_round[bb_idx]; need=self.big_blind-alr; amt=min(need, self.player_stacks[bb_idx]);
        if amt>0: self.player_stacks[bb_idx]-=amt; self.player_bets_in_round[bb_idx]+=amt; self.player_total_bets_in_hand[bb_idx]+=amt; self.pot+=amt;
        if abs(self.player_stacks[bb_idx])<0.01: self.player_all_in[bb_idx]=True
        posted_bb=self.player_bets_in_round[bb_idx]
        self.current_bet = max(self.player_bets_in_round) if self.player_bets_in_round else 0.0; self.last_raise = self.big_blind
        if bb_idx is not None and posted_bb >= self.big_blind-0.01: self.last_raiser = bb_idx
        elif sb_idx is not None and self.player_bets_in_round[sb_idx] >= posted_bb and self.player_bets_in_round[sb_idx] > 0: self.last_raiser = sb_idx
        else: self.last_raiser = None
        if self.verbose_debug: print(f"DEBUG GS: Blinds Posted. SB={sb_idx}, BB={bb_idx}. Pot={self.pot}. CurrentBet={self.current_bet}")

    # --- Round Progression Methods ---
    def _start_betting_round(self):
        # Start round logic... (keep as before)
        self.players_acted_this_round=set(); first_player=None
        if self.betting_round != self.PREFLOP: self.current_bet=0.0; self.last_raiser=None; self.last_raise=self.big_blind;
        for i in range(self.num_players): self.player_bets_in_round[i]=0.0
        if self.betting_round==self.PREFLOP:
            if self.num_players==2: first_player=self.dealer_position if self.dealer_position in self.active_players else None
            else: bb=self._find_player_relative_to_dealer(2); start=bb if bb is not None else self._find_player_relative_to_dealer(1); first_player=self._get_next_active_player(start) if start is not None else self._get_next_active_player(self.dealer_position)
        else: first_player=self._get_next_active_player(self.dealer_position)
        self.current_player_idx = first_player if first_player is not None else -1
        if len(self.active_players)<=1: self.current_player_idx = -1; self.betting_round = self.HAND_OVER
        elif self._check_all_active_are_allin(): self.current_player_idx = -1 # Skip betting if all-in
        if self.verbose_debug: print(f"DEBUG GS: Start Round {self.betting_round}. First Actor: P{self.current_player_idx}")
    def deal_flop(self):
        if self.verbose_debug: print("DEBUG GS: Trying Deal Flop...")
        # Deal logic... (keep as before, with verbose prints possible inside)
        if self.betting_round!=self.PREFLOP or len(self.active_players)<=1: self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
            if len(self.community_cards)<3:
                if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False; self.deck.deal(); # Burn
                for _ in range(3):
                    if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
                    self.community_cards.append(self.deck.deal())
            self.betting_round = self.FLOP; return self.deal_turn()
        if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
        self.deck.deal(); # Burn
        for _ in range(3):
            if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
            self.community_cards.append(self.deck.deal())
        self.betting_round = self.FLOP; self._start_betting_round(); return True
    def deal_turn(self):
        if self.verbose_debug: print("DEBUG GS: Trying Deal Turn...")
        # Deal logic... (keep as before)
        if self.betting_round!=self.FLOP or len(self.active_players)<=1: self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
            if len(self.community_cards)<4:
                if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False; self.deck.deal(); # Burn
                if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
                self.community_cards.append(self.deck.deal())
            self.betting_round = self.TURN; return self.deal_river()
        if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
        self.deck.deal(); # Burn
        if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
        self.community_cards.append(self.deck.deal())
        self.betting_round = self.TURN; self._start_betting_round(); return True
    def deal_river(self):
        if self.verbose_debug: print("DEBUG GS: Trying Deal River...")
        # Deal logic... (keep as before)
        if self.betting_round!=self.TURN or len(self.active_players)<=1: self.betting_round=self.HAND_OVER; return False
        if self._check_all_active_are_allin():
             if len(self.community_cards)<5:
                 if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False; self.deck.deal(); # Burn
                 if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
                 self.community_cards.append(self.deck.deal())
             self.betting_round = self.SHOWDOWN; self.current_player_idx=-1; self.players_acted_this_round=set(); return True
        if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
        self.deck.deal(); # Burn
        if len(self.deck) <= 0: self.betting_round=self.HAND_OVER; return False
        self.community_cards.append(self.deck.deal())
        self.betting_round = self.RIVER; self._start_betting_round(); return True
    def _check_all_active_are_allin(self):
        if not self.active_players: return True
        for p_idx in self.active_players:
             try:
                 if not self.player_folded[p_idx] and not self.player_all_in[p_idx] and self.player_stacks[p_idx] > 0.01: return False
             except IndexError: continue
        return True

    # --- CORRECTED _move_to_next_player ---
    def _move_to_next_player(self):
        if self.current_player_idx == -1:
            if self.verbose_debug: print("DEBUG GS move_next: Already -1, no move.")
            return # Round is already over or not started

        start_search_idx = self.current_player_idx
        next_player_idx = self._get_next_active_player(start_search_idx)

        if self.verbose_debug: print(f"DEBUG GS move_next: From P{start_search_idx} -> Found P{next_player_idx}")
        # Assign the result directly to current_player_idx
        self.current_player_idx = next_player_idx if next_player_idx is not None else -1


    # --- Action Handling ---
    def apply_action(self, action):
        if not isinstance(action, tuple) or len(action)!=2: raise ValueError(f"Invalid action:{action}.");
        action_type, amount_input = action; amount = float(amount_input); player_idx = self.current_player_idx
        if self.verbose_debug: print(f"DEBUG GS apply_action: P{player_idx} attempts {action_type} {amount}")
        if player_idx==-1: raise ValueError("No current player.");
        if not (0 <= player_idx < self.num_players): raise ValueError(f"Invalid P idx {player_idx}")
        if player_idx >= len(self.player_folded) or player_idx >= len(self.player_all_in) or player_idx >= len(self.player_stacks) or player_idx >= len(self.player_bets_in_round): raise ValueError(f"P idx {player_idx} OOB Lists")
        is_active=player_idx in self.active_players; is_all_in=self.player_all_in[player_idx]
        if not is_active and not is_all_in: raise ValueError(f"P{player_idx} not active.");
        if self.player_folded[player_idx]: raise ValueError(f"P{player_idx} folded.");
        if is_all_in and action_type!='fold': raise ValueError(f"P{player_idx} all-in.");

        new_state = self.clone()
        try:
            new_state._apply_action_logic(player_idx, action_type, amount)
            if self.verbose_debug: print(f"DEBUG GS apply_action: Action Logic Applied. State after logic:\n{new_state}")
        except ValueError as e:
            if self.verbose_debug: print(f"ERROR GS apply_action P{player_idx} {action}: {e}\nState BEFORE action:\n{self}")
            raise # Re-raise the error

        # Check round end and advance *in the new state*
        round_over = new_state._is_betting_round_over()
        if self.verbose_debug: print(f"DEBUG GS apply_action: Round Over Check -> {round_over}")

        if round_over:
            if self.verbose_debug: print(f"DEBUG GS apply_action: Trying advance round...")
            new_state._try_advance_round()
        else:
            if self.verbose_debug: print(f"DEBUG GS apply_action: Moving turn...")
            new_state._move_to_next_player()
            # Check again after moving turn (important for heads up etc.)
            round_over_after_move = new_state._is_betting_round_over()
            if new_state.current_player_idx != -1 and round_over_after_move:
                 if self.verbose_debug: print(f"DEBUG GS apply_action: Round ended AFTER moving turn. Advancing...")
                 new_state._try_advance_round()

        if self.verbose_debug: print(f"DEBUG GS apply_action: Returning new state. Next turn: P{new_state.current_player_idx}")
        return new_state

    def _apply_action_logic(self, player_idx, action_type, amount):
        # Apply logic... (keep as before)
        player_stack=self.player_stacks[player_idx]; current_round_bet=self.player_bets_in_round[player_idx]; self.players_acted_this_round.add(player_idx)
        if action_type=="fold": self.player_folded[player_idx]=True; self.active_players.remove(player_idx) if player_idx in self.active_players else None; self.current_player_idx = -1 if len(self.active_players) <= 1 else self.current_player_idx
        elif action_type=="check":
             if self.current_bet > current_round_bet + 0.01: raise ValueError(f"Check invalid: {self.current_bet} > {current_round_bet}")
        elif action_type=="call":
            bet_to_call = self.current_bet - current_round_bet;
            if bet_to_call <= 0.01: return # Treat as check
            actual_call = min(bet_to_call, player_stack);
            self.player_stacks[player_idx]-=actual_call; self.player_bets_in_round[player_idx]+=actual_call; self.player_total_bets_in_hand[player_idx]+=actual_call; self.pot+=actual_call
            if abs(self.player_stacks[player_idx]) < 0.01: self.player_all_in[player_idx] = True
        elif action_type=="bet":
            if self.current_bet > 0.01: raise ValueError("Must raise facing bet.");
            if amount < 0.01: raise ValueError("Bet must be positive.");
            min_bet=max(1.0, min(self.big_blind, player_stack+current_round_bet));
            if amount < min_bet-0.01 and amount < player_stack: raise ValueError(f"Bet {amount} < min {min_bet}");
            actual_bet=min(amount, player_stack);
            self.player_stacks[player_idx]-=actual_bet; self.player_bets_in_round[player_idx]+=actual_bet; self.player_total_bets_in_hand[player_idx]+=actual_bet; self.pot+=actual_bet
            self.current_bet=self.player_bets_in_round[player_idx]; self.last_raise=actual_bet; self.last_raiser=player_idx;
            if abs(self.player_stacks[player_idx]) < 0.01: self.player_all_in[player_idx] = True
            self.players_acted_this_round={player_idx} # Reset
        elif action_type=="raise":
            if self.current_bet <= 0.01: raise ValueError("Must bet when no bet made.");
            total_bet_this_round=amount; amount_increase = total_bet_this_round - current_round_bet;
            if amount_increase <= 0.01: raise ValueError(f"Raise {total_bet_this_round} <= current {current_round_bet}");
            if amount_increase > player_stack + 0.01: raise ValueError(f"Raise requires {amount_increase:.2f}, stack={player_stack:.2f}");
            min_raise_inc = max(self.last_raise, self.big_blind); min_legal_total = self.current_bet + min_raise_inc;
            is_all_in = abs(player_stack - amount_increase) < 0.01
            if total_bet_this_round < min_legal_total - 0.01 and not is_all_in: raise ValueError(f"Raise {total_bet_this_round:.2f} < min {min_legal_total:.2f}");
            actual_inc=min(amount_increase, player_stack);
            self.player_stacks[player_idx]-=actual_inc; self.player_total_bets_in_hand[player_idx]+=actual_inc; self.pot+=actual_inc;
            self.player_bets_in_round[player_idx] = current_round_bet + actual_inc # Correct update
            new_bet_level = self.player_bets_in_round[player_idx]
            self.last_raise = new_bet_level - self.current_bet; self.current_bet = new_bet_level; self.last_raiser = player_idx;
            if is_all_in: self.player_all_in[player_idx] = True
            self.players_acted_this_round = {player_idx} # Reset
        else: raise ValueError(f"Unknown action: {action_type}")
        self.current_bet = max(self.player_bets_in_round) if self.player_bets_in_round else 0.0

    def get_available_actions(self):
        # Get actions logic... (Keep as V7 - it was correct)
        actions = []; player_idx = self.current_player_idx
        if player_idx == -1: return []
        try:
            if self.player_folded[player_idx] or self.player_all_in[player_idx]: return []
        except IndexError: return []
        player_stack = self.player_stacks[player_idx]; player_bet_this_round = self.player_bets_in_round[player_idx]; current_bet_level = self.current_bet

        actions.append(("fold", 0))
        bet_to_call = current_bet_level - player_bet_this_round; can_check = bet_to_call < 0.01

        if can_check: actions.append(("check", 0))
        else:
            call_amount = min(bet_to_call, player_stack)
            if call_amount > 0.01: actions.append(("call", int(round(call_amount))))

        effective_stack_after_call = player_stack - max(0, bet_to_call)
        if effective_stack_after_call > 0.01:
            min_increase = max(self.last_raise, self.big_blind)
            if current_bet_level < 0.01: # Can BET
                prefix="bet"; min_bet=min(max(self.big_blind,1.0), player_stack)
                if player_stack >= min_bet - 0.01: actions.append((prefix,int(round(min_bet))))
                pot_bet=min(player_stack, self.pot); pot_bet=max(min_bet,pot_bet)
                if pot_bet < player_stack-0.01 and abs(pot_bet-min_bet)>0.01: actions.append((prefix, int(round(pot_bet))))
                all_in = player_stack
                if all_in > 0.01 :
                    already = any(abs(a[1]-all_in)<0.01 and a[0]==prefix for a in actions)
                    if not already: actions.append((prefix, int(round(all_in))))
            else: # Can RAISE
                prefix = "raise"; min_legal_raise_to=current_bet_level+min_increase; max_raise_to=player_bet_this_round+player_stack
                min_raise_needed = min_legal_raise_to - player_bet_this_round

                if player_stack >= min_raise_needed - 0.01:
                     actual_min_raise = min(min_legal_raise_to, max_raise_to)
                     if actual_min_raise - player_bet_this_round <= player_stack + 0.01: actions.append((prefix, int(round(actual_min_raise))))
                     pot_after_call=self.pot+max(0, bet_to_call); pot_raise=current_bet_level+pot_after_call
                     pot_raise=max(actual_min_raise, pot_raise); pot_raise_cap=min(pot_raise, max_raise_to)
                     if (pot_raise_cap - player_bet_this_round <= player_stack + 0.01) and \
                        pot_raise_cap < max_raise_to - 0.01 and abs(pot_raise_cap - actual_min_raise) > 0.01:
                           actions.append((prefix, int(round(pot_raise_cap))))
                # All-in raise
                all_in_raise = max_raise_to
                if all_in_raise > current_bet_level + 0.01:
                    is_valid_raise = (all_in_raise >= min_legal_raise_to - 0.01) if (player_stack >= min_raise_needed - 0.01) else True
                    already = any(abs(a[1]-all_in_raise)<0.01 for a in actions if a[0]==prefix or a[0]=='call')
                    if is_valid_raise and not already:
                         if all_in_raise - player_bet_this_round <= player_stack + 0.01: actions.append((prefix, int(round(all_in_raise))))

        # Final Cleanup/Dedupe/Sort
        final = {};
        for act, amt_f in actions:
            amt=max(0, int(round(float(amt_f)))); key=(act, amt);
            if key in final: continue;
            cost=0;
            if act=='call': cost=min(max(0, bet_to_call), player_stack)
            elif act=='bet': cost=amt
            elif act=='raise': cost=amt - player_bet_this_round
            if cost <= player_stack + 0.01: final[key] = key
        def sort_key(a): t,amt=a; o={"fold":0,"check":1,"call":2,"bet":3,"raise":4}; return (o.get(t,99), amt)
        return sorted(list(final.values()), key=sort_key)


    # --- _is_betting_round_over correct ---
    def _is_betting_round_over(self):
        if self.verbose_debug: print(f"DEBUG GS _is_betting_round_over Check: Active={self.active_players}, Acted={self.players_acted_this_round}, CurrBet={self.current_bet}, Bets={[f'{b:.0f}' for b in self.player_bets_in_round]}")
        if len(self.active_players) < 2: return True
        can_act=[p for p in self.active_players if not self.player_all_in[p] and self.player_stacks[p] > 0.01]
        if len(can_act) <= 1:
             if len(can_act) == 1: p=can_act[0]; facing=self.current_bet>self.player_bets_in_round[p]+0.01; acted=p in self.players_acted_this_round; result = not facing or (facing and acted);
             else: result = True # No one can act
             if self.verbose_debug: print(f"DEBUG GS _is_betting_round_over Result (<=1 Can Act): {result}")
             return result
        acted_count=sum(1 for p in can_act if p in self.players_acted_this_round); all_matched=all(abs(self.player_bets_in_round[p]-self.current_bet)<0.01 for p in can_act)
        init_check=not self.players_acted_this_round and self.current_bet <= 0.01; closed=acted_count == len(can_act) and all_matched
        action_happened = len(self.players_acted_this_round)>0 or self.last_raiser is not None
        result = closed and (action_happened or init_check)
        if self.verbose_debug: print(f"DEBUG GS _is_betting_round_over Result (>1 Can Act): Closed={closed}, ActionHappened={action_happened}, InitCheck={init_check} -> {result}")
        return result


    # --- _try_advance_round correct ---
    def _try_advance_round(self):
        if self.verbose_debug: print(f"DEBUG GS _try_advance_round: Current Round={self.betting_round}, Active={self.active_players}, AllInCheck={self._check_all_active_are_allin()}")
        if len(self.active_players)<=1: self.betting_round=self.HAND_OVER; return
        if self._check_all_active_are_allin() and self.betting_round<self.RIVER:
            if self.betting_round==self.PREFLOP: self.deal_flop()
            if self.betting_round==self.FLOP: self.deal_turn()
            if self.betting_round==self.TURN: self.deal_river()
            if self.betting_round==self.RIVER and len(self.community_cards)<5: self.deal_river()
            if self.betting_round < self.SHOWDOWN: self.betting_round = self.SHOWDOWN
            self.current_player_idx=-1; self.players_acted_this_round=set();
            if self.verbose_debug: print(f"DEBUG GS _try_advance_round: Advanced to {self.betting_round} due to all-in")
            return
        rnd=self.betting_round; nxt=False
        if rnd==self.PREFLOP: nxt=self.deal_flop(); new_rnd="Flop"
        elif rnd==self.FLOP: nxt=self.deal_turn(); new_rnd="Turn"
        elif rnd==self.TURN: nxt=self.deal_river(); new_rnd="River"
        elif rnd==self.RIVER: self.betting_round=self.SHOWDOWN; self.current_player_idx=-1; self.players_acted_this_round=set(); new_rnd="Showdown"; nxt=True;
        else: new_rnd="?"; nxt=False; # Already terminal?
        if self.verbose_debug and nxt: print(f"DEBUG GS _try_advance_round: Advanced to {new_rnd}")
        elif self.verbose_debug and not nxt: print(f"DEBUG GS _try_advance_round: Failed to advance from {self.betting_round}. Setting HAND_OVER.")
        if not nxt and self.betting_round<self.SHOWDOWN : self.betting_round=self.HAND_OVER

    # --- Terminal State & Utility correct ---
    def is_terminal(self):
        term = False
        if len(self.active_players)<=1: term = True;
        elif self.betting_round>=self.SHOWDOWN: term = True;
        # Removed the complex check as it should be covered by betting_round advancing
        # elif self.betting_round==self.RIVER and self.current_player_idx == -1 and self._is_betting_round_over(): term = True
        # if self.verbose_debug and term: print(f"DEBUG GS is_terminal: True (Round={self.betting_round}, Active={len(self.active_players)})")
        return term
    def get_utility(self, player_idx): return 0.0

    # --- determine_winners correct ---
    def determine_winners(self, player_names=None):
        if not self.is_terminal():
             if self._is_betting_round_over(): self._try_advance_round()
             if not self.is_terminal(): print("WARN: determine_winners non-terminal."); return []
        if not self.active_players and self.pot < 0.01 : return []
        elif not self.active_players and self.pot > 0.01: self.pot = 0; return []
        total_pot=self.pot; pots_info=[]; showdown_players=self.active_players.copy()
        if len(showdown_players)==1:
            winner=showdown_players[0]; won=total_pot; self.player_stacks[winner]+=won if winner<len(self.player_stacks) else 0;
            self.pot = 0; pots_info.append(([winner], won, [winner])); return pots_info
        hands={}; valid_players=[];
        for p in showdown_players:
            if p>=len(self.hole_cards) or len(self.hole_cards[p])!=2 or len(self.community_cards)<3: continue
            all_c = self.hole_cards[p] + self.community_cards;
            if len(all_c)<5: continue
            try: hands[p]=HandEvaluator.evaluate_hand(all_c); valid_players.append(p);
            except Exception as e: print(f"ERR eval P{p}:{e}")
        if not hands or not valid_players: self.pot=0; return[]
        contrib=sorted([(p, self.player_total_bets_in_hand[p]) for p in valid_players], key=lambda x: x[1])
        created_pots=[]; last_contrib=0.0; eligible=valid_players[:]
        for p, lvl in contrib:
            inc = lvl-last_contrib;
            if inc > 0.01: amt=inc*len(eligible); created_pots.append({'amount':amt, 'eligible':eligible[:]}) if amt > 0.01 else None; last_contrib=lvl;
            if p in eligible: eligible.remove(p)
        dist_total=0
        for info in created_pots:
            amt=info['amount']; elig=info['eligible'];
            if amt < 0.01: continue
            elig_h = {p:hands[p] for p in elig if p in hands};
            if not elig_h: continue
            best_h=max(elig_h.values()); winners=[p for p,v in elig_h.items() if v==best_h];
            if winners: share=amt/len(winners); pots_info.append((winners, amt, elig)); dist_total+=amt;
            for w in winners: self.player_stacks[w]+=share if w<len(self.player_stacks) else 0;
        left=total_pot-dist_total;
        if abs(left)>0.01 and pots_info:
            last_w, last_a, last_e = pots_info[-1]; share=left/len(last_w);
            for w in last_w: self.player_stacks[w]+=share if w<len(self.player_stacks) else 0; pots_info[-1]=(last_w, last_a+left, last_e);
        self.pot=0; return pots_info

    # --- Cloning and Info methods correct ---
    def clone(self):
        new=GameState(self.num_players,0,self.small_blind,self.big_blind); new.num_players=self.num_players; new.pot=self.pot;
        new.current_player_idx=self.current_player_idx; new.betting_round=self.betting_round; new.current_bet=self.current_bet;
        new.last_raiser=self.last_raiser; new.last_raise=self.last_raise; new.dealer_position=self.dealer_position;
        new.small_blind=self.small_blind; new.big_blind=self.big_blind; new.player_stacks=self.player_stacks[:];
        new.hole_cards=[c[:] if c else [] for c in self.hole_cards]; new.community_cards=self.community_cards[:];
        new.player_total_bets_in_hand=self.player_total_bets_in_hand[:]; new.player_bets_in_round=self.player_bets_in_round[:];
        new.player_folded=self.player_folded[:]; new.player_all_in=self.player_all_in[:]; new.active_players=self.active_players[:];
        new.players_acted_this_round=self.players_acted_this_round.copy(); new.deck=self.deck.clone();
        new.verbose_debug = self.verbose_debug # Copy debug flag
        return new
    def get_betting_history(self): # Placeholder - TODO: Needs robust implementation
        pb=int(round(self.pot/self.big_blind)) if self.big_blind>0 else 0; cb=int(round(self.current_bet/self.big_blind)) if self.big_blind>0 else 0
        n=len(self.active_players); act=self.current_player_idx; bh=hash(tuple(int(b) for b in self.player_bets_in_round))
        return f"R{self.betting_round}|P{pb}|CB{cb}|N{n}|Act{act}|BH{bh}"
    def get_position(self, player_idx):
        if self.num_players<=1: return 0; dealer=getattr(self, 'dealer_position', 0) % self.num_players
        if self.num_players==2: return 1 if player_idx == dealer else 0
        else: return (player_idx - dealer + self.num_players) % self.num_players
    def __str__(self):
        rnd=self.ROUND_NAMES.get(self.betting_round, f"Unk({self.betting_round})")
        s=f"-- State (Rnd:{rnd}|Bet:{self.current_bet:.0f}|LastRaise:{self.last_raise:.0f}) Pot:{self.pot:.0f} --\n"; s+=f"Board: {' '.join(str(c) for c in self.community_cards)}\n"
        s+=f"D:{self.dealer_position}, Turn:{self.current_player_idx}, LastRaiser:{self.last_raiser}, Acted:{self.players_acted_this_round}\n"
        for i in range(self.num_players):
            stk="ERR";pos="ERR";rB="ERR";hB="ERR";fld=False;ain=False;act=False;cards="--"
            try: stk=f"{self.player_stacks[i]:.0f}" if i<len(self.player_stacks) else stk
            except IndexError: pass
            try: pos=f"Pos{self.get_position(i)}"
            except Exception: pos=f"ERR_P{i}"
            try: rB=f"{self.player_bets_in_round[i]:.0f}" if i<len(self.player_bets_in_round) else rB
            except IndexError: pass
            try: hB=f"{self.player_total_bets_in_hand[i]:.0f}" if i<len(self.player_total_bets_in_hand) else hB
            except IndexError: pass
            try: fld=self.player_folded[i] if i<len(self.player_folded) else fld
            except IndexError: pass
            try: ain=self.player_all_in[i] if i<len(self.player_all_in) else ain
            except IndexError: pass
            try: act=i in self.active_players
            except Exception: pass
            stat=("F" if fld else "A" if ain else("*" if act else "-")); cards=' '.join(str(c) for c in self.hole_cards[i]) if i<len(self.hole_cards) and self.hole_cards[i] else cards
            s+=f"  P{i}{stat} {pos}: S={stk}, RndB={rB}, HndB={hB}, Cards=[{cards}]\n"
        s+=f"Deck:{len(self.deck)} cards\n"+"-"*20+"\n"; return s
# --- END OF FILE organized_poker_bot/game_engine/game_state.py ---
