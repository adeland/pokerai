"""
Main game class for running a complete poker game.
"""

import os
import sys
import random # Ensure random is imported

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports that work when run directly
from organized_poker_bot.game_engine.game_state import GameState
from organized_poker_bot.game_engine.player import Player # Base Player
# BotPlayer might be needed if type checking, but usually handled by polymorphism
# from organized_poker_bot.bot.bot_player import BotPlayer
from organized_poker_bot.game_engine.deck import Deck
from organized_poker_bot.game_engine.hand_evaluator import HandEvaluator

# --- Helper for Position Names ---
# Assuming BTN=0, SB=1, BB=2, LJ=3, HJ=4, CO=5 relative to dealer
POS_NAMES_6MAX = ["BTN", "SB", "BB", "LJ", "HJ", "CO"]
# ---

class PokerGame:
    """
    A class for running a complete 6-max No-Limit Hold'em poker game.
    """

    def __init__(self, players, small_blind=50, big_blind=100, interactive=False):
        """ Initializes the Poker Game. """
        if len(players) < 2 or len(players) > 6:
            raise ValueError("Number of players must be between 2 and 6")

        for i, p in enumerate(players):
            if not hasattr(p, 'stack'): p.stack = 10000
            if not hasattr(p, 'name'): p.name = f"Player_{i}"
            p.position = i # Initial seat index

        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = random.randint(0, len(players) - 1)
        self.game_state = None
        self.hand_history = []
        self.interactive = interactive
        # Removed starting print message

    def get_position_name(self, player_idx, dealer_idx, num_players):
        """ Gets the standard position name for a player. """
        if num_players == 6:
            relative_pos = (player_idx - dealer_idx + num_players) % num_players
            return POS_NAMES_6MAX[relative_pos]
        elif num_players == 2: # Heads Up
             return "SB" if player_idx == (dealer_idx + 1) % 2 else "BB/BTN"
        else: # Fallback for other player counts
             return f"Pos{player_idx}"


    def run(self, num_hands=10):
        """ Runs the game for num_hands. """
        print(f"Starting poker game run with {len(self.players)} players for {num_hands} hands...")
        for i in range(num_hands):
            print(f"\n{'='*10} Hand {i+1}/{num_hands} {'='*10}")
            self._rotate_dealer()
            dealer_name = self.players[self.dealer_position].name
            dealer_pos_name = self.get_position_name(self.dealer_position, self.dealer_position, len(self.players))
            print(f"Dealer: {dealer_pos_name} ({dealer_name}, Index: {self.dealer_position})")

            self._play_hand()

            print("\nPlayer stacks after hand:")
            for p_idx, player in enumerate(self.players):
                try: stack_val = player.stack
                except AttributeError: stack_val = "N/A"
                pos_name = self.get_position_name(p_idx, self.dealer_position, len(self.players))
                print(f"  {pos_name} {p_idx} ({player.name}): {stack_val}")

            players_with_chips = [p for p in self.players if hasattr(p, 'stack') and p.stack > 0]
            if len(players_with_chips) <= 1:
                 print("\nGame over: Only one player (or fewer) has chips remaining.")
                 break

            if self.interactive and i < num_hands - 1:
                try: input("\nPress Enter to continue...")
                except EOFError: print("\nEOF detected, ending game."); break


    def _play_hand(self):
        """ Plays a single hand. """
        current_stacks = [getattr(p, 'stack', 0) for p in self.players]
        active_player_indices = [i for i, stack in enumerate(current_stacks) if stack > 0]

        if len(active_player_indices) < 2:
             print("Not enough active players with chips to play a hand.")
             for i, player in enumerate(self.players): player.stack = current_stacks[i]
             return

        self.game_state = GameState(len(self.players), 0, self.small_blind, self.big_blind)
        self.game_state.player_stacks = current_stacks
        self.game_state.dealer_position = self.dealer_position
        self.game_state.active_players = active_player_indices

        try: self.game_state.deal_hole_cards()
        except Exception as e: print(f"ERROR dealing hole cards: {e}"); return

        if self.interactive:
            for i, player in enumerate(self.players):
                if hasattr(player, 'is_human') and player.is_human and i in self.game_state.active_players:
                    hole_cards_list = self.game_state.hole_cards[i] if i < len(self.game_state.hole_cards) else []
                    hole_cards_str = ' '.join(str(c) for c in hole_cards_list) if hole_cards_list else "N/A"
                    pos_name = self.get_position_name(i, self.dealer_position, len(self.players))
                    print(f"\nYour ({player.name} - {pos_name}) hole cards: {hole_cards_str}")

        try:
            self.game_state.post_blinds()
            print(f"Blinds posted. Pot: {self.game_state.pot}")
        except Exception as e: print(f"ERROR posting blinds: {e}"); return

        try:
             if len([p for p in self.game_state.active_players if self.game_state.player_stacks[p] > 0]) > 1:
                 self._betting_round("Preflop")

             if len(self.game_state.active_players) > 1:
                 self.game_state.deal_flop()
                 print(f"\nFlop: {' '.join(str(card) for card in self.game_state.community_cards)} | Pot: {self.game_state.pot}")
                 self._betting_round("Flop")

             if len(self.game_state.active_players) > 1:
                 self.game_state.deal_turn()
                 print(f"\nTurn: {' '.join(str(card) for card in self.game_state.community_cards)} | Pot: {self.game_state.pot}")
                 self._betting_round("Turn")

             if len(self.game_state.active_players) > 1:
                 self.game_state.deal_river()
                 print(f"\nRiver: {' '.join(str(card) for card in self.game_state.community_cards)} | Pot: {self.game_state.pot}")
                 self._betting_round("River")

        except Exception as e:
            print(f"ERROR during betting rounds: {e}")
            import traceback; traceback.print_exc() # Print detailed traceback for betting errors
            # Attempt to conclude hand based on current state if error occurs mid-round
            self._conclude_hand()
            for i, player in enumerate(self.players): player.stack = self.game_state.player_stacks[i]
            return

        self._conclude_hand()

        for i, player in enumerate(self.players):
            if i < len(self.game_state.player_stacks): player.stack = self.game_state.player_stacks[i]


    def _betting_round(self, round_name):
        """ Conducts a betting round. """
        print(f"\n--- {round_name} Betting Round ---")

        active_players_can_act = [p_idx for p_idx in self.game_state.active_players if self.game_state.player_stacks[p_idx] > 0]
        if len(active_players_can_act) <= 1:
             print(f"(Skipping betting: <=1 player can act voluntarily)")
             return

        # Verify starting player is valid
        if self.game_state.current_player_idx not in self.game_state.active_players or self.game_state.player_stacks[self.game_state.current_player_idx] <= 0:
             start_search = (self.dealer_position + 1) % len(self.players) if round_name != "Preflop" else (self.dealer_position + 3) % len(self.players)
             initial_player = start_search; player_found = False
             for _ in range(len(self.players)):
                  if initial_player < len(self.players) and initial_player in self.game_state.active_players and self.game_state.player_stacks[initial_player] > 0:
                       self.game_state.current_player_idx = initial_player; player_found = True; break
                  initial_player = (initial_player + 1) % len(self.players)
             if not player_found: print(f"ERROR: No active player found for {round_name}."); return

        num_actions_this_round = 0
        max_actions = len(self.players) * 4

        while not self.game_state._is_betting_round_over() and num_actions_this_round < max_actions:
            player_idx = self.game_state.current_player_idx

            if player_idx not in self.game_state.active_players or self.game_state.player_stacks[player_idx] <= 0:
                next_player_check_idx = self.game_state._get_next_player_idx(player_idx)
                self.game_state._move_to_next_player()
                if self.game_state.current_player_idx == player_idx: break # Avoid infinite loop if stuck
                continue

            player = self.players[player_idx]
            available_actions = self.game_state.get_available_actions()

            if not available_actions:
                 print(f"DEBUG: No actions for active player {player_idx} ({player.name}) Stack: {self.game_state.player_stacks[player_idx]}. Moving turn.")
                 self.game_state._move_to_next_player()
                 if self.game_state.current_player_idx == player_idx: break
                 continue

            pos_name = self.get_position_name(player_idx, self.game_state.dealer_position, self.game_state.num_players)
            bet_to_call = max(0, self.game_state.current_bet - self.game_state.player_bets[player_idx])
            print(f"\n---\nTurn: {pos_name} (Idx:{player_idx}, {player.name}) | Stack: {self.game_state.player_stacks[player_idx]} | RoundBet: {self.game_state.player_bets[player_idx]} | Pot: {self.game_state.pot} | ToCall: {bet_to_call}")

            action = self._get_player_action(player, player_idx, available_actions, self.game_state)

            # Validate action format
            if not isinstance(action, tuple) or len(action) != 2:
                 print(f"ERROR: Invalid action format: {action}. Defaulting.")
                 check_action = ('check', 0); fold_action = ('fold', 0)
                 if check_action in available_actions: action = check_action
                 elif fold_action in available_actions: action = fold_action
                 else: action = available_actions[0] if available_actions else ('fold', 0)

            action_type, amount = action
            action_str = f"{action_type} {amount}" if amount > 0 and action_type != 'all_in' else action_type
            if action_type == 'all_in': action_str = f"all_in (total bet: {amount})"
            print(f"Action: {action_str}")

            try:
                acting_player_index = self.game_state.current_player_idx
                new_game_state = self.game_state.apply_action(action)
                self.game_state = new_game_state
            except Exception as e:
                 print(f"FATAL ERROR applying action {action} by {player_idx}: {e}")
                 import traceback; traceback.print_exc()
                 raise

            num_actions_this_round += 1
            # betting_complete = self.game_state._is_betting_round_over() # Check happens at start of loop now

        if num_actions_this_round >= max_actions:
            print(f"WARNING: Betting round {round_name} exceeded max actions safeguard.")


    def _get_player_action(self, player, player_idx, available_actions, current_game_state):
        """ Gets action from Bot, Human, or Random player. """
        action = None
        try:
            if hasattr(player, 'get_action') and callable(player.get_action): # Bot Player
                action = player.get_action(current_game_state, player_idx)
            elif hasattr(player, 'is_human') and player.is_human and self.interactive: # Human Player
                action = self._get_human_action(player, available_actions, current_game_state)
            elif hasattr(player, 'is_random') and player.is_random: # Random Player
                if available_actions:
                    action = random.choice(available_actions)
                else: action = ('fold', 0)
        except Exception as e: print(f"ERROR getting action from {player.name}: {e}")

        # Validate / Default Action
        if action is None or not (isinstance(action, tuple) and len(action) == 2):
            if action is not None: print(f"WARNING: Player {player_idx} invalid action format: {action}. Defaulting.")
            else: print(f"WARNING: Player {player_idx} ({player.name}) failed/no action method. Defaulting.")
            check_action = ('check', 0); fold_action = ('fold', 0)
            action = check_action if check_action in available_actions else (fold_action if fold_action in available_actions else (available_actions[0] if available_actions else ('fold', 0)))
        return action


    def _get_human_action(self, player, available_actions, current_game_state):
        """ Gets action from human via console input. """
        player_idx = current_game_state.current_player_idx
        pos_name = self.get_position_name(player_idx, current_game_state.dealer_position, current_game_state.num_players)
        print(f"\n*** Your Turn ({player.name} - {pos_name}) ***")
        print(f"Stack: {current_game_state.player_stacks[player_idx]}")
        print(f"Round Bet: {current_game_state.player_bets[player_idx]}")
        print(f"Pot: {current_game_state.pot}")
        bet_to_call = max(0, current_game_state.current_bet - current_game_state.player_bets[player_idx])
        print(f"Bet To Call: {bet_to_call}")
        print(f"Community: {' '.join(str(c) for c in current_game_state.community_cards)}")
        hole_cards_str = ' '.join(str(c) for c in current_game_state.hole_cards[player_idx]) if current_game_state.hole_cards[player_idx] else "N/A"
        print(f"Your Cards: {hole_cards_str}")
        print("---")
        print("Available actions:")
        action_map = {}
        for i, action_tuple in enumerate(available_actions):
             action_map[i+1] = action_tuple
             action_type, amount = action_tuple
             amount_str = ""
             if action_type == "all_in": amount_str = f" (commit {current_game_state.player_stacks[player_idx]} -> total {amount})"
             elif action_type == "raise": amount_str = f" to {amount}"
             elif action_type == "call": amount_str = f" {amount}"
             elif action_type == "bet": amount_str = f" {amount}"
             print(f"  {i+1}. {action_type}{amount_str}")
        print("---")

        while True:
            try:
                choice_str = input("Enter choice number: ")
                choice_num = int(choice_str)
                if choice_num in action_map: return action_map[choice_num]
                else: print("Invalid choice.")
            except ValueError: print("Invalid input.")
            except EOFError: print("EOF detected. Folding."); return ('fold', 0)


    def _conclude_hand(self):
         """ Handles awarding uncontested pots or initiating showdown. """
         print(f"\n--- Concluding Hand ---")
         final_active_players = self.game_state.active_players.copy()
         print(f"Players active at end: {final_active_players}")
         print(f"Final pot: {self.game_state.pot}")

         if len(final_active_players) == 1:
             winner_idx = final_active_players[0]
             amount_won = self.game_state.pot
             if winner_idx < len(self.game_state.player_stacks):
                  self.game_state.player_stacks[winner_idx] += amount_won
                  print(f"\n{self.players[winner_idx].name} wins uncontested pot of {amount_won} chips")
             else: print(f"ERROR: Winner index {winner_idx} out of range.")
             self.game_state.pot = 0
         elif len(final_active_players) > 1:
             print("Proceeding to showdown...")
             # Pass the main player list for names
             self.game_state.determine_winners(self.players) # Should update stacks in game_state
         else:
             print("WARNING: Hand concluded with no active players?")
             self.game_state.pot = 0


    def _rotate_dealer(self):
        """ Rotates the dealer button. """
        self.dealer_position = (self.dealer_position + 1) % len(self.players)
