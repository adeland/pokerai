# --- START OF FILE organized_poker_bot/cfr/action_abstraction.py ---
"""
Implementation of action abstraction techniques for poker CFR.
This module provides methods for abstracting action information to reduce the complexity
of the game state space while maintaining strategic relevance.
(Refactored V2: Add stack limit checks)
"""

import os
import sys
import math # Import math for ceil

# Add the parent directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute imports
from organized_poker_bot.game_engine.game_state import GameState # Needed for type hinting and state access

class ActionAbstraction:
    """
    Action abstraction techniques for poker CFR implementation.
    Implements methods for abstracting actions, reducing complexity.
    Now includes stack validation for generated actions.
    """

    @staticmethod
    def abstract_actions(available_actions, game_state):
        """
        Abstract the available actions to a fixed, smaller set based on pot size,
        ensuring generated actions are actually affordable.

        Args:
            available_actions (list): Original list of valid actions from game_state.get_available_actions().
            game_state (GameState): Current game state.

        Returns:
            list: Abstracted actions, including fold, check/call, and specific bet/raise sizes.
                  Returns original actions if abstraction fails or no betting actions exist.
        """
        abstracted_actions_dict = {} # Use dict to prevent duplicates

        # Always keep non-betting actions if available
        if ('fold', 0) in available_actions: abstracted_actions_dict[('fold', 0)] = ('fold', 0)
        if ('check', 0) in available_actions: abstracted_actions_dict[('check', 0)] = ('check', 0)
        # Find the specific call action amount if present
        call_action = next((a for a in available_actions if a[0] == 'call'), None)
        if call_action: abstracted_actions_dict[call_action] = call_action

        # Find if betting or raising is possible from original actions
        can_bet = any(a[0] == 'bet' for a in available_actions)
        can_raise = any(a[0] == 'raise' for a in available_actions)

        # Only proceed with bet/raise abstraction if possible
        if not (can_bet or can_raise):
            return list(abstracted_actions_dict.values()) # Return fold/check/call only

        player_idx = game_state.current_player_idx
        if player_idx < 0 or player_idx >= len(game_state.player_stacks): # Safety check
             print("WARN: ActionAbstraction received invalid player_idx.")
             return list(abstracted_actions_dict.values())

        player_stack = game_state.player_stacks[player_idx]
        player_bet_this_round = game_state.player_bets_in_round[player_idx]
        current_bet_level = game_state.current_bet

        # Find the original min/max bet/raise from GameState for validation
        original_min_bet = min((a[1] for a in available_actions if a[0]=='bet'), default=float('inf'))
        original_min_raise = min((a[1] for a in available_actions if a[0]=='raise'), default=float('inf'))
        original_max_aggressive = max((a[1] for a in available_actions if a[0] in ['bet', 'raise']), default=0) # Should be all-in
        all_in_amount = player_bet_this_round + player_stack # Correct total all-in bet

        # Determine action type ('bet' or 'raise')
        action_type = "bet" if current_bet_level < 0.01 else "raise"

        # 1. Add Minimum Legal Bet/Raise (if possible from original actions)
        min_legal_action = None
        if action_type == "bet" and original_min_bet != float('inf'):
             min_legal_action = (action_type, int(round(original_min_bet)))
        elif action_type == "raise" and original_min_raise != float('inf'):
             min_legal_action = (action_type, int(round(original_min_raise)))

        if min_legal_action and min_legal_action not in abstracted_actions_dict:
             # Ensure cost is affordable (redundant check, but safe)
             cost = min_legal_action[1] - player_bet_this_round
             if cost <= player_stack + 0.01:
                 abstracted_actions_dict[min_legal_action] = min_legal_action

        # 2. Add Pot-Based Sizings (if affordable and valid)
        pot_fractions = [0.5, 0.75, 1.0, 1.5] # Example fractions
        for fraction in pot_fractions:
            # Calculate desired *increase* based on pot size
            # Pot size approx = current pot + current round bets (simple version)
            pot_approx = game_state.pot # More sophisticated: include bets made this round
            desired_increase = pot_approx * fraction

            # Calculate the total bet amount this would correspond TO
            desired_total_bet = player_bet_this_round + desired_increase

            # Clamp the desired total bet by the player's all-in amount
            potential_total_bet = min(all_in_amount, desired_total_bet)
            potential_increase = potential_total_bet - player_bet_this_round

            # --- Validate the potential action ---
            is_valid = False
            # Cost Check: Player must afford the increase
            if potential_increase <= player_stack + 0.01 and potential_increase > 0.01:
                # Size Check: Must be >= min legal bet/raise size (if one exists)
                min_legal_size = original_min_bet if action_type == 'bet' else original_min_raise
                if min_legal_size == float('inf'): min_legal_size = 1.0 # Absolute minimum if no original found
                # Total bet must be >= min legal bet/raise (unless all-in for less)
                is_all_in_for_less = abs(potential_total_bet - all_in_amount) < 0.01

                if potential_total_bet >= min_legal_size - 0.01 or is_all_in_for_less:
                     # Specific check for raises: must increase bet level
                     if action_type == 'raise' and potential_total_bet <= current_bet_level + 0.01:
                          is_valid = False # Not a valid raise
                     else:
                          is_valid = True

            if is_valid:
                 action_tuple = (action_type, int(round(potential_total_bet)))
                 if action_tuple not in abstracted_actions_dict:
                      abstracted_actions_dict[action_tuple] = action_tuple

        # 3. Add All-In Action (if distinct and valid)
        # Check if all-in is actually a bet/raise compared to current situation
        is_all_in_bet = action_type == 'bet' and all_in_amount > 0.01
        is_all_in_raise = action_type == 'raise' and all_in_amount > current_bet_level + 0.01

        if is_all_in_bet or is_all_in_raise:
            all_in_action_tuple = (action_type, int(round(all_in_amount)))
            if all_in_action_tuple not in abstracted_actions_dict:
                 abstracted_actions_dict[all_in_action_tuple] = all_in_action_tuple


        # Sort final list like GameState does
        def sort_key(a): t,amt=a; o={"fold":0,"check":1,"call":2,"bet":3,"raise":4}; return (o.get(t,99), amt)
        final_list = sorted(list(abstracted_actions_dict.values()), key=sort_key)

        # Basic sanity check
        if not final_list: return available_actions # Return original if abstraction failed badly

        # print(f"DEBUG Abstract Actions: Orig={available_actions} -> Abstr={final_list}") # Optional debug
        return final_list


    # --- Helper methods _get_min_bet_amount / _get_max_bet_amount are NOT robust enough ---
    # --- It's better to rely on the valid actions generated by GameState initially ---
    # @staticmethod
    # def _get_min_bet_amount(game_state): ... deprecated ...
    # @staticmethod
    # def _get_max_bet_amount(game_state): ... deprecated ...

# --- END OF FILE organized_poker_bot/cfr/action_abstraction.py ---
