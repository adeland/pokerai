# --- START OF FILE organized_poker_bot/cfr/action_abstraction.py ---
"""
Implementation of action abstraction techniques for poker CFR.
(Refactored V4: Separate bet/raise sizing based on request. Pot frac for bet, multiplier for raise.)
"""

import os
import sys
import math

# Add the parent directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute imports
from organized_poker_bot.game_engine.game_state import GameState

class ActionAbstraction:
    """
    Action abstraction with distinct sizing logic:
    - Opening Bets: Fractions of the current pot.
    - Re-raises: Multipliers of the last aggressor's raise amount.
    Includes fold, check/call, min-legal aggressive action, and all-in.
    Checks GameState's raise cap.
    """

    # Define desired sizings
    # For opening BETS (first aggressive action on street) - Fractions of POT
    POT_FRACTIONS_BET = [0.33, 0.5, 0.75, 1.0]
    # For RAISES (when facing a bet/raise) - Multipliers of LAST RAISE AMOUNT
    # e.g., 2.5 means raise TO current_bet + 2.5 * last_raise_amount
    RAISE_MULTIPLIERS = [2.5, 3.5] # Example multipliers (can add 4.0 if needed)

    @staticmethod
    def abstract_actions(available_actions, game_state):
        """
        Abstract available actions to a limited set with specific sizing logic.

        Args:
            available_actions (list): Original list from get_available_actions().
            game_state (GameState): Current game state.

        Returns:
            list: Abstracted actions, filtered for legality and distinctness.
        """
        abstracted_actions_dict = {} # Use amount as key for aggressive actions for distinctness

        player_idx = game_state.current_player_idx
        if player_idx < 0 or player_idx >= game_state.num_players: return []

        player_stack = game_state.player_stacks[player_idx]
        player_bet_this_round = game_state.player_bets_in_round[player_idx]
        current_bet_level = game_state.current_bet
        pot_size = game_state.pot
        last_raise_amount = max(game_state.last_raise, game_state.big_blind) # Use BB as min if last_raise is invalid/zero

        # 1. Add essential non-aggressive actions
        fold_action = ('fold', 0)
        check_action = ('check', 0)
        call_action = next((a for a in available_actions if a[0] == 'call'), None)

        if fold_action in available_actions: abstracted_actions_dict[('fold', 0)] = fold_action # Use tuple as key for passive
        if check_action in available_actions: abstracted_actions_dict[('check', 0)] = check_action
        if call_action: abstracted_actions_dict[call_action] = call_action # Use tuple as key

        # 2. Determine if aggression is possible & action type
        # Check original actions AND game state's raise cap
        can_bet = any(a[0] == 'bet' for a in available_actions)
        can_raise = any(a[0] == 'raise' for a in available_actions)
        allow_aggression = (game_state.raise_count_this_street < game_state.MAX_RAISES_PER_STREET) and (can_bet or can_raise)
        action_type = "bet" if current_bet_level < 0.01 else "raise"

        if not allow_aggression:
             # Only return passive actions if aggression not possible
             return sorted(list(abstracted_actions_dict.values()), key=lambda a: ({"fold":0,"check":1,"call":2}[a[0]], a[1]))

        # --- Aggression Possible ---
        # Find true min/max available aggressive actions TO these amounts
        original_min_aggressive_to = float('inf')
        original_max_aggressive_to = 0.0 # This should be the all-in TO amount
        valid_original_aggressive_actions = []
        for act, amt in available_actions:
            if act in ['bet', 'raise']:
                valid_original_aggressive_actions.append((act, amt))
                original_min_aggressive_to = min(original_min_aggressive_to, amt)
                original_max_aggressive_to = max(original_max_aggressive_to, amt)

        if not valid_original_aggressive_actions: allow_aggression = False # Should not happen if checks above passed

        # 3. Always add MIN-LEGAL aggressive action (if possible)
        min_legal_action_tuple = None
        if allow_aggression and original_min_aggressive_to != float('inf'):
            min_legal_amount = int(round(original_min_aggressive_to))
            # Ensure affordable cost
            cost_min_legal = min_legal_amount - player_bet_this_round
            if cost_min_legal <= player_stack + 0.01:
                min_legal_action_tuple = (action_type, min_legal_amount)
                abstracted_actions_dict[min_legal_action_tuple[1]] = min_legal_action_tuple # Amount is key

        # 4. Add specific sizings based on action type (BET vs RAISE)
        if allow_aggression:
            if action_type == 'bet':
                # --- BET SIZING: Use Pot Fractions ---
                pot_for_sizing = pot_size
                for fraction in ActionAbstraction.POT_FRACTIONS_BET:
                    target_total_bet = pot_for_sizing * fraction

                    # Clamp by min legal bet and max possible bet (all-in)
                    target_total_bet = max(original_min_aggressive_to, target_total_bet)
                    target_total_bet = min(original_max_aggressive_to, target_total_bet)

                    # Check affordability
                    cost_to_reach_target = target_total_bet - player_bet_this_round # Cost is the bet amount here
                    if cost_to_reach_target <= player_stack + 0.01 and cost_to_reach_target > 0.01:
                         action_tuple = (action_type, int(round(target_total_bet)))
                         # Add using amount as key for distinctness
                         abstracted_actions_dict[action_tuple[1]] = action_tuple

            elif action_type == 'raise':
                # --- RAISE SIZING: Use Multipliers of Last Raise ---
                for multiplier in ActionAbstraction.RAISE_MULTIPLIERS:
                    # Calculate the size of the raise increment
                    # Use max(last_raise, big_blind) to ensure reasonable sizing baseline
                    raise_increment_amount = multiplier * last_raise_amount
                    # Calculate the target total bet level TO reach
                    target_total_bet = current_bet_level + raise_increment_amount

                    # Clamp by min legal raise and max possible raise (all-in)
                    target_total_bet = max(original_min_aggressive_to, target_total_bet)
                    target_total_bet = min(original_max_aggressive_to, target_total_bet)

                    # Check affordability
                    cost_to_reach_target = target_total_bet - player_bet_this_round
                    if cost_to_reach_target <= player_stack + 0.01 and cost_to_reach_target > 0.01:
                         action_tuple = (action_type, int(round(target_total_bet)))
                         # Add using amount as key for distinctness
                         abstracted_actions_dict[action_tuple[1]] = action_tuple

        # 5. Always add ALL-IN aggressive action if valid & distinct
        if allow_aggression and original_max_aggressive_to > 0.01:
             # Check if all-in amount is actually > current bet level for raises
             is_valid_aggressive_all_in = True
             if action_type == 'raise' and original_max_aggressive_to <= current_bet_level + 0.01:
                 is_valid_aggressive_all_in = False

             if is_valid_aggressive_all_in:
                 all_in_action_tuple = (action_type, int(round(original_max_aggressive_to)))
                 # Amount as key ensures distinctness automatically
                 abstracted_actions_dict[all_in_action_tuple[1]] = all_in_action_tuple

        # Sort final list like GameState does
        def sort_key(a): t,amt=a; o={"fold":0,"check":1,"call":2,"bet":3,"raise":4}; return (o.get(t,99), amt)
        # Convert dict values back to list and sort
        final_list = sorted(list(abstracted_actions_dict.values()), key=sort_key)

        # Sanity check
        if not any(a[0] in ['bet','raise'] for a in final_list) and allow_aggression:
            # If aggression was allowed but no agg actions were added (e.g. all calcs failed), return original
             print(f"WARN ActionAbstraction: No aggressive actions added despite allow_aggression=True. Orig: {available_actions}")
             return available_actions

        return final_list

# --- END OF FILE organized_poker_bot/cfr/action_abstraction.py ---
