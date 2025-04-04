# --- START OF FILE organized_poker_bot/cfr/information_set.py ---
"""
Implementation of information sets for poker CFR.
This module provides the InformationSet class for managing regrets and strategies.
(Refactored V2: Added strategy_sum update, removed num_visits)
"""

import numpy as np
from collections import defaultdict

class InformationSet:
    """
    A class representing an information set in the game.

    An information set is a collection of game states that are indistinguishable
    to a player. This class manages the regrets and strategies for an information set.

    Attributes:
        actions (list): List of available actions as (type, amount) tuples.
        regret_sum (defaultdict): Dictionary mapping actions to cumulative regrets.
        strategy_sum (defaultdict): Dictionary mapping actions to cumulative strategies,
                                   weighted by player reach probability.
    """

    def __init__(self, actions):
        """
        Initialize an information set.

        Args:
            actions (list): List of available actions, expected as (type, amount) tuples.
        """
        # Ensure actions are unique tuples
        unique_actions = []
        seen_actions = set()
        if not isinstance(actions, list): actions = [] # Handle non-list input

        for action in actions:
            # Ensure action is a tuple, default amount to 0 if string
            if isinstance(action, str):
                 action_tuple = (action, 0)
            elif isinstance(action, tuple) and len(action) == 2:
                 action_tuple = action
            else: continue # Skip invalid action formats

            if action_tuple not in seen_actions:
                unique_actions.append(action_tuple)
                seen_actions.add(action_tuple)

        self.actions = unique_actions # Store unique, tuple actions
        # Initialize sums for all defined actions to avoid KeyErrors later
        # Use defaultdict for safety, although keys should be set here
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        for action in self.actions:
             self.regret_sum[action] = 0.0
             self.strategy_sum[action] = 0.0


    def get_strategy(self):
        """
        Get the current strategy for this information set based on Regret Matching.
        Uses positive regrets to determine action probabilities for the current iteration.

        Returns:
            dict: A dictionary mapping actions to probabilities for the current iteration.
                  Returns empty dict if no actions are defined for this info set.
        """
        if not self.actions: # Handle case with no actions
            return {}

        strategy = {}
        normalization_sum = 0.0
        positive_regrets = {}

        for action in self.actions:
             # Get regret, default to 0.0 if somehow missing (shouldn't happen with init)
             regret = self.regret_sum.get(action, 0.0)
             positive_regret = max(0.0, regret)
             positive_regrets[action] = positive_regret
             normalization_sum += positive_regret

        # Calculate strategy probabilities
        if normalization_sum > 0:
            for action in self.actions:
                strategy[action] = positive_regrets[action] / normalization_sum
        else:
            # Default to uniform strategy if no positive regrets
            num_actions = len(self.actions)
            # Ensure num_actions > 0 to avoid division by zero
            if num_actions > 0:
                prob = 1.0 / num_actions
                for action in self.actions:
                    strategy[action] = prob
            # If num_actions is 0 (self.actions is empty), strategy remains empty {}

        # This function ONLY returns the current iteration's strategy based on regrets.
        # The accumulation for the average strategy happens elsewhere.
        return strategy


    def update_strategy_sum(self, current_strategy, weighted_reach_prob):
        """
        Update the cumulative strategy sum using the current strategy
        and the player's weighted reach probability for this iteration.

        Args:
            current_strategy (dict): The strategy (action probs) used in the current iteration.
            weighted_reach_prob (float): The reach probability of the player reaching this
                                         info set in this iteration, potentially weighted
                                         by iteration number for CFR variants (like Linear CFR).
        """
        for action in self.actions:
             # Get probability from current strategy, default to 0 if action missing
             action_prob = current_strategy.get(action, 0.0)
             # Accumulate weighted probability into the strategy sum
             self.strategy_sum[action] += weighted_reach_prob * action_prob


    def get_average_strategy(self):
        """
        Get the average strategy over all iterations based on the cumulative strategy_sum.

        Returns:
            dict: A dictionary mapping actions to average probabilities.
                  Returns empty dict if the info set was never reached or has no actions.
        """
        if not self.actions: # Handle case with no actions
             return {}

        avg_strategy = {}
        # Sum of all accumulated weighted strategy probabilities
        normalization_sum = sum(self.strategy_sum.values())

        if normalization_sum > 0:
            for action in self.actions:
                # Get accumulated sum for this action, default to 0.0
                action_sum = self.strategy_sum.get(action, 0.0)
                avg_strategy[action] = action_sum / normalization_sum
        else:
            # If the info set was effectively never reached with positive probability
            # (normalization_sum is 0), default the average strategy to uniform.
            num_actions = len(self.actions)
            # Ensure num_actions > 0 before calculating uniform probability
            if num_actions > 0:
                prob = 1.0 / num_actions
                for action in self.actions:
                    avg_strategy[action] = prob
            # If num_actions is 0, avg_strategy remains empty {}

        return avg_strategy


    def __str__(self):
        """ String representation of the information set for debugging. """
        # Safely format average strategy
        try:
            avg_strat_items = self.get_average_strategy().items()
            # Format actions consistently: fold, check, call amount, bet amount, raise amount
            def format_action(a_tuple):
                 typ, amt = a_tuple
                 if typ in ('bet', 'raise', 'call') and amt > 0: return f"{typ}{amt}"
                 return typ
            avg_strat_repr = {format_action(a): f"{p:.3f}" for a, p in avg_strat_items}
        except Exception as e: avg_strat_repr = f"Error: {e}"

        # Safely format regrets
        try:
            regret_repr = {format_action(a): f"{r:.3f}" for a, r in self.regret_sum.items()}
        except Exception as e: regret_repr = f"Error: {e}"

        # Format actions list for display
        actions_str = [format_action(a) for a in self.actions]

        return f"InfoSet(Actions={actions_str}, AvgStrat={avg_strat_repr}, Regrets={regret_repr})"
# --- END OF FILE organized_poker_bot/cfr/information_set.py ---
