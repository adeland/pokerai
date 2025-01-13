
import random
from itertools import combinations
from collections import defaultdict

def canonicalize_hand(hand):
    """Canonicalize hand: higher card first, and mark as suited ('s') or off-suit ('o')."""
    suited = hand[0]["suit"] == hand[1]["suit"]
    sorted_ranks = tuple(sorted([card["rank"] for card in hand], reverse=True))
    return (sorted_ranks, "s" if suited else "o")
def get_player_position(player_index, num_players):
    """Determine player's position."""
    if player_index == 0:
        return "small_blind"
    elif player_index == 1:
        return "big_blind"
    elif player_index < num_players // 2:
        return "early"
    elif player_index < num_players - 2:
        return "middle"
    else:
        return "late"

class MCCFR:
    def __init__(self, game):
        self.game = game
        self.info_sets = defaultdict(lambda: {"regret_sum": defaultdict(float), "strategy_sum": defaultdict(float)})

    def initialize_info_sets(self):
        all_hands = combinations(self.game.initialize_deck(), 2)
        for hand in all_hands:
            for history in ["small_blind", "big_blind"]:
                info_set = (tuple(sorted(card["rank"] for card in hand)), history)
                self.info_sets[info_set]

    def get_strategy(self, info_set):
        """Regret-matching strategy computation with exploration."""
        strategy = {}
        exploration_chance = 0.05  # 5% chance to explore all actions
        normalizing_sum = 0
        for action, regret in self.info_sets[info_set]["regret_sum"].items():
            strategy[action] = max(regret, 0) + exploration_chance
            normalizing_sum += strategy[action]
        if normalizing_sum > 0:
            for action in strategy:
                strategy[action] /= normalizing_sum
        else:
            for action in ["fold", "call", "raise"]:
                strategy[action] = 1 / 3
        return strategy
    def cfr(self, state, iteration, depth=0, max_depth=10):
        current_player = state["current_player"]
        position = get_player_position(current_player, len(state["players"]))
        hand = canonicalize_hand(state["players"][current_player]["hand"])
        betting_history = tuple(state["betting_history"])
        info_set = (hand, betting_history, position)

        if self.game.is_terminal(state, preflop_only=True) or depth >= max_depth:
            payoff = self.game.get_payoff(state)
            return {action: payoff[current_player] for action in ["fold", "call", "raise"]}

        strategy = self.get_strategy(info_set)
        actions = list(strategy.keys())
        utilities = {}
        node_utility = 0.0

        for action in actions:
            next_state = self.game.apply_action(state, action)
            action_utility = self.cfr(next_state, iteration, depth + 1, max_depth)
            utilities[action] = action_utility[action]
            node_utility += strategy[action] * utilities[action]

        for action in strategy:
            regret = utilities[action] - node_utility
            self.info_sets[info_set]["regret_sum"][action] += regret
            self.info_sets[info_set]["strategy_sum"][action] += strategy[action]

        return {action: node_utility for action in actions}
    def get_average_strategy(self, info_set):
        """Compute average strategy from cumulative strategy sums."""
        strategy_sum = self.info_sets[info_set]["strategy_sum"]
        total_sum = sum(strategy_sum.values())
        if total_sum == 0:
            return {action: 1 / len(strategy_sum) for action in strategy_sum}
        return {action: strategy_sum[action] / total_sum for action in strategy_sum}
