class CFRSolver:
    def __init__(self, game_state):
        self.game_state = game_state
        self.regret_sum = {}
        self.strategy_sum = {}
        self.node_utilities = {}

    def cfr(self, game_state, player, probability):
        if game_state.is_game_over():
            return self.calculate_terminal_utility(game_state, player)

        node_id = self.get_node_id(game_state, player)
        available_actions = self.get_available_actions(game_state, player)

        if node_id not in self.regret_sum:
            self.regret_sum[node_id] = {action: 0 for action in available_actions}
            self.strategy_sum[node_id] = {action: 0 for action in available_actions}

        strategy = self.compute_strategy(node_id)
        self.strategy_sum[node_id] = {
            action: self.strategy_sum[node_id][action] + strategy[action] * probability
            for action in available_actions
        }

        util = {action: 0 for action in available_actions}
        node_utility = 0

        for action in available_actions:
            new_game_state = self.simulate_action(game_state, player, action)
            next_player = (player + 1) % len(game_state.players)
            util[action] = -self.cfr(new_game_state, next_player, probability * strategy[action])
            node_utility += strategy[action] * util[action]

        for action in available_actions:
            regret = util[action] - node_utility
            self.regret_sum[node_id][action] += regret

        return node_utility

    def compute_strategy(self, node_id):
        regrets = self.regret_sum[node_id]
        positive_regrets = {k: max(v, 0) for k, v in regrets.items()}
        total_positive_regret = sum(positive_regrets.values())
        if total_positive_regret > 0:
            return {k: v / total_positive_regret for k, v in positive_regrets.items()}
        else:
            return {k: 1 / len(positive_regrets) for k in positive_regrets}

    def get_node_id(self, game_state, player):
        return f"{player}-{tuple(game_state.board)}-{game_state.pot}-{game_state.current_bet}"

    def get_available_actions(self, game_state, player):
        return ["fold", "call", "bet", "raise", "check"]

    def simulate_action(self, game_state, player, action):
        new_game_state = game_state.copy()
        if action == "fold":
            new_game_state.fold(new_game_state.players[player])
        elif action == "call":
            new_game_state.call(new_game_state.players[player])
        elif action == "bet":
            new_game_state.place_bet(new_game_state.players[player], amount=10)
        elif action == "check":
            new_game_state.check(new_game_state.players[player])
        return new_game_state

    def calculate_terminal_utility(self, game_state, player):
        active_players = [p for p in game_state.players if not p.is_folded]
        if len(active_players) == 1:
            return game_state.pot if active_players[0] == player else -game_state.pot
        return 0

