class GameState:

    def __init__(self, pot = 0, board = None, players = None):
        self.pot = pot  # Pot size at each game state
        self.board = board if board else []  # List to hold board cards
        self.players = players if players else []  # List of player objects

    def __repr__(self):
        return f"GameState(pot = {self.pot}, board = {self.board}, players = {self.players})"

#Basic game betting functions
def add_to_pot(self, amount):
    self.pot += amount
#
def place_bet(self, player, amount):
    if player.stack >= amount:
        player.stack -= amount
        self.add_to_pot(amount)
    else:
        raise ValueError("Player cannot bet more than their stack")


def evaluate_hand(self, player_hand, board):
    # Placeholder function to compare hands
    #TO-DO
    pass
    return "Ranking based on basic criteria"



