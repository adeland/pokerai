from game_state import GameState
from player import Player

def initialize_game():
    # Create initial game state
    players = [
        Player(stack = 100, position = "SB"),
        Player(stack = 100, position = "BB"),
        Player(stack = 100, position = "UTG"),
        Player(stack = 100, position = "MP"),
        Player(stack = 100, position = "CO"),
        Player(stack = 100, position = "BTN"),
    ]
    game_state = GameState(pot = 0, board = [], players = players)
    return game_state

if __name__ == "__main__":
    game_state = initialize_game()
    print("Initial Game State:", game_state)
