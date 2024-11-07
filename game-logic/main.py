from game_state import GameState
from player import Player
from cfr import CFRSolver

def initialize_game():
    players = [
        Player(stack=100, position="SB"),
        Player(stack=100, position="BB"),
        Player(stack=100, position="UTG"),
        Player(stack=100, position="MP"),
        Player(stack=100, position="CO"),
        Player(stack=100, position="BTN"),
    ]
    game_state = GameState(players=players)
    return game_state

if __name__ == "__main__":
    game_state = initialize_game()
    print("Initial Game State:", game_state)

    game_state.play_round()

    # Placeholder for CFRSolver integration
    # cfr_solver = CFRSolver(game_state)
    # cfr_solver.train(iterations=1000)

