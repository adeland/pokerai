class Player:
    def __init__(self, stack, position, hand=None):
        self.stack = stack  # Player's stack size
        self.position = position  # Position (e.g., SB, BB, UTG)
        self.hand = hand  # Player's hand, initialized as None
        self.current_bet = 0  # Tracks the current bet by this player in the current round
        self.is_folded = False  # Tracks if player has folded
        self.is_all_in = False  # Tracks if player is all-in

    def reset_for_new_round(self):
        self.current_bet = 0
        self.is_folded = False
        self.is_all_in = False

    def __repr__(self):
        return (f"Player(stack={self.stack}, position={self.position}, "
                f"current_bet={self.current_bet}, is_folded={self.is_folded}, "
                f"is_all_in={self.is_all_in}, hand={self.hand})")

