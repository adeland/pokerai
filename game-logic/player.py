class Player:
    def __init__(self, stack, position, hand = None):
        self.stack = stack  # Player's stack size
        self.position = position  # Position (e.g., SB, BB, UTG)
        self.hand = hand  # Player's hand, initialized as None

    def __repr__(self):
        return f"Player(stack = {self.stack}, position = {self.position}, hand = {self.hand})"
