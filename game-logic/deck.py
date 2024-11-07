import random

class Deck:
    def __init__(self):
        self.cards = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "CDHS"]
        random.shuffle(self.cards)

    def deal(self, num_cards):
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def reset(self):
        self.__init__()

