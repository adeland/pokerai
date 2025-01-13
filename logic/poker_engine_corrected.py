import random
from itertools import combinations
from hand_evaluator import HandEvaluator

class PokerEngine:
    def __init__(self, num_players=6, small_blind=25, big_blind=50):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = self.initialize_deck()
        self.players = [{"stack": 5000, "hand": [], "active": True} for _ in range(num_players)]
        self.pot = 0
        self.betting_history = []
        self.current_dealer = 0
        self.current_player = None
        self.current_bet = 0  # Track the highest bet in the current round
        self.pot_contributions = [0] * self.num_players  # Track each player's pot contribution
        self.hand_evaluator = HandEvaluator()

    def initialize_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [{"rank": r, "suit": s} for r in ranks for s in suits]

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal_hands(self):
        for player in self.players:
            if player["active"]:
                player["hand"] = [self.deck.pop(), self.deck.pop()]

    def reset(self):
        self.deck = self.initialize_deck()
        self.shuffle_deck()
        self.betting_history = []
        self.current_bet = 0
        self.pot_contributions = [0] * self.num_players
        for player in self.players:
            player["hand"] = []
            player["active"] = True
        self.pot = 0
        self.current_player = (self.current_dealer + 1) % self.num_players

    def post_blinds(self):
        small_blind_position = (self.current_dealer + 1) % self.num_players
        big_blind_position = (self.current_dealer + 2) % self.num_players

        self.players[small_blind_position]["stack"] -= self.small_blind
        self.pot_contributions[small_blind_position] += self.small_blind
        self.pot += self.small_blind
        self.betting_history.append("small_blind")

        self.players[big_blind_position]["stack"] -= self.big_blind
        self.pot_contributions[big_blind_position] += self.big_blind
        self.pot += self.big_blind
        self.betting_history.append("big_blind")

        self.current_player = (big_blind_position + 1) % self.num_players

    def rotate_dealer(self):
        self.current_dealer = (self.current_dealer + 1) % self.num_players

    def assign_positions(self):
        """Assign positions to players based on dealer."""
        positions = ["small_blind", "big_blind"]
        for i in range(2, self.num_players):
            if i == self.num_players - 1:
                positions.append("late")
            elif i == 2:
                positions.append("middle")
            else:
                positions.append("early")
        return positions

    def initial_state(self):
        self.reset()
        self.rotate_dealer()
        self.post_blinds()
        self.deal_hands()
        positions = self.assign_positions()
        return {
            "players": self.players,
            "betting_history": tuple(self.betting_history),
            "pot": self.pot,
            "current_player": self.current_player,
            "positions": positions
        }

    def is_terminal(self, state, preflop_only=False):
        active_players = [p for p in state["players"] if p["active"]]
        if preflop_only:
            return len(active_players) <= 1 or "call" in state["betting_history"]
        return False

    def apply_action(self, state, action):
        current_player = state["current_player"]
        if action == "fold":
            state["players"][current_player]["active"] = False
        elif action == "call":
            amount_to_call = self.current_bet - self.pot_contributions[current_player]
            self.pot_contributions[current_player] += amount_to_call
            self.pot += amount_to_call
        elif action == "raise":
            raise_amount = self.current_bet + 50  # Arbitrary raise amount
            self.current_bet = raise_amount
            self.pot_contributions[current_player] += raise_amount
            self.pot += raise_amount
        state["betting_history"] = tuple(list(state["betting_history"]) + [action])
        state["current_player"] = self.get_next_active_player(current_player, state["players"])
        return state

    def get_next_active_player(self, current_player, players):
        next_player = (current_player + 1) % self.num_players
        while not players[next_player]["active"]:
            next_player = (next_player + 1) % self.num_players
        return next_player

    def get_payoff(self, state):
        """Calculate payoffs based on preflop equity."""
        active_players = [p for p in state["players"] if p["active"]]

        if len(active_players) == 1:
            winner = active_players[0]
            return [self.pot if p == winner else 0 for p in state["players"]]

        # Evaluate equity for showdown
        equities = [self.calculate_equity(p["hand"], opponents=len(active_players) - 1) for p in active_players]
        total_equity = sum(equities)

        payoffs = [0] * len(state["players"])
        for i, player in enumerate(state["players"]):
            if player in active_players:
                payoff_share = (equities[active_players.index(player)] / total_equity) * self.pot
                payoffs[i] = payoff_share

        return payoffs

    def calculate_equity(self, hand, opponents=1, iterations=1000):
        """
        Estimate preflop equity using Monte Carlo simulation with accurate hand evaluation.
        """
        deck = self.initialize_deck()
        # Remove hole cards from deck
        for card in hand:
            deck.remove(card)
        
        wins = 0
        for _ in range(iterations):
            random.shuffle(deck)
            # Deal opponent hands
            opponent_hands = [deck[i:i+2] for i in range(0, opponents * 2, 2)]
            # Deal community cards
            community_cards = deck[opponents * 2:opponents * 2 + 5]
            
            # Evaluate hero hand
            hero_strength = self.evaluate_postflop_strength(hand, community_cards)
            
            # Evaluate opponent hands
            opponent_strengths = [
                self.evaluate_postflop_strength(opp_hand, community_cards)
                for opp_hand in opponent_hands
            ]
            
            # Count wins and ties
            if hero_strength > max(opponent_strengths):
                wins += 1
            elif hero_strength == max(opponent_strengths):
                wins += 0.5  # Split pot
                
        return wins / iterations

    def evaluate_hands(self, hands):
        """Evaluate hand strength, favoring suited hands."""
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                 '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        hand_values = []
        for hand in hands:
            hand_rank = sum(ranks[card["rank"]] for card in hand)
            suited_bonus = 2 if hand[0]["suit"] == hand[1]["suit"] else 0
            hand_values.append(hand_rank + suited_bonus)
        return hand_values

def evaluate_postflop_strength(self, hand, community_cards):
    """
    Evaluate hand strength using the new hand evaluator.
    Returns a numerical value that can be used to compare hands.
    Higher values indicate stronger hands.
    """
    hand_strength, best_five = self.hand_evaluator.evaluate_hand(hand, community_cards)
    return hand_strength
