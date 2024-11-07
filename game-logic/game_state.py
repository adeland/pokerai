from actions import Action
from player import Player
from deck import Deck
from collections import Counter

class GameState:
    def __init__(self, players):
        self.pot = 0
        self.side_pots = []
        self.board = []
        self.players = players
        self.current_bet = 0
        self.previous_bet = 0
        self.deck = Deck()
        self.current_player_index = 0

    def __repr__(self):
        players_info = "\n  ".join(str(player) for player in self.players)
        side_pots_info = ", ".join(str(pot) for pot in self.side_pots)
        return (f"GameState(\n"
                f"  Pot: {self.pot}\n"
                f"  Side Pots: [{side_pots_info}]\n"
                f"  Board: {self.board}\n"
                f"  Current Bet: {self.current_bet}\n"
                f"  Players:\n  {players_info}\n)")

    def reset_for_new_hand(self):
        self.pot = 0
        self.side_pots = []
        self.board = []
        self.current_bet = 0
        self.previous_bet = 0
        self.deck = Deck()
        for player in self.players:
            player.reset_for_new_round()

    def deal_hands(self):
        for player in self.players:
            player.hand = self.deck.deal(2)
        print("Hands dealt:", {player.position: player.hand for player in self.players})

    def place_blinds(self):
        sb_amount = 1
        bb_amount = 2
        self.players[0].current_bet = sb_amount
        self.players[0].stack -= sb_amount
        self.players[1].current_bet = bb_amount
        self.players[1].stack -= bb_amount
        self.pot += sb_amount + bb_amount
        self.current_bet = bb_amount

    def add_to_pot(self, amount):
        self.pot += amount

    def player_action(self, player, action, amount=0):
        if player.is_folded or player.is_all_in:
            return  # Skip players who are folded or all-in

        if action == Action.FOLD:
            self.fold(player)
        elif action == Action.CHECK:
            if player.current_bet == self.current_bet:
                self.check(player)
            else:
                print(f"Invalid check by {player.position}: must call or fold.")
                return  # Prevent further looping when check is not valid
        elif action == Action.CALL:
            if player.current_bet < self.current_bet:
                self.call(player)
            else:
                print(f"{player.position} cannot call: already at current bet.")
        elif action in [Action.BET, Action.RAISE]:
            if amount > 0 and amount >= (self.current_bet - player.current_bet):
                self.place_bet(player, amount)
            else:
                print(f"Invalid {action.name.lower()} amount by {player.position}.")

    def fold(self, player):
        player.is_folded = True
        print(f"{player.position} folds")

    def check(self, player):
        if player.current_bet == self.current_bet:
            print(f"{player.position} checks")
        else:
            print("Invalid check: there is a bet to call")

    def call(self, player):
        amount_to_call = self.current_bet - player.current_bet
        if player.stack <= amount_to_call:
            amount_to_call = player.stack
            player.is_all_in = True
        player.stack -= amount_to_call
        player.current_bet += amount_to_call
        self.pot += amount_to_call
        print(f"{player.position} calls {amount_to_call}")

    def place_bet(self, player, amount):
        if amount > player.stack:
            print("Invalid bet: not enough chips")
            return
        player.stack -= amount
        player.current_bet += amount
        self.current_bet = max(self.current_bet, player.current_bet)
        self.pot += amount
        print(f"{player.position} bets {amount}")

    def betting_round(self, start_position=0):
        print("Starting betting round...")
        players_in_round = [p for p in self.players if not p.is_folded and not p.is_all_in]
        if len(players_in_round) <= 1:
            return  # No betting needed if one or fewer players are active

        current_index = start_position
        initial_bet = self.current_bet
        acted_players = set()

        while True:
            player = self.players[current_index % len(self.players)]
            if not player.is_folded and not player.is_all_in:
                if player.current_bet < self.current_bet:
                    self.player_action(player, Action.CALL)
                else:
                    self.player_action(player, Action.CHECK)
                acted_players.add(player)

            current_index += 1

            if self.is_betting_round_complete(players_in_round, initial_bet, acted_players):
                break

        # Reset the current bet after the round is complete
        self.current_bet = 0

    def is_betting_round_complete(self, players_in_round, initial_bet, acted_players):
        highest_bet = max(p.current_bet for p in players_in_round)
        return all(p.current_bet == highest_bet or p.is_all_in for p in players_in_round) and len(acted_players) >= len(players_in_round)

    def play_round(self):
        self.reset_for_new_hand()
        self.place_blinds()
        self.deal_hands()

        print("\n--- Pre-flop ---")
        self.betting_round(start_position=2)

        if not self.is_game_over():
            self.deal_flop()
            print("\n--- Flop ---")
            self.reset_players_for_new_round()
            self.betting_round()

        if not self.is_game_over():
            self.deal_turn()
            print("\n--- Turn ---")
            self.reset_players_for_new_round()
            self.betting_round()

        if not self.is_game_over():
            self.deal_river()
            print("\n--- River ---")
            self.reset_players_for_new_round()
            self.betting_round()

        if not self.is_game_over():
            self.showdown()

    def deal_flop(self):
        self.board.extend(self.deck.deal(3))
        print("Flop:", self.board)

    def deal_turn(self):
        self.board.append(self.deck.deal(1)[0])
        print("Turn:", self.board[-1])

    def deal_river(self):
        self.board.append(self.deck.deal(1)[0])
        print("River:", self.board[-1])

    def is_game_over(self):
        active_players = [p for p in self.players if not p.is_folded]
        return len(active_players) <= 1

    def showdown(self):
        print("\n--- Showdown ---")
        print("Board:", self.board)
        active_players = [p for p in self.players if not p.is_folded]
        if not active_players:
            print("All players folded. No showdown.")
            return

        hands = {player: self.evaluate_hand(player.hand + self.board) for player in active_players}
        winner = max(hands, key=hands.get)
        winning_hand = hands[winner]

        hand_ranks = {
            8: "Straight Flush",
            7: "Four of a Kind",
            6: "Full House",
            5: "Flush",
            4: "Straight",
            3: "Three of a Kind",
            2: "Two Pairs",
            1: "One Pair",
            0: "High Card"
        }

        rank_to_card = {i: rank for i, rank in enumerate('23456789TJQKA', start=2)}

        try:
            if winning_hand[0] == 1:  # One Pair
                pair_rank = rank_to_card[winning_hand[1]]
                kickers = [rank_to_card[k] for k in winning_hand[2:]] if len(winning_hand) > 2 else []
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]} of {pair_rank}s, kicker {' '.join(kickers) if kickers else 'N/A'}")
            elif winning_hand[0] == 2:  # Two Pairs
                pair_rank = rank_to_card[winning_hand[1]]
                second_pair_rank = rank_to_card[winning_hand[2]]
                kicker = rank_to_card[winning_hand[3]] if len(winning_hand) > 3 else 'N/A'
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]} of {pair_rank}s and {second_pair_rank}s, kicker {kicker}")
            elif winning_hand[0] == 3:  # Three of a Kind
                trip_rank = rank_to_card[winning_hand[1]]
                kickers = [rank_to_card[k] for k in winning_hand[2:]] if len(winning_hand) > 2 else []
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]} of {trip_rank}s, kicker {' '.join(kickers) if kickers else 'N/A'}")
            elif winning_hand[0] == 6:  # Full House
                trip_rank = rank_to_card[winning_hand[1]]
                pair_rank = rank_to_card[winning_hand[2]] if len(winning_hand) > 2 else 'N/A'
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]}, {trip_rank}s over {pair_rank}s")
            elif winning_hand[0] == 7:  # Four of a Kind
                four_rank = rank_to_card[winning_hand[1]]
                kicker = rank_to_card[winning_hand[2]] if len(winning_hand) > 2 else 'N/A'
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]} of {four_rank}s, kicker {kicker}")
            else:  # High Card, Straight, Flush, Straight Flush
                print(f"Winner: {winner.position} wins with {hand_ranks[winning_hand[0]]}")
        except KeyError as e:
            print(f"Error displaying winning hand: Invalid rank {e}. Please check hand evaluation logic.")
        except IndexError:
            print("Error: Missing hand details for displaying the winning hand. Please verify hand evaluation logic.")

        self.distribute_pot(winner)

    def distribute_pot(self, winner):
        winner.stack += self.pot
        print(f"{winner.position} wins the pot of {self.pot}")
        self.pot = 0

    def evaluate_hand(self, full_hand):
        ranks = '23456789TJQKA'
        rank_values = {r: i for i, r in enumerate(ranks, start=2)}
        suits = [card[1] for card in full_hand]
        rank_counts = Counter(card[0] for card in full_hand)
        sorted_ranks = sorted((rank_values[rank], rank) for rank in rank_counts.keys())
        sorted_ranks.reverse()

        is_flush = len(set(suits)) == 1
        is_straight = len(rank_counts) == 5 and (sorted_ranks[0][0] - sorted_ranks[-1][0] == 4)
        max_count = max(rank_counts.values())

        if is_straight and is_flush:
            return (8, sorted_ranks[0][0])  # Straight flush
        elif max_count == 4:
            four_of_a_kind = max(rank_counts, key=lambda r: rank_counts[r] == 4)
            return (7, rank_values[four_of_a_kind])  # Four of a kind
        elif max_count == 3 and 2 in rank_counts.values():
            three_of_a_kind = max(rank_counts, key=lambda r: rank_counts[r] == 3)
            pair = max(rank_counts, key=lambda r: rank_counts[r] == 2)
            return (6, rank_values[three_of_a_kind], rank_values[pair])  # Full house
        elif is_flush:
            return (5, [rank_values[r] for r in rank_counts])  # Flush
        elif is_straight:
            return (4, sorted_ranks[0][0])  # Straight
        elif max_count == 3:
            three_of_a_kind = max(rank_counts, key=lambda r: rank_counts[r] == 3)
            return (3, rank_values[three_of_a_kind])  # Three of a kind
        elif list(rank_counts.values()).count(2) == 2:
            pairs = [rank_values[r] for r in rank_counts if rank_counts[r] == 2]
            return (2, max(pairs), min(pairs))  # Two pairs
        elif max_count == 2:
            pair = max(rank_counts, key=lambda r: rank_counts[r] == 2)
            return (1, rank_values[pair])  # One pair
        else:
            return (0, [rank_values[r] for _, r in sorted_ranks])  # High card

    def reset_players_for_new_round(self):
        for player in self.players:
            player.current_bet = 0
            player.is_all_in = False
        self.current_bet = 0
        self.previous_bet = 0

