from typing import List, Dict, Tuple
import operator
from functools import total_ordering

@total_ordering
class Card:
    # We'll use prime numbers for ranks to make hand detection easier
    PRIMES = {
        '2': 2, '3': 3, '4': 5, '5': 7, '6': 11, '7': 13, '8': 17,
        '9': 19, '10': 23, 'J': 29, 'Q': 31, 'K': 37, 'A': 41
    }
    
    RANK_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.rank_value = self.RANK_VALUES[rank]
        self.prime = self.PRIMES[rank]
        
    def __eq__(self, other):
        return self.rank_value == other.rank_value
        
    def __lt__(self, other):
        return self.rank_value < other.rank_value
    
    def __repr__(self):
        return f"{self.rank}{self.suit[0]}"

class HandEvaluator:
    # Hand rankings from highest to lowest
    HAND_RANKINGS = {
        'straight_flush': 8,
        'four_of_a_kind': 7,
        'full_house': 6,
        'flush': 5,
        'straight': 4,
        'three_of_a_kind': 3,
        'two_pair': 2,
        'one_pair': 1,
        'high_card': 0
    }
    
    def __init__(self):
        # Precompute straight patterns for quick lookup
        self.straight_patterns = self._generate_straight_patterns()
    
    def _generate_straight_patterns(self) -> set:
        """Generate all possible products of prime numbers that represent straights."""
        patterns = set()
        # Regular straights
        for i in range(2, 11):  # 2-6, 3-7, ..., 10-A
            pattern = 1
            for j in range(5):
                pattern *= Card.PRIMES[str(i + j)]
            patterns.add(pattern)
        # Special case: Ace-low straight (A,2,3,4,5)
        ace_low = Card.PRIMES['A'] * Card.PRIMES['2'] * Card.PRIMES['3'] * \
                 Card.PRIMES['4'] * Card.PRIMES['5']
        patterns.add(ace_low)
        return patterns

    def evaluate_hand(self, hole_cards: List[Dict], community_cards: List[Dict]) -> Tuple[int, List[Card]]:
        """
        Evaluate the best 5-card poker hand from hole cards and community cards.
        Returns (hand_strength, best_five_cards)
        """
        # Convert dictionary cards to Card objects
        cards = [Card(card['rank'], card['suit']) for card in hole_cards + community_cards]
        
        # Get all possible 5-card combinations
        best_hand = None
        best_hand_rank = -1
        best_hand_value = -1
        
        # Check for flush first (most efficient)
        flush_hand = self._find_flush(cards)
        if flush_hand:
            # Check if it's a straight flush
            straight_value = self._is_straight(flush_hand)
            if straight_value:
                return (self.HAND_RANKINGS['straight_flush'] * 1000000 + straight_value,
                       sorted(flush_hand, reverse=True)[:5])
            best_hand = flush_hand
            best_hand_rank = self.HAND_RANKINGS['flush']
            best_hand_value = self._calculate_flush_value(flush_hand)
        
        # Get frequency of each rank
        rank_freq = {}
        for card in cards:
            rank_freq[card.rank_value] = rank_freq.get(card.rank_value, 0) + 1
        
        # Check for four of a kind
        four_kind = self._find_four_of_a_kind(cards, rank_freq)
        if four_kind and (best_hand_rank < self.HAND_RANKINGS['four_of_a_kind']):
            return (self.HAND_RANKINGS['four_of_a_kind'] * 1000000 + 
                   self._calculate_quads_value(four_kind),
                   four_kind)
        
        # Check for full house
        full_house = self._find_full_house(cards, rank_freq)
        if full_house and (best_hand_rank < self.HAND_RANKINGS['full_house']):
            return (self.HAND_RANKINGS['full_house'] * 1000000 + 
                   self._calculate_full_house_value(full_house),
                   full_house)
        
        # Check for straight if we haven't found a flush
        if not best_hand:
            straight = self._find_straight(cards)
            if straight:
                return (self.HAND_RANKINGS['straight'] * 1000000 + 
                       self._calculate_straight_value(straight),
                       straight)
        
        # Check for three of a kind
        three_kind = self._find_three_of_a_kind(cards, rank_freq)
        if three_kind and (best_hand_rank < self.HAND_RANKINGS['three_of_a_kind']):
            return (self.HAND_RANKINGS['three_of_a_kind'] * 1000000 + 
                   self._calculate_three_kind_value(three_kind),
                   three_kind)
        
        # Check for two pair
        two_pair = self._find_two_pair(cards, rank_freq)
        if two_pair and (best_hand_rank < self.HAND_RANKINGS['two_pair']):
            return (self.HAND_RANKINGS['two_pair'] * 1000000 + 
                   self._calculate_two_pair_value(two_pair),
                   two_pair)
        
        # Check for one pair
        one_pair = self._find_pair(cards, rank_freq)
        if one_pair and (best_hand_rank < self.HAND_RANKINGS['one_pair']):
            return (self.HAND_RANKINGS['one_pair'] * 1000000 + 
                   self._calculate_pair_value(one_pair),
                   one_pair)
        
        # If we've found a flush earlier, return it now
        if best_hand:
            return (best_hand_rank * 1000000 + best_hand_value, best_hand)
        
        # Otherwise, return high card
        high_cards = sorted(cards, reverse=True)[:5]
        return (self._calculate_high_card_value(high_cards), high_cards)

    def _find_flush(self, cards: List[Card]) -> List[Card]:
        """Find the highest flush if it exists."""
        suit_counts = {'hearts': [], 'diamonds': [], 'clubs': [], 'spades': []}
        for card in cards:
            suit_counts[card.suit].append(card)
        
        for suit_cards in suit_counts.values():
            if len(suit_cards) >= 5:
                return sorted(suit_cards, reverse=True)[:5]
        return None

    def _is_straight(self, cards: List[Card]) -> int:
        """Check if cards form a straight. Returns highest card value if true."""
        if len(cards) < 5:
            return 0
            
        # Calculate product of prime numbers
        product = 1
        for card in cards[:5]:  # Use only first 5 cards
            product *= card.prime
            
        if product in self.straight_patterns:
            # Handle Ace-low straight specially
            if product == (Card.PRIMES['A'] * Card.PRIMES['2'] * Card.PRIMES['3'] * 
                         Card.PRIMES['4'] * Card.PRIMES['5']):
                return 5  # Return 5 as highest card for A-5 straight
            return max(card.rank_value for card in cards[:5])
        return 0

    def _find_four_of_a_kind(self, cards: List[Card], rank_freq: Dict) -> List[Card]:
        """Find four of a kind if it exists."""
        for rank, freq in rank_freq.items():
            if freq == 4:
                quads = [card for card in cards if card.rank_value == rank]
                kicker = max([card for card in cards if card.rank_value != rank],
                           key=operator.attrgetter('rank_value'))
                return quads + [kicker]
        return None

    def _find_full_house(self, cards: List[Card], rank_freq: Dict) -> List[Card]:
        """Find the highest full house if it exists."""
        three_kind_rank = None
        pair_rank = None
        
        # Find highest three of a kind
        for rank, freq in sorted(rank_freq.items(), reverse=True):
            if freq >= 3 and three_kind_rank is None:
                three_kind_rank = rank
            elif freq >= 2 and pair_rank is None and rank != three_kind_rank:
                pair_rank = rank
                
        if three_kind_rank is not None and pair_rank is not None:
            three_kind = [card for card in cards if card.rank_value == three_kind_rank][:3]
            pair = [card for card in cards if card.rank_value == pair_rank][:2]
            return three_kind + pair
        return None

    def _find_straight(self, cards: List[Card]) -> List[Card]:
        """Find the highest straight if it exists."""
        unique_values = sorted(set(card.rank_value for card in cards), reverse=True)
        
        # Check for Ace-low straight specially
        if 14 in unique_values and 2 in unique_values:  # If we have Ace and 2
            ace_low = [2, 3, 4, 5, 14]
            if all(value in unique_values for value in ace_low[:-1]):
                straight_cards = []
                for value in ace_low:
                    # Find first card of each required rank
                    for card in cards:
                        if card.rank_value == value:
                            straight_cards.append(card)
                            break
                return straight_cards
        
        # Check for regular straights
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i + 4] == 4:
                straight_values = unique_values[i:i + 5]
                straight_cards = []
                for value in straight_values:
                    # Find first card of each required rank
                    for card in cards:
                        if card.rank_value == value:
                            straight_cards.append(card)
                            break
                return straight_cards
        return None

    def _calculate_flush_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a flush hand."""
        value = 0
        for i, card in enumerate(sorted(cards, reverse=True)[:5]):
            value += card.rank_value * (14 ** (4 - i))
        return value

    def _calculate_straight_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a straight hand."""
        return max(card.rank_value for card in cards)

    def _calculate_quads_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a four of a kind hand."""
        quad_rank = next(card.rank_value for card in cards[:4])
        kicker = cards[4].rank_value
        return quad_rank * 14 + kicker

    def _calculate_full_house_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a full house hand."""
        three_kind_rank = cards[0].rank_value
        pair_rank = cards[3].rank_value
        return three_kind_rank * 14 + pair_rank

    def _calculate_three_kind_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a three of a kind hand."""
        value = 0
        trips_rank = cards[0].rank_value
        kickers = sorted([card.rank_value for card in cards[3:]], reverse=True)
        value = trips_rank * (14 ** 2)
        for i, kicker in enumerate(kickers):
            value += kicker * (14 ** (1 - i))
        return value

    def _calculate_two_pair_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a two pair hand."""
        high_pair = cards[0].rank_value
        low_pair = cards[2].rank_value
        kicker = cards[4].rank_value
        return (high_pair * 14 * 14) + (low_pair * 14) + kicker

    def _calculate_pair_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a one pair hand."""
        value = 0
        pair_rank = cards[0].rank_value
        kickers = sorted([card.rank_value for card in cards[2:]], reverse=True)
        value = pair_rank * (14 ** 3)
        for i, kicker in enumerate(kickers):
            value += kicker * (14 ** (2 - i))
        return value

    def _calculate_high_card_value(self, cards: List[Card]) -> int:
        """Calculate a unique value for a high card hand."""
        value = 0
        for i, card in enumerate(sorted(cards, reverse=True)[:5]):
            value += card.rank_value * (14 ** (4 - i))
        return value