import unittest
from hand_evaluator import HandEvaluator, Card
from poker_engine_corrected import PokerEngine

class TestHandEvaluator(unittest.TestCase):
    def setUp(self):
        """Initialize the hand evaluator before each test."""
        self.evaluator = HandEvaluator()
        self.engine = PokerEngine()

    def create_cards(self, card_strings):
        """Helper method to create card dictionaries from strings.
        
        Example: ['Ah', 'Kh'] creates [{'rank': 'A', 'suit': 'hearts'}, ...]
        """
        suit_mapping = {
            'h': 'hearts',
            'd': 'diamonds',
            'c': 'clubs',
            's': 'spades'
        }
        
        cards = []
        for card in card_strings:
            rank = card[0] if card[0] != '1' else '10'
            suit = suit_mapping[card[-1]]
            cards.append({'rank': rank, 'suit': suit})
        return cards

    def test_straight_flush(self):
        """Test detection and ranking of straight flushes."""
        # Royal flush
        royal = self.create_cards(['Ah', 'Kh', 'Qh', 'Jh', '10h'])
        # Lower straight flush
        lower = self.create_cards(['9h', '8h', '7h', '6h', '5h'])
        
        royal_strength, _ = self.evaluator.evaluate_hand(royal[:2], royal[2:])
        lower_strength, _ = self.evaluator.evaluate_hand(lower[:2], lower[2:])
        
        self.assertTrue(royal_strength > lower_strength, 
                       "Royal flush should beat lower straight flush")

    def test_four_of_a_kind(self):
        """Test detection and ranking of four of a kind hands."""
        # Aces four of a kind
        quad_aces = self.create_cards(['Ah', 'Ad', 'Ac', 'As', 'Kh'])
        # Kings four of a kind
        quad_kings = self.create_cards(['Kh', 'Kd', 'Kc', 'Ks', 'Ah'])
        
        aces_strength, _ = self.evaluator.evaluate_hand(quad_aces[:2], quad_aces[2:])
        kings_strength, _ = self.evaluator.evaluate_hand(quad_kings[:2], quad_kings[2:])
        
        self.assertTrue(aces_strength > kings_strength,
                       "Four aces should beat four kings")

    def test_full_house(self):
        """Test detection and ranking of full house hands."""
        # Aces full of kings
        aces_full = self.create_cards(['Ah', 'Ad', 'Ac', 'Kh', 'Kd'])
        # Kings full of aces
        kings_full = self.create_cards(['Kh', 'Kd', 'Kc', 'Ah', 'Ad'])
        
        aces_strength, _ = self.evaluator.evaluate_hand(aces_full[:2], aces_full[2:])
        kings_strength, _ = self.evaluator.evaluate_hand(kings_full[:2], kings_full[2:])
        
        self.assertTrue(aces_strength > kings_strength,
                       "Aces full should beat kings full")

    def test_flush(self):
        """Test detection and ranking of flush hands."""
        # Ace-high flush
        ace_flush = self.create_cards(['Ah', 'Kh', 'Qh', 'Jh', '9h'])
        # King-high flush
        king_flush = self.create_cards(['Kh', 'Qh', 'Jh', '10h', '8h'])
        
        ace_strength, _ = self.evaluator.evaluate_hand(ace_flush[:2], ace_flush[2:])
        king_strength, _ = self.evaluator.evaluate_hand(king_flush[:2], king_flush[2:])
        
        self.assertTrue(ace_strength > king_strength,
                       "Ace-high flush should beat king-high flush")

    def test_straight(self):
        """Test detection and ranking of straight hands."""
        # Broadway straight (Ace to Ten)
        broadway = self.create_cards(['Ah', 'Kd', 'Qh', 'Jc', '10s'])
        # Wheel straight (Five to Ace)
        wheel = self.create_cards(['5h', '4d', '3h', '2c', 'As'])
        # Middle straight
        middle = self.create_cards(['8h', '7d', '6h', '5c', '4s'])
        
        broadway_strength, _ = self.evaluator.evaluate_hand(broadway[:2], broadway[2:])
        wheel_strength, _ = self.evaluator.evaluate_hand(wheel[:2], wheel[2:])
        middle_strength, _ = self.evaluator.evaluate_hand(middle[:2], middle[2:])
        
        self.assertTrue(broadway_strength > wheel_strength,
                       "Broadway straight should beat wheel straight")
        self.assertTrue(broadway_strength > middle_strength,
                       "Broadway straight should beat middle straight")

    def test_three_of_a_kind(self):
        """Test detection and ranking of three of a kind hands."""
        # Three aces
        three_aces = self.create_cards(['Ah', 'Ad', 'Ac', 'Kh', 'Qd'])
        # Three kings
        three_kings = self.create_cards(['Kh', 'Kd', 'Kc', 'Ah', 'Qd'])
        
        aces_strength, _ = self.evaluator.evaluate_hand(three_aces[:2], three_aces[2:])
        kings_strength, _ = self.evaluator.evaluate_hand(three_kings[:2], three_kings[2:])
        
        self.assertTrue(aces_strength > kings_strength,
                       "Three aces should beat three kings")

    def test_two_pair(self):
        """Test detection and ranking of two pair hands."""
        # Aces and kings
        aces_kings = self.create_cards(['Ah', 'Ad', 'Kc', 'Kh', 'Qd'])
        # Kings and queens
        kings_queens = self.create_cards(['Kh', 'Kd', 'Qc', 'Qh', 'Ad'])
        
        aces_kings_strength, _ = self.evaluator.evaluate_hand(aces_kings[:2], aces_kings[2:])
        kings_queens_strength, _ = self.evaluator.evaluate_hand(kings_queens[:2], kings_queens[2:])
        
        self.assertTrue(aces_kings_strength > kings_queens_strength,
                       "Aces and kings should beat kings and queens")

    def test_one_pair(self):
        """Test detection and ranking of one pair hands."""
        # Pair of aces
        pair_aces = self.create_cards(['Ah', 'Ad', 'Kc', 'Qh', 'Jd'])
        # Pair of kings
        pair_kings = self.create_cards(['Kh', 'Kd', 'Ac', 'Qh', 'Jd'])
        
        aces_strength, _ = self.evaluator.evaluate_hand(pair_aces[:2], pair_aces[2:])
        kings_strength, _ = self.evaluator.evaluate_hand(pair_kings[:2], pair_kings[2:])
        
        self.assertTrue(aces_strength > kings_strength,
                       "Pair of aces should beat pair of kings")

    def test_high_card(self):
        """Test ranking of high card hands."""
        # Ace high
        ace_high = self.create_cards(['Ah', '2d', '3c', '4h', '5d'])
        # King high
        king_high = self.create_cards(['Kh', 'Qd', 'Jc', '10h', '9d'])
        
        ace_strength, _ = self.evaluator.evaluate_hand(ace_high[:2], ace_high[2:])
        king_strength, _ = self.evaluator.evaluate_hand(king_high[:2], king_high[2:])
        
        self.assertTrue(ace_strength > king_strength,
                       "Ace high should beat king high")

    def test_hand_ranking_order(self):
        """Test that all hand rankings are properly ordered."""
        # Create one of each hand type
        straight_flush = self.create_cards(['Ah', 'Kh', 'Qh', 'Jh', '10h'])
        four_kind = self.create_cards(['Ah', 'Ad', 'Ac', 'As', 'Kh'])
        full_house = self.create_cards(['Ah', 'Ad', 'Ac', 'Kh', 'Kd'])
        flush = self.create_cards(['Ah', 'Kh', 'Qh', 'Jh', '9h'])
        straight = self.create_cards(['Ah', 'Kd', 'Qc', 'Js', '10h'])
        three_kind = self.create_cards(['Ah', 'Ad', 'Ac', 'Kh', 'Qd'])
        two_pair = self.create_cards(['Ah', 'Ad', 'Kh', 'Kd', 'Qc'])
        one_pair = self.create_cards(['Ah', 'Ad', 'Kh', 'Qd', 'Jc'])
        high_card = self.create_cards(['Ah', 'Kd', 'Qc', 'Js', '9h'])
        
        hands = [
            straight_flush, four_kind, full_house, flush, straight,
            three_kind, two_pair, one_pair, high_card
        ]
        
        strengths = []
        for hand in hands:
            strength, _ = self.evaluator.evaluate_hand(hand[:2], hand[2:])
            strengths.append(strength)
        
        # Verify that each hand beats all hands below it
        for i in range(len(strengths) - 1):
            self.assertTrue(strengths[i] > strengths[i + 1],
                          f"Hand ranking order incorrect at index {i}")

    def test_equity_calculation(self):
        """Test the equity calculation functionality."""
        # Test AK suited vs pocket pairs
        ak_suited = self.create_cards(['Ah', 'Kh'])
        pocket_queens = self.create_cards(['Qh', 'Qd'])
        
        # Calculate equity for both hands
        ak_equity = self.engine.calculate_equity(ak_suited, opponents=1, iterations=1000)
        qq_equity = self.engine.calculate_equity(pocket_queens, opponents=1, iterations=1000)
        
        # AK suited should have around 43% equity vs QQ
        self.assertTrue(0.40 <= ak_equity <= 0.46,
                       f"AK suited equity vs QQ should be ~43%, got {ak_equity:.2%}")
        self.assertTrue(0.54 <= qq_equity <= 0.60,
                       f"QQ equity vs AK suited should be ~57%, got {qq_equity:.2%}")

if __name__ == '__main__':
    unittest.main()