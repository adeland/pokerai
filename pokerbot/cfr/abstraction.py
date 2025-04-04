"""
Game abstraction utilities for reducing the complexity of the poker game.
"""

import numpy as np

from ..game_engine.card import Card

class Abstraction:
    """
    A class for creating abstractions of the poker game to reduce complexity.
    
    This includes card abstractions (grouping similar hands) and action abstractions
    (limiting the action space to a manageable size).
    """
    
    @staticmethod
    def create_card_abstraction(hole_cards, community_cards=None):
        """
        Create a card abstraction key for the given cards.
        
        For preflop, we use card ranks and suits directly.
        For postflop, we use a simplified strength-based bucketing.
        
        Args:
            hole_cards (list): List of Card objects representing hole cards
            community_cards (list, optional): List of Card objects representing community cards
            
        Returns:
            str: A string key representing the abstracted cards
        """
        if not community_cards:
            # Preflop abstraction - use card ranks and suits
            # Sort cards by rank for consistent representation
            sorted_cards = sorted(hole_cards, key=lambda card: card.rank, reverse=True)
            
            # Check if suited
            is_suited = sorted_cards[0].suit == sorted_cards[1].suit
            
            # Create key based on ranks and whether suited
            if is_suited:
                return f"{sorted_cards[0].rank}s{sorted_cards[1].rank}"
            else:
                return f"{sorted_cards[0].rank}o{sorted_cards[1].rank}"
        else:
            # Postflop abstraction - use a simplified strength-based bucketing
            # This is a very basic abstraction; more sophisticated methods would be used in practice
            
            # Calculate a simple hand strength score
            strength = Abstraction._calculate_hand_strength(hole_cards, community_cards)
            
            # Create buckets based on hand strength
            if strength < 0.2:
                bucket = "very_weak"
            elif strength < 0.4:
                bucket = "weak"
            elif strength < 0.6:
                bucket = "medium"
            elif strength < 0.8:
                bucket = "strong"
            else:
                bucket = "very_strong"
            
            # Include the betting round in the key
            round_name = Abstraction._determine_round(community_cards)
            
            return f"{round_name}_{bucket}"
    
    @staticmethod
    def _calculate_hand_strength(hole_cards, community_cards):
        """
        Calculate a simplified hand strength score.
        
        This is a placeholder for a more sophisticated hand strength calculation.
        In a real implementation, this would use the HandEvaluator and Monte Carlo simulations.
        
        Args:
            hole_cards (list): List of Card objects representing hole cards
            community_cards (list): List of Card objects representing community cards
            
        Returns:
            float: A hand strength score between 0 and 1
        """
        # This is a very simplified placeholder
        # In a real implementation, we would:
        # 1. Evaluate the current hand strength
        # 2. Run Monte Carlo simulations to estimate potential
        # 3. Consider drawing possibilities
        
        # For now, we'll just use a simple heuristic based on card ranks
        total_rank = sum(card.rank for card in hole_cards) + sum(card.rank for card in community_cards)
        max_possible_rank = 14 * 2 + 14 * len(community_cards)  # Maximum possible rank sum
        
        # Normalize to [0, 1]
        return total_rank / max_possible_rank
    
    @staticmethod
    def _determine_round(community_cards):
        """
        Determine the current betting round based on community cards.
        
        Args:
            community_cards (list): List of Card objects representing community cards
            
        Returns:
            str: The current betting round ('flop', 'turn', or 'river')
        """
        if len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) == 5:
            return "river"
        else:
            return "preflop"
    
    @staticmethod
    def create_action_abstraction(game_state, available_actions):
        """
        Create an action abstraction to limit the action space.
        
        Args:
            game_state: The current game state
            available_actions (list): List of available actions
            
        Returns:
            list: A list of abstracted actions
        """
        # For simplicity, we'll use a basic action abstraction
        # In a real implementation, this would be more sophisticated
        
        abstracted_actions = []
        
        for action in available_actions:
            if action == "fold":
                abstracted_actions.append(("fold", 0))
            
            elif action == "check" or action == "call":
                abstracted_actions.append((action, 0))
            
            elif action == "bet" or action == "raise":
                # Create a few bet/raise sizes based on the pot
                pot = game_state.pot
                big_blind = game_state.big_blind
                
                # Pot-based sizing
                abstracted_actions.append((action, big_blind * 2))  # Min-bet/raise
                abstracted_actions.append((action, pot // 2))       # Half pot
                abstracted_actions.append((action, pot))            # Pot-sized
                abstracted_actions.append((action, pot * 2))        # 2x pot
                
                # All-in is always an option
                current_player = game_state.players[game_state.current_player_idx]
                abstracted_actions.append((action, current_player.stack))
        
        return abstracted_actions
