"""
Bot player implementation for poker games.
This module provides a bot player that uses trained CFR strategies and depth-limited search.
"""

import random
import os
import sys

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- ADD THIS IMPORT ---
from organized_poker_bot.game_engine.player import Player
# --- END IMPORT ---

# Use absolute imports that work when run directly
from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
from organized_poker_bot.bot.depth_limited_search import DepthLimitedSearch

# --- MODIFY CLASS DEFINITION TO INHERIT FROM Player ---
class BotPlayer(Player):
    """
    A bot player for poker games. Inherits from Player.

    This class implements a poker bot that can use either a pre-trained CFR strategy
    or real-time depth-limited search to make decisions.

    Attributes:
        strategy_obj: The CFRStrategy object to use
        use_depth_limited_search: Whether to use depth-limited search
        search_depth: How many betting rounds to look ahead in depth-limited search
        search_iterations: Number of iterations for Monte Carlo simulations
        # Inherited attributes like name, stack, position, etc., from Player
    """

    # --- MODIFY __init__ ---
    def __init__(self, strategy, name="Bot", stack=10000, use_depth_limited_search=True, search_depth=1, search_iterations=1000):
        """
        Initialize the bot player.

        Args:
            strategy: The CFR strategy to use (can be a dictionary or CFRStrategy object)
            name (str): Name for the bot player.
            stack (int): Initial stack for the bot player.
            use_depth_limited_search: Whether to use depth-limited search
            search_depth: How many betting rounds to look ahead in depth-limited search
            search_iterations: Number of iterations for Monte Carlo simulations
        """
        # --- CALL PARENT Player CONSTRUCTOR ---
        # Make sure Player.__init__ accepts is_human/is_random if you applied the previous fix
        super().__init__(name=name, stack=stack, is_human=False, is_random=False)
        # --- END PARENT CALL ---

        # Initialize BotPlayer specific attributes
        # Use self.strategy_obj to avoid potential name clashes if Player class had a 'strategy' attribute
        if isinstance(strategy, dict):
            # If strategy is passed as a dict, load it into a CFRStrategy object
            self.strategy_obj = CFRStrategy()
            self.strategy_obj.strategy = strategy
        elif isinstance(strategy, CFRStrategy):
             self.strategy_obj = strategy # Use the passed strategy object directly
        else:
            # Handle cases where the strategy might not be loaded correctly yet
            # Or raise an error if the type is unexpected
            print(f"Warning: Unexpected strategy type {type(strategy)}. Initializing empty strategy.")
            self.strategy_obj = CFRStrategy()


        self.use_depth_limited_search = use_depth_limited_search
        self.search_depth = search_depth
        self.search_iterations = search_iterations

        # Initialize depth-limited search if enabled
        # Ensure strategy_obj is valid before initializing DLS
        if self.use_depth_limited_search and hasattr(self, 'strategy_obj'):
            self.dls = DepthLimitedSearch(
                self.strategy_obj, # Use the strategy object here
                search_depth=self.search_depth,
                num_iterations=self.search_iterations
            )
        elif self.use_depth_limited_search:
             print(f"Warning: Could not initialize DLS for {self.name} due to missing strategy object.")
             self.dls = None # Explicitly set dls to None
    # --- END MODIFIED __init__ ---

    def get_action(self, game_state, player_idx):
        """
        Get the best action for the current game state.

        Args:
            game_state: The current game state
            player_idx: The player index (Note: BotPlayer itself now holds its state via inheritance)

        Returns:
            tuple or str: The chosen action (either a tuple of (action_type, amount) or a string)
        """
        # Use depth-limited search if enabled and we're in a real game
        if self.use_depth_limited_search and hasattr(self, 'dls') and self.dls is not None and not game_state.is_terminal():
            try:
                 return self.dls.get_action(game_state, player_idx)

            except Exception as e:
                # Fallback to pre-trained strategy if DLS fails
                print(f"DLS failed for {self.name}: {e}, falling back to pre-trained strategy")
                # Ensure strategy_obj exists before calling its method
                if hasattr(self, 'strategy_obj'):
                    return self.strategy_obj.get_action(game_state, player_idx)
                else:
                    print(f"Error: {self.name} has no strategy object to fall back on.")
                    # Define a safe fallback action if strategy object is missing
                    available_actions = game_state.get_available_actions()
                    return "check" if "check" in available_actions else "fold"


        # Otherwise, use the pre-trained strategy
        # Ensure strategy_obj exists before calling its method
        if hasattr(self, 'strategy_obj'):
            return self.strategy_obj.get_action(game_state, player_idx)
        else:
            print(f"Error: {self.name} has no strategy object.")
            # Define a safe fallback action if strategy object is missing
            available_actions = game_state.get_available_actions()
            return "check" if "check" in available_actions else "fold"


    def update_strategy(self, new_strategy):
        """
        Update the bot's strategy.

        Args:
            new_strategy: The new strategy to use (dict or CFRStrategy object)
        """
        if isinstance(new_strategy, dict):
            self.strategy_obj = CFRStrategy()
            self.strategy_obj.strategy = new_strategy
        elif isinstance(new_strategy, CFRStrategy):
             self.strategy_obj = new_strategy
        else:
             print(f"Warning: Cannot update strategy for {self.name} with type {type(new_strategy)}")
             return # Or raise error


        # Update depth-limited search if enabled
        if self.use_depth_limited_search:
             if hasattr(self, 'strategy_obj'):
                 self.dls = DepthLimitedSearch(
                     self.strategy_obj, # Use the updated strategy object
                     search_depth=self.search_depth,
                     num_iterations=self.search_iterations
                 )
             else:
                  print(f"Warning: Could not update DLS for {self.name} due to missing strategy object.")
                  self.dls = None


    def __str__(self):
        """Return a string representation of the bot player."""
        # Use the name and stack inherited from Player
        strategy_info = "No strategy loaded"
        if hasattr(self, 'strategy_obj') and hasattr(self.strategy_obj, 'strategy') and self.strategy_obj.strategy:
             strategy_info = f"{len(self.strategy_obj.strategy)} info sets"

        return f"BotPlayer(name={self.name}, stack={self.stack}, strategy={strategy_info}, use_dls={self.use_depth_limited_search})"

# ... (End of BotPlayer class)
