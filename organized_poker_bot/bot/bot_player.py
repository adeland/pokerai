# --- START OF FILE organized_poker_bot/bot/bot_player.py ---
"""
Bot player implementation for poker games.
This module provides a bot player that uses trained CFR strategies and depth-limited search.
(Refactored V2: Handle initial_stacks for DLS)
"""

import random
import os
import sys
import numpy as np # Import numpy for safe checking

# Add the parent directory to the path to make imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Imports ---
try:
    from organized_poker_bot.game_engine.player import Player
    from organized_poker_bot.cfr.cfr_strategy import CFRStrategy
    from organized_poker_bot.bot.depth_limited_search import DepthLimitedSearch
except ImportError:
    print("ERROR importing BotPlayer dependencies.")
    sys.exit(1)


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
        # Inherited attributes like name, stack, etc., from Player
    """

    def __init__(self, strategy, name="Bot", stack=10000, use_depth_limited_search=False, search_depth=1, search_iterations=100):
        """
        Initialize the bot player.

        Args:
            strategy: The CFR strategy (CFRStrategy object expected, can handle dict).
            name (str): Name for the bot player.
            stack (int): Initial stack for the bot player.
            use_depth_limited_search (bool): Whether to use DLS.
            search_depth (int): DLS search depth.
            search_iterations (int): DLS MCTS iterations.
        """
        # --- CALL PARENT Player CONSTRUCTOR ---
        super().__init__(name=name, stack=stack, is_human=False, is_random=False)

        # --- Initialize BotPlayer specific attributes ---
        self.strategy_obj = None
        if isinstance(strategy, CFRStrategy):
            self.strategy_obj = strategy
        elif isinstance(strategy, dict): # Handle dict for backward compatibility or loading
            print(f"WARN: Initializing BotPlayer '{name}' with strategy dict. Creating CFRStrategy object.")
            self.strategy_obj = CFRStrategy()
            self.strategy_obj.strategy = strategy
        else:
            # print(f"WARN: Unexpected strategy type for BotPlayer '{name}'. Type: {type(strategy)}. Strategy set to None.") # Reduce noise
            pass # Keep strategy_obj as None

        self.use_depth_limited_search = use_depth_limited_search
        self.search_depth = search_depth
        self.search_iterations = search_iterations
        self.dls = None # Initialize dls attribute

        # Initialize depth-limited search if enabled and strategy is valid
        if self.use_depth_limited_search:
            if self.strategy_obj and self.strategy_obj.strategy:
                try:
                     self.dls = DepthLimitedSearch(
                         self.strategy_obj,
                         search_depth=self.search_depth,
                         num_iterations=self.search_iterations
                     )
                     # print(f"DLS initialized for BotPlayer '{self.name}'") # Reduce noise
                except Exception as e:
                    print(f"ERROR initializing DLS for '{self.name}': {e}. Disabling DLS.")
                    self.dls = None
                    self.use_depth_limited_search = False # Disable DLS if init fails
            else:
                # print(f"WARN: Cannot initialize DLS for '{self.name}' due to missing/empty blueprint strategy. DLS Disabled.") # Reduce noise
                self.use_depth_limited_search = False # Ensure DLS is off

    # --- MODIFIED get_action ---
    def get_action(self, game_state, player_idx):
        """
        Get the best action for the current game state. Passes estimated
        initial stacks to DLS if used.

        Args:
            game_state (GameState): The current game state.
            player_idx (int): The player index (redundant with self but needed by external calls).

        Returns:
            tuple: The chosen action (action_type, amount).
        """
        # 1. Use DLS if enabled, valid, and game not over
        if self.use_depth_limited_search and self.dls is not None and not game_state.is_terminal():
            try:
                # --- Estimate initial stacks for DLS's utility calculation ---
                # This is an approximation. Best solution is if the game loop *provides*
                # the actual starting stack for the hand to this function.
                initial_stacks_estimation = [0.0] * game_state.num_players
                for i in range(game_state.num_players):
                    current_stack = float(game_state.player_stacks[i]) if i < len(game_state.player_stacks) and not np.isnan(game_state.player_stacks[i]) else 0.0
                    total_invested = float(game_state.player_total_bets_in_hand[i]) if i < len(game_state.player_total_bets_in_hand) and not np.isnan(game_state.player_total_bets_in_hand[i]) else 0.0
                    initial_stacks_estimation[i] = current_stack + total_invested

                # Call DLS with the current state (it will clone) and estimated initial stacks
                action = self.dls.get_action(game_state, player_idx, initial_stacks_estimation) # Pass original state
                # Ensure DLS returns a valid tuple
                if isinstance(action, tuple) and len(action) == 2: return action
                else: raise ValueError(f"DLS returned invalid action format: {action}")

            except Exception as e:
                print(f"DLS failed for {self.name}: {e}, falling back to blueprint strategy")
                # Fallthrough to blueprint logic

        # 2. Use Blueprint Strategy if DLS not used or failed
        if self.strategy_obj and self.strategy_obj.strategy:
            try:
                 action = self.strategy_obj.get_action(game_state, player_idx) # Pass original state
                 # Ensure blueprint returns a valid tuple
                 if isinstance(action, tuple) and len(action) == 2: return action
                 else: raise ValueError(f"Blueprint strategy returned invalid action format: {action}")
            except Exception as e:
                 print(f"Blueprint strategy failed for {self.name}: {e}")
                 # Fallthrough to default action

        # 3. Default Action (if blueprint missing or failed)
        # print(f"Warning: {self.name} falling back to default action (Check/Fold).") # Reduce noise
        available_actions = game_state.get_available_actions()
        if ('check', 0) in available_actions: return ('check', 0)
        if ('fold', 0) in available_actions: return ('fold', 0)
        # If neither available (e.g., forced all-in call), return first available action or None
        return available_actions[0] if available_actions else ('fold',0) # Force fold if somehow no actions


    def update_strategy(self, new_strategy):
        """
        Update the bot's strategy.

        Args:
            new_strategy: The new strategy to use (CFRStrategy object expected).
        """
        if isinstance(new_strategy, CFRStrategy):
             self.strategy_obj = new_strategy
        elif isinstance(new_strategy, dict): # Allow updating with dict for flexibility
            # print(f"WARN: Updating strategy for {self.name} with a dict. Re-initializing CFRStrategy.")
            self.strategy_obj = CFRStrategy()
            self.strategy_obj.strategy = new_strategy
        else:
             # print(f"Warning: Cannot update strategy for {self.name} with type {type(new_strategy)}")
             return

        # Update depth-limited search if enabled
        if self.use_depth_limited_search:
             if self.strategy_obj and self.strategy_obj.strategy:
                 try:
                      self.dls = DepthLimitedSearch(
                          self.strategy_obj, # Use the updated strategy object
                          search_depth=self.search_depth,
                          num_iterations=self.search_iterations
                      )
                 except Exception as e:
                     print(f"ERROR re-initializing DLS for '{self.name}' after strategy update: {e}")
                     self.dls = None
                     self.use_depth_limited_search = False # Disable if error
             else:
                  # print(f"Warning: Cannot update DLS for {self.name} due to missing/empty strategy object.")
                  self.dls = None


    def __str__(self):
        """Return a string representation of the bot player."""
        strategy_info = "No strategy"
        if self.strategy_obj and self.strategy_obj.strategy:
             strategy_info = f"{len(self.strategy_obj.strategy)} info sets"

        dls_status = f"DLS(d{self.search_depth}, i{self.search_iterations})" if self.use_depth_limited_search and self.dls else "No DLS"
        # Use the inherited Player stack attribute
        stack_str = f"{self.stack:.0f}" if hasattr(self, 'stack') else "N/A"

        return f"BotPlayer(name={self.name}, stack={stack_str}, strategy={strategy_info}, {dls_status})"

# --- END OF FILE organized_poker_bot/bot/bot_player.py ---
