# --- START OF FILE organized_poker_bot/cfr/info_set_util.py ---
"""
Utility function for generating Information Set keys consistently.
"""

import sys
# Assuming GameState, CardAbstraction are needed and importable absolutely
try:
    from organized_poker_bot.game_engine.game_state import GameState # Assuming GameState lives here
    from organized_poker_bot.cfr.card_abstraction import CardAbstraction
    # Import Enhanced if used, handle import error if optional
    try:
        from organized_poker_bot.cfr.enhanced_card_abstraction import EnhancedCardAbstraction
    except ImportError:
        EnhancedCardAbstraction = None
    # ActionAbstraction isn't typically needed for the *key*, just the card state + history
    # from organized_poker_bot.cfr.action_abstraction import ActionAbstraction
except ImportError as e:
     print(f"FATAL Error importing dependencies in info_set_util.py: {e}")
     sys.exit(1)

# Configuration Flags (Match Trainer's defaults or desired setup)
USE_CARD_ABSTRACTION = True
USE_ENHANCED_CARD_ABSTRACTION = True

def generate_info_set_key(game_state, player_idx):
    """
    Generates a consistent information set key based on game state and config.

    Args:
        game_state (GameState): The current game state object.
        player_idx (int): The index of the player whose perspective the key is for.

    Returns:
        str: The generated information set key, or None if error.
    """
    cards_part = "NOCARDS"
    pos_part = "POS_ERR"
    hist_part = "BH_ERR"
    board_features_part = "" # Optional part for enhanced postflop

    try: # Card Part Calculation
        # Safely access hole cards and community cards
        hole = []
        if game_state.hole_cards and 0 <= player_idx < len(game_state.hole_cards):
            hole = game_state.hole_cards[player_idx]

        comm = game_state.community_cards if hasattr(game_state, 'community_cards') else []
        num_comm = len(comm)

        # --- Card Abstraction Logic ---
        if USE_CARD_ABSTRACTION and hole and len(hole) == 2:
            use_enhanced = USE_ENHANCED_CARD_ABSTRACTION and EnhancedCardAbstraction is not None

            if num_comm == 0: # Preflop
                if use_enhanced:
                    # Use enhanced if available AND configured
                    try: cards_part = f"PRE_ENH{EnhancedCardAbstraction.get_preflop_abstraction(hole)}"
                    except Exception: cards_part = f"PRE{CardAbstraction.get_preflop_abstraction(hole)}" # Fallback
                else:
                     # Use basic preflop bucket
                     cards_part = f"PRE{CardAbstraction.get_preflop_abstraction(hole)}"

            else: # Postflop (Flop, Turn, River)
                round_names = {3: "FLOP", 4: "TURN", 5: "RIVER"}
                round_prefix = round_names.get(num_comm, f"POST{num_comm}")

                if use_enhanced:
                     # Use enhanced postflop bucket
                     try:
                          bucket = EnhancedCardAbstraction.get_postflop_abstraction(hole, comm)
                          cards_part = f"{round_prefix}_ENH{bucket}"
                     except Exception: # Fallback to basic postflop if enhanced fails
                           postflop_abs_tuple = CardAbstraction.get_postflop_abstraction(hole, comm)
                           s_buck, b_pair, b_flush = postflop_abs_tuple
                           cards_part = f"{round_prefix}B{s_buck}P{b_pair}F{b_flush}"
                else:
                     # Use basic postflop abstraction (includes board features)
                     postflop_abs_tuple = CardAbstraction.get_postflop_abstraction(hole, comm)
                     s_buck, b_pair, b_flush = postflop_abs_tuple
                     cards_part = f"{round_prefix}B{s_buck}P{b_pair}F{b_flush}"

        elif hole: # Fallback to raw cards if no abstraction or invalid hole cards
             hole_str = "_".join(sorted(str(c) for c in hole))
             comm_str = "_".join(sorted(str(c) for c in comm)) if comm else "noc"
             cards_part = f"RAW_{hole_str}_{comm_str}"

    except Exception as e:
        print(f"WARN: Error during card abstraction in key gen: {e}")
        cards_part = f"CARDS_ERR_{type(e).__name__}"

    # --- Position Part ---
    try:
        position_relative = game_state.get_position(player_idx)
        pos_part = f"POS{position_relative}"
    except Exception as e:
        print(f"WARN: Error getting position in key gen: {e}")
        pos_part = f"POS_ERR_{type(e).__name__}"

    # --- Betting History Part ---
    try:
        hist_part = game_state.get_betting_history()
        # Use "start" for initial preflop state before any actions (other than blinds)
        if not hist_part and game_state.betting_round == GameState.PREFLOP:
             # Distinguish between first action preflop and later empty history postflop
             if not game_state.players_acted_this_round: # Check if anyone acted yet
                 hist_part = "start"
             else: # Should have history if actions occurred
                  hist_part = "empty_hist?" # Indicates possible issue or check situation postflop
        elif not hist_part: # Handle genuinely empty history postflop if needed
             hist_part = "postflop_start?" # Or just keep BH_ERR? Let's default to BH_ERR
             hist_part = "BH_ERR_Empty"
        # Optional truncation
        # MAX_HIST_LEN = 30
        # if len(hist_part) > MAX_HIST_LEN: hist_part = hist_part[-MAX_HIST_LEN:]

    except Exception as e:
         print(f"WARN: Error getting history in key gen: {e}")
         hist_part = f"BH_ERR_{type(e).__name__}"

    # Combine parts ensuring they are strings
    final_key = f"{str(cards_part)}|{str(pos_part)}|{str(hist_part)}"
    return final_key

# --- END OF FILE organized_poker_bot/cfr/info_set_util.py ---
