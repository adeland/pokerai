from poker_engine_corrected import PokerEngine
from mccfr_corrected import MCCFR

def main():
    # Initialize PokerEngine and MCCFR
    engine = PokerEngine(num_players=6)
    mccfr = MCCFR(engine)

    # Training parameters
    iterations = 1000000  # Number of training iterations
    log_frequency = 10000  # Log progress every 100,000 iterations

    # Training loop
    for iteration in range(1, iterations + 1):
        state = engine.initial_state()
        mccfr.cfr(state, iteration)

        if iteration % log_frequency == 0:
            print(f"Iteration {iteration}: Training in progress...")
            # Log strategy for an example suited hand information set
            example_info_set_suited = (("Q", "J"), "s", ("small_blind", "big_blind"), "small_blind")
            strategy_suited = mccfr.get_average_strategy(example_info_set_suited)
            print(f"Average strategy for {example_info_set_suited} after {iteration} iterations: {strategy_suited}")

            # Log strategy for an example off-suit hand information set
            example_info_set_offsuit = (("Q", "J"), "o", ("small_blind", "big_blind"), "small_blind")
            strategy_offsuit = mccfr.get_average_strategy(example_info_set_offsuit)
            print(f"Average strategy for {example_info_set_offsuit} after {iteration} iterations: {strategy_offsuit}")

    # Show strategies for all information sets
    print("\nFinal Strategies for All Information Sets:")
    for info_set, data in mccfr.info_sets.items():
        # Handle different info_set structures
        if len(info_set) == 3:
            hand, suitedness, betting_history = info_set
            position = "Unknown"
        elif len(info_set) == 4:
            hand, suitedness, betting_history, position = info_set
        else:
            print(f"Unexpected info_set structure: {info_set}")
            continue

        print(f"Info set Hand: {hand}, Suitedness: {suitedness}, Betting History: {betting_history}, Position: {position}")
        print(f"Average Strategy: {mccfr.get_average_strategy(info_set)}")

if __name__ == "__main__":
    main()

