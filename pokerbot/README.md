# 6-Max No-Limit Hold'em Poker Bot: A CFR/DLS Implementation

This repository contains a high-quality implementation of a No-Limit Hold'em (NLHE) poker bot designed for 6-max games. The approach draws inspiration from groundbreaking work like Pluribus, utilizing Counterfactual Regret Minimization (CFR) for generating a robust blueprint strategy and Depth-Limited Search (DLS) for real-time refinement during play. The focus is on applying rigorous game-theoretic algorithms to solve imperfect information games.

## Key Features

*   **Counterfactual Regret Minimization (CFR):** Employs CFR variants for computing near-equilibrium strategies in the complex 6-max NLHE domain.
*   **Blueprint Strategy:** CFR training generates a comprehensive strategy profile covering numerous game states.
*   **Depth-Limited Search (Optional):** Integrates real-time search (inspired by Pluribus) to refine blueprint strategy decisions based on the exact game state encountered during play.
*   **Card Abstraction:** Utilizes enhanced potential-aware card abstraction techniques (`EnhancedCardAbstraction`) to group strategically similar hands, managing state space complexity.
*   **Action Abstraction:** Implements techniques to abstract the continuous action space of NLHE into a manageable set of discrete betting options.
*   **Optimized Training:** Supports optimized self-play training leveraging parallel processing (`OptimizedSelfPlayTrainer`) for faster convergence on multi-core systems.
*   **Comprehensive Game Engine:** Includes a detailed game engine (`game_engine`) accurately modeling NLHE rules for 2-6 players.
*   **Command-Line Interface:** Provides a flexible CLI (`main.py`) for training, evaluation, playing against the bot, and running validation tests.
*   **Validation & Evaluation:** Includes testing utilities (`simple_test.py`, `test_integration.py`) and evaluation tools (`BotEvaluator`) to verify implementation correctness and measure performance.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd pokerbot # Or your repository's root directory name
    ```
2.  **Install Dependencies:** Ensure you have Python 3.7+ installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is inside a subdirectory, adjust the path accordingly)*

## Usage (Command-Line Interface)

The primary interface is `main.py`. Use `--help` to see all available options:

```bash
python main.py --help
```

1. Training the Bot
Train a blueprint strategy using CFR.

Standard CFR (2-Player Example - Extend Trainer for 6):
```bash
python main.py --mode train --iterations 10000 --output_dir ./models/cfr_blueprint --checkpoint_freq 1000 --num_players 2
```
(Note: The current CFRTrainer in main.py seems geared towards 2 players. Adaptation for direct 6-player standard CFR might be needed)
Optimized Self-Play (6-Player):
```bash
python main.py --mode train --iterations 5000 --output_dir ./models/optimized_6max --checkpoint_freq 500 --num_players 6 --optimized --num_workers 4
```
--iterations: Number of training iterations. More iterations generally yield stronger strategies but require more time.
--output_dir: Directory to save strategy checkpoints and the final strategy (final_strategy.pkl).
--checkpoint_freq: How often to save intermediate models.
--num_players: Set to 6 for 6-max training.
--optimized: Use the parallel OptimizedSelfPlayTrainer.
--num_workers: Number of CPU cores for parallel training.

2. Playing Against the Bot
Play interactively against trained bots.
```bash
python main.py --mode play --strategy ./models/optimized_6max/final_strategy.pkl --num_opponents 5 --opponent human --use_dls --search_depth 2
```

--strategy: Path to the saved strategy file (.pkl).
--num_opponents: Number of bot opponents (total players = this + 1 human).
--opponent: Set to human for interactive play. Can also be random (play vs random bots) or bot (watch bots play each other).
--use_dls: (Optional) Enable Depth-Limited Search for the bot(s).
--search_depth: (Optional) Set the lookahead depth for DLS.
3. Evaluating the Bot
Measure the performance of a trained strategy.
```bash
python main.py --mode evaluate --strategy ./models/optimized_6max/final_strategy.pkl --num_games 1000 --num_opponents 5 --use_dls
```

--strategy: Path to the strategy file to evaluate.
--num_games: Number of hands to simulate for evaluation against random opponents.
--num_opponents: Number of random opponents in evaluation games.
--use_dls: (Optional) Evaluate the bot using DLS.

4. Running Tests
Execute validation tests to ensure core components function correctly.
```bash
python main.py --mode test
```

Project Structure
main.py: Command-line interface entry point.
game_engine/: Core poker game mechanics (Card, Deck, HandEvaluator, GameState, Player, PokerGame).
cfr/: Counterfactual Regret Minimization implementation (CFRTrainer, InformationSet, Abstraction classes).
bot/: Bot agent logic (BotPlayer, DepthLimitedSearch, BotEvaluator).
training/: Training infrastructure (OptimizedSelfPlayTrainer).
utils/: Testing scripts and utility functions (simple_test.py).
models/: (Default) Directory for saving trained strategies.
research/: (Optional) Contains background research files (cfr_research.md, etc.).
Core Concepts
Counterfactual Regret Minimization (CFR): An iterative algorithm designed to find approximate Nash equilibria in large imperfect information games. It works by repeatedly traversing the game tree, calculating the regret (how much better an action would have performed in hindsight), and adjusting strategy probabilities to minimize cumulative regret over time.
Depth-Limited Search (DLS): A real-time search technique used during gameplay. Instead of relying solely on the pre-computed CFR blueprint strategy, DLS performs a limited lookahead search (e.g., using an MCTS-like approach) from the current public game state, potentially biasing the search with the blueprint, to arrive at a more refined action for the specific situation.
Abstraction (Card & Action): Techniques essential for making NLHE computationally tractable.
Card Abstraction: Groups strategically similar hands (e.g., based on strength, potential, board texture) into buckets, reducing the number of distinct card states.
Action Abstraction: Reduces the infinite number of possible bet sizes in NLHE to a smaller, representative set (e.g., fixed fractions of the pot, all-in).
Performance & Tuning (via CLI)
Training Iterations (--iterations): More iterations generally produce stronger, less exploitable strategies but increase training time proportionally.
Optimized Training (--optimized, --num_workers): Utilize multi-core systems for significantly faster training via parallel self-play simulations.
Abstraction Granularity: (Currently adjusted in code) Finer-grained abstractions (more buckets in CardAbstraction, more bet sizes in ActionAbstraction) can improve strategy quality but drastically increase the state space size and training time.
Depth-Limited Search (--use_dls, --search_depth): Enabling DLS (--use_dls) improves real-time decision-making but increases computation time per action. Deeper search (--search_depth) offers potentially stronger play at the cost of performance.


