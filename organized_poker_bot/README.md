# 6-Max No-Limit Hold'em Poker Bot: A CFR/DLS Implementation
♠️♥️♣️♦️

This repository contains a implementation of a No-Limit Hold'em (NLHE) poker bot designed for 6-max games. The approach draws inspiration from work like Pluribus, utilizing Counterfactual Regret Minimization (CFR) for generating a blueprint strategy and Depth-Limited Search (DLS) for real-time refinement during play.

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
```bash
--iterations: Number of training iterations. More iterations generally yield stronger strategies but require more time.
--output_dir: Directory to save strategy checkpoints and the final strategy (final_strategy.pkl).
--checkpoint_freq: How often to save intermediate models.
--num_players: Set to 6 for 6-max training.
--optimized: Use the parallel OptimizedSelfPlayTrainer.
--num_workers: Number of CPU cores for parallel training.
```

2. Playing Against the Bot
Play interactively against trained bots.
```bash
python main.py --mode play --strategy ./models/optimized_6max/final_strategy.pkl --num_opponents 5 --opponent human --use_dls --search_depth 2
```
```bash
--strategy: Path to the saved strategy file (.pkl).
--num_opponents: Number of bot opponents (total players = this + 1 human).
--opponent: Set to human for interactive play. Can also be random (play vs random bots) or bot (watch bots play each other).
--use_dls: (Optional) Enable Depth-Limited Search for the bot(s).
--search_depth: (Optional) Set the lookahead depth for DLS.
```
3. Evaluating the Bot
Measure the performance of a trained strategy.
```bash
python main.py --mode evaluate --strategy ./models/optimized_6max/final_strategy.pkl --num_games 1000 --num_opponents 5 --use_dls
```
```bash
--strategy: Path to the strategy file to evaluate.
--num_games: Number of hands to simulate for evaluation against random opponents.
--num_opponents: Number of random opponents in evaluation games.
--use_dls: (Optional) Evaluate the bot using DLS.
```
4. Running Tests
Execute validation tests to ensure core components function correctly.
```bash
python main.py --mode test
```
```bash
Project Structure
.
├── main.py                 # CLI Entry Point
├── requirements.txt        # Project dependencies
├── setup.py                # Packaging script (optional)
├── README.md               # This file
├── models/                 # Default directory for trained strategies
├── research/               # Background research documents (optional)
│   ├── cfr_research.md
│   └── ...
└── organized_poker_bot/
    ├── __init__.py
    ├── game_engine/        # Core poker rules and state
    │   ├── __init__.py
    │   ├── card.py
    │   ├── deck.py
    │   ├── game_state.py
    │   ├── hand_evaluator.py
    │   ├── player.py
    │   └── poker_game.py
    ├── cfr/                # Counterfactual Regret Minimization
    │   ├── __init__.py
    │   ├── cfr_trainer.py
    │   ├── information_set.py
    │   ├── abstraction.py  # Base/Old abstraction (verify usage)
    │   ├── card_abstraction.py
    │   ├── action_abstraction.py
    │   └── enhanced_card_abstraction.py
    ├── bot/                # Bot agent implementation
    │   ├── __init__.py
    │   ├── bot_player.py
    │   ├── depth_limited_search.py
    │   ├── bot_evaluator.py
    │   └── bot_optimizer.py # (If used)
    ├── training/           # Training infrastructure
    │   ├── __init__.py
    │   ├── optimized_self_play_trainer.py
    │   └── ...
    └── utils/              # Utilities and tests
        ├── __init__.py
        └── simple_test.py
        └── test_integration.py # (If used)
```


