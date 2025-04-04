# NLH 6-Max Poker Bot

A high-quality No-Limit Hold'em 6-max poker bot implementation inspired by Pluribus, using Counterfactual Regret Minimization (CFR) and depth-limited search.

## Features

- **Counterfactual Regret Minimization (CFR)**: Advanced algorithm for training poker strategies
- **Depth-Limited Search**: Real-time search to refine the blueprint strategy during gameplay
- **Enhanced Card Abstraction**: Potential-aware clustering for better hand representation
- **Action Abstraction**: Realistic bet sizing for different betting rounds
- **Optimized Self-Play Training**: Parallel processing and linear CFR for faster convergence
- **Comprehensive Game Engine**: Complete implementation of No-Limit Hold'em rules

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/poker-bot.git
cd poker-bot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- **Game Engine**: Core poker game mechanics
  - `card.py`: Card representation
  - `deck.py`: Deck of cards with shuffling and dealing
  - `game_state.py`: Game state representation
  - `hand_evaluator.py`: Hand strength evaluation
  - `player.py`: Base player class
  - `poker_game.py`: Main game loop

- **CFR Implementation**: Counterfactual Regret Minimization algorithm
  - `abstraction.py`: Base abstraction class
  - `action_abstraction.py`: Action abstraction techniques
  - `card_abstraction.py`: Card abstraction techniques
  - `enhanced_card_abstraction.py`: Advanced card abstraction with potential-aware clustering
  - `cfr_strategy.py`: Strategy representation
  - `cfr_trainer.py`: CFR training algorithm
  - `information_set.py`: Information set representation

- **Bot Implementation**: Poker bot using CFR and depth-limited search
  - `bot_player.py`: Bot player implementation
  - `depth_limited_search.py`: Real-time search for strategy refinement
  - `bot_evaluator.py`: Bot performance evaluation
  - `bot_optimizer.py`: Bot parameter optimization

- **Training**: Training and self-play
  - `train.py`: Main training script
  - `self_play_trainer.py`: Self-play training
  - `optimized_self_play_trainer.py`: Optimized self-play with parallel processing

## Usage

### Training a Bot

To train a poker bot using CFR:

```python
from cfr_trainer import CFRTrainer
from game_state import GameState

# Create a CFR trainer
trainer = CFRTrainer(
    game_state_class=lambda num_players: GameState(num_players),
    num_players=6
)

# Train for 1000 iterations
strategy = trainer.train(
    iterations=1000,
    checkpoint_freq=100,
    output_dir="models"
)
```

### Using the Bot

To use a trained bot in a game:

```python
from bot_player import BotPlayer
from poker_game import PokerGame
from player import Player

# Load a trained strategy
from cfr_strategy import CFRStrategy
strategy = CFRStrategy()
strategy.load("models/strategy_1000.pkl")

# Create a bot player
bot = BotPlayer(
    strategy=strategy,
    use_depth_limited_search=True,
    search_depth=2,
    search_iterations=1000
)

# Create a game with the bot and human players
players = [bot]
for i in range(5):
    players.append(Player(f"Player {i+1}"))

# Create and run the game
game = PokerGame(players=players, small_blind=50, big_blind=100)
game.run()
```

### Enhanced Features

#### Using Enhanced Card Abstraction

```python
from enhanced_card_abstraction import EnhancedCardAbstraction

# Train clustering models (only needed once)
EnhancedCardAbstraction.train_clustering_models()

# Use in a bot
from bot_player import BotPlayer
from cfr_strategy import CFRStrategy

strategy = CFRStrategy()
strategy.load("models/strategy_1000.pkl")

# Create a bot with enhanced features
bot = BotPlayer(
    strategy=strategy,
    use_depth_limited_search=True,
    search_depth=2,
    search_iterations=1000
)
```

#### Optimized Self-Play Training

```python
from optimized_self_play_trainer import OptimizedSelfPlayTrainer
from game_state import GameState

# Create an optimized self-play trainer
trainer = OptimizedSelfPlayTrainer(
    game_state_class=lambda num_players: GameState(num_players),
    num_players=6,
    num_workers=4  # Use 4 CPU cores
)

# Train using optimized self-play
strategy = trainer.train(
    iterations=1000,
    checkpoint_freq=100,
    output_dir="models"
)
```

## Performance Tuning

For best performance:

1. **Training Time**: Longer training (more iterations) generally produces stronger strategies
2. **Abstraction Granularity**: More buckets in card abstraction improves strategy quality but increases training time
3. **Depth-Limited Search**: Deeper search improves real-time decisions but requires more computation
4. **Parallel Processing**: Use more workers for faster training on multi-core systems

## Comparison to Pluribus

This implementation includes key features from Pluribus:

1. **Counterfactual Regret Minimization**: Core algorithm for strategy training
2. **Depth-Limited Search**: Real-time search for strategy refinement
3. **Abstraction Techniques**: Card and action abstraction to reduce complexity
4. **Self-Play Training**: Learning through playing against itself

While not an exact replica, this implementation captures the essential elements that make Pluribus strong.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pluribus paper: "Superhuman AI for multiplayer poker" by Brown and Sandholm
- The CFR algorithm: "Regret Minimization in Games with Incomplete Information" by Zinkevich et al.
