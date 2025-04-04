import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Training script for running self-play training of the poker bot.
"""

import os
import argparse
import time
from ..self_play import SelfPlayTrainer

def main():
    """
    Main function to run self-play training.
    """
    parser = argparse.ArgumentParser(description='Train poker bot using self-play')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='How often to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='../../models',
                        help='Directory to save training outputs')
    parser.add_argument('--small_blind', type=int, default=50,
                        help='Small blind amount')
    parser.add_argument('--big_blind', type=int, default=100,
                        help='Big blind amount')
    parser.add_argument('--starting_stack', type=int, default=10000,
                        help='Starting chip stack for each player')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Create absolute path for output directory
    output_dir = os.path.abspath(args.output_dir)
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        output_dir=output_dir,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Run training
    print(f"Starting training for {args.iterations} iterations...")
    start_time = time.time()
    
    stats = trainer.train(
        iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Total iterations: {stats['iterations']}")
    if stats['exploitability']:
        print(f"Final exploitability: {stats['exploitability'][-1]:.6f}")
    
    print(f"Trained strategy saved to: {output_dir}")

if __name__ == "__main__":
    main()
