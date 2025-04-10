# Counterfactual Regret Minimization (CFR) Research

## Overview
Counterfactual Regret Minimization (CFR) is a powerful algorithm for finding Nash equilibrium strategies in imperfect-information games like poker. It was developed by researchers at the University of Alberta and has become the foundation for most modern poker AI systems.

## Key Concepts

### Nash Equilibrium
A Nash equilibrium is a set of strategies for each player where no player can improve their expected outcome by unilaterally changing their strategy. In two-player zero-sum games like poker, finding a Nash equilibrium strategy means finding an unexploitable strategy.

### Regret Minimization
Regret is defined as the difference between the utility a player could have received by playing the best possible action and the utility they actually received. CFR aims to minimize this regret over time through iterative self-play.

### Counterfactual Regret
Counterfactual regret extends the concept of regret to imperfect-information games. It measures the regret of not taking alternative actions at decision points, weighted by the probability of reaching those decision points.

## CFR Algorithm
The CFR algorithm works by:
1. Traversing the game tree
2. Computing counterfactual values for each action
3. Updating regret values for each information set
4. Updating the strategy based on accumulated regrets
5. Repeating this process iteratively

The algorithm converges to a Nash equilibrium as the number of iterations increases.

## Variants and Improvements
Several variants of CFR have been developed to improve performance:
- Monte Carlo CFR (MCCFR): Uses sampling to reduce computational requirements
- CFR+: An optimized version with faster convergence
- Discounted CFR: Applies discounting to historical regrets
- Linear CFR: Uses linear weighting of iterations

## Application to Poker
CFR has been successfully applied to poker, allowing AI systems to achieve superhuman performance:
- Limit Texas Hold'em was essentially solved using CFR variants
- No-limit Texas Hold'em has been tackled by systems like Libratus and Pluribus
- These systems use abstraction techniques to reduce the enormous state space of poker

## Challenges in 6-max NLHE
Applying CFR to 6-max No-Limit Hold'em presents several challenges:
- Much larger state space compared to heads-up play
- More complex multi-way pot dynamics
- Need for effective abstraction techniques to make computation feasible

## References
- "Regret Minimization in Games with Incomplete Information" (Zinkevich et al.)
- "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (Brown et al.)
