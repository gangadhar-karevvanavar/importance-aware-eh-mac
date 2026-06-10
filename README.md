# Importance-Aware Progressive Delivery of Version Updates over Multiple Access Channels

This repository contains simulation codes for the paper:

**"Importance-Aware Progressive Delivery of Version Updates over Multiple Access Channels"**

---

## Paper Summary

This work studies importance-aware progressive delivery of version updates over an energy harvesting fading multiple access channel (MAC) using non-orthogonal multiple access (NOMA).

Unlike conventional packet-based freshness models where updates are either fully delivered or lost, this framework enables progressive transmission of version updates across multiple time slots. Partial transmission allows the receiver to obtain intermediate information quality, while additional transmitted bits progressively refine the received version.

Each version is associated with:
- An externally assigned importance weight
- A convex urgency-aware cost based on remaining undelivered bits

The proposed cost function combines:
- Importance weighting
- Convex increasing penalty on remaining bits

This formulation incentivizes:
- Early transmission of more bits
- Timely refinement of partially delivered updates
- Efficient use of harvested energy resources

The work jointly optimizes:
- User scheduling
- Bit allocation
- Power control

subject to:
- Multiple access channel achievable rate region constraints
- Energy harvesting constraints
- Battery dynamics at transmitters

The problem is formulated as a Markov Decision Process (MDP) with:
- Mixed continuous and discrete state spaces
- Mixed continuous and discrete action spaces

Key contributions include:
- Progressive delivery framework for version updates
- Importance-aware urgency-sensitive cost modeling
- Structural analysis proving convexity of the value function
- Monotonicity properties of optimal power and bit allocation policies
- Approximate Dynamic Programming (ADP) based solution
- Convex quadratic value function approximation
- Neural network based online control policy
- Benchmark discretized MDP and greedy policies

Simulation results demonstrate:
- Superior performance of the ADP-based policy
- Efficient energy utilization under harvesting constraints
- Significant reduction in long-term average urgency-weighted cost
- Improved progressive refinement performance compared to greedy benchmarks

The framework generalizes traditional freshness-aware communication by enabling partial and progressively improving information delivery with urgency-sensitive scheduling.

---

## Paper Links

### IEEE
[IEEE Xplore](https://ieeexplore.ieee.org/document/10283619)

