# Toy CFR Solver

This is a toy implementation of counterfactual regret minimization (CFR), used to solve a very simple modified game of rock paper scissors (where winning with rock wins double points, and losing with paper loses double points) and compute the nash-equilibrium strategy.

There is also a sample implementation of solving a slightly more complex game of Blotto, with 2 castles worth (1, 2) and 5 troops.


## Usage

You can compile the solver with any modern C++ solver, with say

```
$ g++ solver.cpp -o solver -O3 -std=c++11
$ ./solver
```

And it should output the computed solution:

```
rock: 0.248074, paper: 0.252397, scissors: 0.499529
```
