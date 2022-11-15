#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

std::mt19937 mt(42);

// Randomly select an action i with probability strategy[i]
int get_action_from_strategy(std::vector<float> strategy) {
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  float r = dist(mt);
  float cum_prob = 0.0;

  int i = 0;
  while (cum_prob < r && i < strategy.size()) {
    cum_prob += strategy[i];
    if (cum_prob < r) i++; 
  }

  return i;
}

// Implementation of the game CFR is being ran on
//
// This is made up of a set of actions and also a function to
// find the EV of playing one action against another 
class Game {
  public:
    int num_actions;
    std::vector<std::string> action_space;

    Game() {
      // This can be modified to automatically generate if needed.
      action_space = { "rock", "paper", "scissors" };
      num_actions = action_space.size();
    }

    float get_ev(int a, int b) {
      std::string my_action = action_space[a];
      std::string opp_action = action_space[b];

      // This is the main implementation of the game rules (and will 
      // change according to the game).
      //
      // The current implementation is a rock-paper-scissors
      // game where winning with paper gives double the payout 
      // and losing with rock to paper gives double the loss.
      float ev = 0.0;

      if (my_action == "rock") {
        if (opp_action == "paper") ev = -2.0;
        if (opp_action == "scissors") ev = 1.0;
      } else if (my_action == "paper") {
        if (opp_action == "rock") ev = 2.0;
        if (opp_action == "scissors") ev = -1.0; 
      } else if (my_action == "scissors") {
        if (opp_action == "rock") ev = -1.0;
        if (opp_action == "paper") ev = 1.0; 
      }

      return ev;
    }
};

// Main CFR agent
//
// Keeps track of the regrets of not playing different moves (where the
// regret is the lost EV) and this when averaged will converge to a nash-
// equilibrium mixed strategy.
class Player {
  public: 
    std::vector<float> regret_sum;
    std::vector<float> strategy_sum;
    Game game;

    Player(Game g) {
      game = g;
      regret_sum = std::vector<float>(g.num_actions, 0.0);
      strategy_sum = std::vector<float>(g.num_actions, 0.0);
    }

    int get_action() {
      std::vector<float> strategy = regret_sum;

      // Take only the positive values and sum
      float normalizing_sum = 0;
      for (int i = 0; i < strategy.size(); i++) {
        if (strategy[i] < 0) strategy[i] = 0;

        normalizing_sum += strategy[i];
      }

      if (normalizing_sum > 0) {
        for (int i = 0; i < strategy.size(); i++) {
          strategy[i] = strategy[i] / normalizing_sum;
        }
      } else {
        // If we don't have any particular preference for a strategy,
        // we just take each action at equal probability
        for (int i = 0; i < strategy.size(); i++) {
          strategy[i] = 1.0 / strategy.size();
        }
      }

      for (int i = 0; i < strategy.size(); i++) {
        strategy_sum[i] += strategy[i];
      }

      return get_action_from_strategy(strategy);
    }

    void update_regrets(int my_action, int opp_action) {
      float base_ev = game.get_ev(my_action, opp_action);

      for (int i = 0; i < regret_sum.size(); i++) {
        regret_sum[i] += game.get_ev(i, opp_action) - base_ev;
      }
    }

    std::vector<float> get_average_strategy() {
      float normalizing_sum = 0;
      for (int i = 0; i < strategy_sum.size(); i++) {
        normalizing_sum += strategy_sum[i];
      } 

      if (normalizing_sum <= 0) {
        return std::vector<float>(strategy_sum.size(), 1.0 / strategy_sum.size());
      }

      std::vector<float> avg_strategy = strategy_sum;
      for (int i = 0; i < strategy_sum.size(); i++) {
        avg_strategy[i] /= normalizing_sum;
      }  

      return avg_strategy;
    }
};

// Trains the CFR agent
class Trainer {
  Player p1;
  Player p2;

  public:
    Trainer(Player a, Player b) : p1 { a }, p2 { b } {}

    void train(int iterations) {
      for (int i = 0; i < iterations; i++) {
        int p1_action = p1.get_action();
        int p2_action = p2.get_action();

        p1.update_regrets(p1_action, p2_action);
        p2.update_regrets(p2_action, p1_action);
      }
    } 

    std::vector<float> get_average_strategy_p1() {
      return p1.get_average_strategy();
    }

    std::vector<float> get_average_strategy_p2() {
      return p2.get_average_strategy();
    }
};

int main() {
  Game g;

  // Setup two CFR agents and generate a strategy by running CFR
  Player p1(g);
  Player p2(g);

  Trainer trainer(p1, p2);

  // The number of iterations will need to be adjusted based on
  // the desired accuracy and the size of the action space
  trainer.train(50000);

  // This is the final strategy - a list of probabilities at which 
  // you play each of the actions at
  std::vector<float> strategy = trainer.get_average_strategy_p1();

  // Print out a list of the probabilities.
  for (int i = 0; i < strategy.size(); i++) {
    std::cout << g.action_space[i] << ": " << strategy[i];

    if (i != strategy.size() - 1) std::cout << ", ";
  }

  std::cout << std::endl;
}