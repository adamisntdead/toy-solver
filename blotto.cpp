#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <functional>

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
    std::vector<std::vector<int>> action_space;
    std::vector<int> tower_values;

    Game() {
      tower_values = {1, 2};
      action_space = generate_action_space();
      num_actions = action_space.size();
    }

  std::vector<std::vector<int>> generate_action_space() {
    std::vector<std::vector<int>> actions;
    int num_towers = tower_values.size();
    int num_troops = 5;

    // n is num_towers, m is num troops
    std::function<void(int, int, std::vector<int>&)> generate = [&](int index, int remaining, std::vector<int>& current) {
      if (index == num_towers - 1) {
        current[index] = remaining;
        actions.push_back(current);
        return;
      }
      
      for (int i = 0; i <= remaining; ++i) {
        current[index] = i;
        generate(index + 1, remaining - i, current);
      }
    };
    
    std::vector<int> current(num_towers, 0);
    generate(0, num_troops, current);

    return actions;
  }

    float get_ev(int a, int b) {
      const std::vector<int>& my_distribution = action_space[a];
      const std::vector<int>& opp_distribution = action_space[b];
      float score = 0;

      for (int i = 0; i < 5; ++i) {
        if (my_distribution[i] > opp_distribution[i]) {
          score += tower_values[i];
        } else if (my_distribution[i] < opp_distribution[i]) {
          score -= tower_values[i];
        }
      }

      return score;
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
      for (unsigned i = 0; i < strategy.size(); i++) {
        if (strategy[i] < 0) strategy[i] = 0;

        normalizing_sum += strategy[i];
      }

      if (normalizing_sum > 0) {
        for (unsigned i = 0; i < strategy.size(); i++) {
          strategy[i] = strategy[i] / normalizing_sum;
        }
      } else {
        // If we don't have any particular preference for a strategy,
        // we just take each action at equal probability
        for (unsigned i = 0; i < strategy.size(); i++) {
          strategy[i] = 1.0 / strategy.size();
        }
      }

      for (unsigned i = 0; i < strategy.size(); i++) {
        strategy_sum[i] += strategy[i];
      }

      return get_action_from_strategy(strategy);
    }

    void update_regrets(int my_action, int opp_action) {
      float base_ev = game.get_ev(my_action, opp_action);

      for (unsigned i = 0; i < regret_sum.size(); i++) {
        regret_sum[i] += game.get_ev(i, opp_action) - base_ev;
      }
    }

    std::vector<float> get_average_strategy() {
      float normalizing_sum = 0;
      for (unsigned i = 0; i < strategy_sum.size(); i++) {
        normalizing_sum += strategy_sum[i];
      } 

      if (normalizing_sum <= 0) {
        return std::vector<float>(strategy_sum.size(), 1.0 / strategy_sum.size());
      }

      std::vector<float> avg_strategy = strategy_sum;
      for (unsigned i = 0; i < strategy_sum.size(); i++) {
        avg_strategy[i] /= normalizing_sum;
      }  

      return avg_strategy;
    }

    float compare_strategy(std::vector<float> opp_strategy) {
      std::vector<float> my_strategy = get_average_strategy();

      float ev = 0;

      for (int my_action = 0; my_action < game.num_actions; my_action++) {
        for (int opp_action = 0; opp_action < game.num_actions; opp_action++) {
          float my_ev = game.get_ev(my_action, opp_action);
          float action_probability = my_strategy[my_action] * opp_strategy[opp_action];

          ev += my_ev * action_probability;
        }
      }

      return ev;
    }

    float opponent_best_action_ev(std::vector<float> opp_strategy) {
      float best_action_ev = 0.0;

      for (int my_action = 0; my_action < game.num_actions; my_action++) {
        float current_action_ev = 0.0;

        for (int opp_action = 0; opp_action < game.num_actions; opp_action++) {
          float my_ev = game.get_ev(my_action, opp_action);
          float play_probability = opp_strategy[opp_action];

          current_action_ev += my_ev * play_probability;
        }
      
        if (best_action_ev <= current_action_ev) {
            best_action_ev = current_action_ev;
          }
      }

      return best_action_ev;
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

    float get_ev_p1() {
      std::vector<float> p2_strategy = get_average_strategy_p2();
  
      return p1.compare_strategy(p2_strategy);
    }

    float get_exploitability_p1() {
      float p1_ev = get_ev_p1();
      float ev_loss_to_best_response = p2.opponent_best_action_ev(get_average_strategy_p1());

      std::cout << "EV Loss to Best Response: " << ev_loss_to_best_response << std::endl;
      return ev_loss_to_best_response - p1_ev;
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

    // Create a vector of pairs to store moves and their probabilities
    std::vector<std::pair<std::string, float>> move_probabilities;

    // Populate the vector with moves and their probabilities
    for (unsigned i = 0; i < strategy.size(); i++) {
        std::ostringstream move_str;
        move_str << "Move (";
        for (int j = 0; j < g.action_space[i].size(); j++) {
            move_str << g.action_space[i][j];
            if (j < g.action_space[i].size() - 1) move_str << ",";
        }
        move_str << ")";
        move_probabilities.push_back({move_str.str(), strategy[i]});
    }

    // Sort the vector based on probabilities in descending order
    std::sort(move_probabilities.begin(), move_probabilities.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Print out the sorted moves and their probabilities
    std::cout << "Move & Strategy (Frequency):" << std::endl;
    for (const auto& move : move_probabilities) {
        std::cout << move.first << ": " << move.second << std::endl;
    }

    float p1_ev = trainer.get_ev_p1();
    float p1_exploitability = trainer.get_exploitability_p1();

    std::cout << "\nPlayer 1 EV: " << p1_ev << std::endl;
    std::cout << "Player 1 Exploitability: " << p1_exploitability << std::endl;

    return 0;
}