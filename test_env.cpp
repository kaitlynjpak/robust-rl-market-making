// test_integration.cpp
// Compile with: g++ -std=c++17 -O2 -o test_integration test_integration.cpp market_making_env.cpp sim.cpp matching_engine.cpp order_book.cpp
//
// This test runs one episode with random actions to verify the integration works.

#include "market_making_env.h"
#include <iostream>
#include <cmath>

rl::Action inventory_skew_policy(const rl::Observation& obs) {
    double inv = obs.normalized_inventory;
    
    int base_offset = 1;
    double skew_strength = 2.0;
    int base_size = 1;
    
    double bid_offset = base_offset + skew_strength * inv;
    double ask_offset = base_offset - skew_strength * inv;
    
    bid_offset = std::max(0.0, bid_offset);
    ask_offset = std::max(0.0, ask_offset);
    
    rl::Action a;
    a.bid_offset = static_cast<int>(std::round(bid_offset));
    a.ask_offset = static_cast<int>(std::round(ask_offset));
    a.bid_size = base_size;
    a.ask_size = base_size;
    
    return a;
}

int main() {
    rl::EnvConfig cfg;
    cfg.episode_duration = 10.0;
    cfg.decision_interval = 0.1;
    cfg.inventory_penalty_coeff = 0.05;  // HIGH PENALTY
    
    cfg.sim_config.initial_mid_ticks = 1000;
    cfg.sim_config.min_price_ticks = 1;
    cfg.sim_config.max_offset_ticks = 10;
    cfg.sim_config.mean_limit_qty = 5.0;
    cfg.sim_config.mean_market_qty = 3.0;
    cfg.sim_config.geolap_alpha = 0.5;
    cfg.sim_config.seed = 1;
    
    cfg.sim_config.regime.p_LL = 0.995;
    cfg.sim_config.regime.p_HH = 0.990;
    cfg.sim_config.regime.low.lambda = 1000.0;
    cfg.sim_config.regime.low.mix = {0.35, 0.35, 0.10, 0.10, 0.10};
    cfg.sim_config.regime.high.lambda = 2000.0;
    cfg.sim_config.regime.high.mix = {0.30, 0.30, 0.15, 0.15, 0.10};
    
    rl::MarketMakingEnv env(cfg);
    
    auto obs = env.reset();
    
    int64_t max_inv = 0;
    int64_t min_inv = 0;
    double total_reward = 0.0;
    int steps = 0;
    
    while (true) {
        rl::Action action = inventory_skew_policy(obs);
        auto result = env.step(action);
        
        if (result.final_inventory > max_inv) max_inv = result.final_inventory;
        if (result.final_inventory < min_inv) min_inv = result.final_inventory;
        total_reward += result.reward;
        steps++;
        
        obs = result.observation;
        
        if (result.done) {
            std::cout << "\n=== SKEW POLICY (0.05 penalty) ===\n"
                      << "steps=" << steps << "\n"
                      << "final_inv=" << result.final_inventory << "\n"
                      << "max_inv=" << max_inv << "\n"
                      << "min_inv=" << min_inv << "\n"
                      << "final_pnl=" << result.pnl << "\n"
                      << "total_reward=" << total_reward << "\n";
            break;
        }
    }
    
    return 0;
}