#include "market_making_env.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

rl::Action inventory_skew_policy(const rl::Observation& obs) {
    double inv = obs.normalized_inventory;
    double bid_off = 1.0 + 2.0 * inv;
    double ask_off = 1.0 - 2.0 * inv;
    
    rl::Action a;
    a.bid_offset = std::max(1, static_cast<int>(std::round(bid_off)));
    a.ask_offset = std::max(1, static_cast<int>(std::round(ask_off)));
    a.bid_size = 1;
    a.ask_size = 1;
    return a;
}

struct EpisodeResult {
    double pnl;
    double reward;
    int64_t min_inv;
    int64_t max_inv;
};

EpisodeResult run_episode(rl::EnvConfig& cfg, bool use_skew) {
    rl::MarketMakingEnv env(cfg);
    
    rl::Action fixed;
    fixed.bid_offset = 1;
    fixed.ask_offset = 1;
    fixed.bid_size = 1;
    fixed.ask_size = 1;
    
    auto obs = env.reset();
    double total_reward = 0;
    int64_t min_inv = 0, max_inv = 0;
    
    while (true) {
        auto result = env.step(use_skew ? inventory_skew_policy(obs) : fixed);
        total_reward += result.reward;
        
        int64_t inv = env.current_inventory();
        min_inv = std::min(min_inv, inv);
        max_inv = std::max(max_inv, inv);
        
        obs = result.observation;
        if (result.done) {
            return {result.pnl, total_reward, min_inv, max_inv};
        }
    }
}

void run_baseline(const char* label, rl::EnvConfig& cfg, int n_seeds) {
    double fixed_pnl_sum = 0, skew_pnl_sum = 0;
    double fixed_reward_sum = 0, skew_reward_sum = 0;
    int64_t fixed_inv_range_sum = 0, skew_inv_range_sum = 0;
    
    for (int seed = 1; seed <= n_seeds; ++seed) {
        cfg.sim_config.seed = seed;
        auto fixed_res = run_episode(cfg, false);
        
        cfg.sim_config.seed = seed;
        auto skew_res = run_episode(cfg, true);
        
        fixed_pnl_sum += fixed_res.pnl;
        fixed_reward_sum += fixed_res.reward;
        fixed_inv_range_sum += (fixed_res.max_inv - fixed_res.min_inv);
        
        skew_pnl_sum += skew_res.pnl;
        skew_reward_sum += skew_res.reward;
        skew_inv_range_sum += (skew_res.max_inv - skew_res.min_inv);
    }
    
    std::cout << label << " (n=" << n_seeds << ")\n";
    std::cout << "  FIXED: avg_pnl=" << fixed_pnl_sum/n_seeds 
              << ", avg_reward=" << fixed_reward_sum/n_seeds
              << ", avg_inv_range=" << fixed_inv_range_sum/n_seeds << "\n";
    std::cout << "  SKEW:  avg_pnl=" << skew_pnl_sum/n_seeds 
              << ", avg_reward=" << skew_reward_sum/n_seeds
              << ", avg_inv_range=" << skew_inv_range_sum/n_seeds << "\n";
    std::cout << "\n";
}

int main() {
    rl::EnvConfig cfg;
    cfg.episode_duration = 10.0;
    cfg.decision_interval = 0.1;
    cfg.inventory_penalty_coeff = 0.01;
    
    cfg.sim_config.initial_mid_ticks = 1000;
    cfg.sim_config.min_price_ticks = 1;
    cfg.sim_config.max_offset_ticks = 10;
    cfg.sim_config.geolap_alpha = 0.5;
    
    cfg.sim_config.regime.high.lambda = 2000.0;
    cfg.sim_config.regime.high.mix = {0.15, 0.15, 0.30, 0.30, 0.10};
    cfg.sim_config.regime.high.mean_limit_qty = 5.0;
    cfg.sim_config.regime.high.mean_market_qty = 8.0;
    
    cfg.sim_config.regime.p_LL = 0.0;
    cfg.sim_config.regime.p_HH = 1.0;
    
    const int N_SEEDS = 20;
    
    std::cout << "=== BASELINE EXPERIMENTS ===\n\n";
    
    // NO TOXICITY
    cfg.sim_config.regime.high.toxicity_prob = 0.0;
    cfg.sim_config.regime.high.toxicity_duration = 0;
    cfg.sim_config.regime.high.toxicity_strength = 0.0;
    cfg.sim_config.regime.high.toxicity_drift = 0.0;
    cfg.sim_config.regime.high.impact_decay = 0.0;
    run_baseline("NO TOXICITY", cfg, N_SEEDS);
    
    // LIGHT TOXICITY
    cfg.sim_config.regime.high.toxicity_prob = 1.0;
    cfg.sim_config.regime.high.toxicity_duration = 10;
    cfg.sim_config.regime.high.toxicity_strength = 0.15;
    cfg.sim_config.regime.high.toxicity_drift = 0.05;
    cfg.sim_config.regime.high.impact_decay = 0.05;
    run_baseline("LIGHT TOXICITY (drift=0.05)", cfg, N_SEEDS);
    
    // STRONG TOXICITY
    cfg.sim_config.regime.high.toxicity_drift = 0.2;
    run_baseline("STRONG TOXICITY (drift=0.2)", cfg, N_SEEDS);
    
    return 0;
}