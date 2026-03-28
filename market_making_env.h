#ifndef MARKET_MAKING_ENV_H
#define MARKET_MAKING_ENV_H

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

// Include your simulator headers
#include "order_book.hpp"
#include "matching_engine.hpp"
#include "sim.hpp"

namespace rl {

// ============================================================================
// Configuration
// ============================================================================
// TIME CONVENTION: All time values are in SIMULATOR UNITS (seconds).
// With lambda=1000 events/sec, one event arrives every ~0.001 seconds on average.
// ============================================================================

struct EnvConfig {
    // Timing (all in simulator time units = seconds)
    double episode_duration = 60.0;       // 60 seconds
    double decision_interval = 0.1;       // 100ms = 0.1 seconds between decisions
    
    // Position limits
    int max_inventory = 10;
    
    // Reward shaping
    double inventory_penalty_coeff = 0.001;
    double turnover_penalty_coeff = 0.0001;
    
    // Volatility estimation
    size_t volatility_window = 20;              // Number of mid samples for rolling vol
    
    // Feature scaling
    double max_spread_ticks = 20.0;
    double max_volatility = 0.01;               // 1% for normalization
    double max_time_since_fill = 10.0;          // Cap at 10 seconds
    
    // Tick size (in your sim, prices are integer ticks, so tick_size = 1)
    double tick_size = 1.0;
    
    // Simulator config to use for each episode
    SimConfig sim_config;
};

// ============================================================================
// Observation (6 features, all normalized to roughly [-1, 1] or [0, 1])
// ============================================================================

struct Observation {
    static constexpr size_t DIM = 6;
    
    double spread_ticks;            // (ask - bid), normalized
    double recent_mid_return;       // (mid - prev_mid) / prev_mid, scaled
    double volatility_proxy;        // Rolling std of mid returns, normalized
    double top_of_book_imbalance;   // (bid_qty - ask_qty) / (bid_qty + ask_qty)
    double normalized_inventory;    // inventory / max_inventory, in [-1, 1]
    double time_since_last_fill;    // Normalized time since last fill
    
    std::array<double, DIM> to_array() const {
        return {spread_ticks, recent_mid_return, volatility_proxy,
                top_of_book_imbalance, normalized_inventory, time_since_last_fill};
    }
};

// ============================================================================
// Action (offset-based quoting)
// Each field specifies how to quote relative to mid price
// ============================================================================

struct Action {
    int bid_offset;  // Ticks below mid for bid quote (0 = at mid, positive = deeper)
    int ask_offset;  // Ticks above mid for ask quote (0 = at mid, positive = deeper)
    int bid_size;    // Number of lots to bid
    int ask_size;    // Number of lots to ask
    
    // For discrete action spaces, you might map a single int to these 4 fields.
    // Example: 3 offsets × 3 offsets × 3 sizes × 3 sizes = 81 actions
    // Or use a smaller subset. This is left to your RL design.
    
    static constexpr int MAX_OFFSET = 5;   // Max ticks from mid
    static constexpr int MAX_SIZE = 3;     // Max lots per side
};

// ============================================================================
// Step result
// ============================================================================

struct StepResult {
    Observation observation;
    double reward;
    bool done;
    bool truncated;
    
    // Info dict equivalent
    double pnl;
    int64_t final_inventory;
    double total_turnover;
};

// ============================================================================
// Main Environment Class
// ============================================================================

class MarketMakingEnv {
public:
    explicit MarketMakingEnv(const EnvConfig& config = EnvConfig{});
    ~MarketMakingEnv();
    
    // Disable copy (simulator holds resources)
    MarketMakingEnv(const MarketMakingEnv&) = delete;
    MarketMakingEnv& operator=(const MarketMakingEnv&) = delete;
    
    // Core RL interface
    Observation reset();
    StepResult step(const Action& action);
    
    // Accessors for debugging/logging
    int64_t current_inventory() const { return inventory_; }
    double current_cash() const { return cash_; }
    double current_time() const { return current_time_; }
    bool is_done() const { return done_; }
    
private:
    // ========================================================================
    // Simulator Integration (now using real types)
    // ========================================================================
    
    // Get current mid price from the book (floating-point)
    double current_mid() const;
    
    // Get best bid/ask prices and quantities
    Price best_bid_price() const;
    Price best_ask_price() const;
    Qty best_bid_quantity() const;
    Qty best_ask_quantity() const;
    
    // Cancel all agent orders
    void cancel_agent_orders();
    
    // Place new agent quotes - returns order ID, processes any immediate fills
    std::optional<OrderId> place_bid(Qty quantity, Price price);
    std::optional<OrderId> place_ask(Qty quantity, Price price);
    
    // Run simulator until next decision time, processing fills internally
    void run_until_next_decision_time();
    
    // Reset or reinitialize the underlying simulator
    void reset_simulator();
    
    // Warm up the market so best bid/ask exist
    void warmup_market();
    
    // ========================================================================
    // Internal State Management
    // ========================================================================
    
    // Clamp action fields to valid ranges
    Action clamp_action(const Action& raw_action) const;
    
    // Replace existing quotes with new ones at specified prices and sizes
    void replace_agent_quotes(Price bid_price, Qty bid_size, 
                              Price ask_price, Qty ask_size);
    
    // Process fills and update cash/inventory (filters for agent participation)
    // - For market-event fills: uses active_bid_id_ / active_ask_id_ to identify agent
    // - For immediate fills on submission: pass the submitting order ID explicitly
    void process_fills(const std::vector<Fill>& fills, 
                       std::optional<OrderId> submitting_id = std::nullopt);
    
    // Compute current portfolio value (cash + inventory * mid)
    double mark_to_market() const;
    
    // Compute reward for this step
    double compute_reward();
    
    // Build observation from current state
    Observation build_observation() const;
    
    // Update volatility estimator with new mid
    void update_volatility(double new_mid);
    
    // Compute rolling volatility from recent returns
    double compute_rolling_volatility() const;
    
    // ========================================================================
    // Configuration
    // ========================================================================
    EnvConfig config_;
    
    // ========================================================================
    // Agent Financial State
    // ========================================================================
    double cash_ = 0.0;
    int64_t inventory_ = 0;  // Signed 64-bit to safely handle uint64_t quantities
    std::optional<OrderId> active_bid_id_;
    std::optional<OrderId> active_ask_id_;
    
    // Turnover tracking
    double total_turnover_ = 0.0;    // Cumulative for episode info
    double step_turnover_ = 0.0;     // Per-step for reward calculation
    
    // ========================================================================
    // Timing State
    // ========================================================================
    // TIME CONVENTION: All times in simulator units (seconds).
    // With lambda=1000, events arrive ~every 0.001s on average.
    // decision_interval=0.1 means ~100 events per RL step on average.
    double episode_start_time_ = 0.0;
    double current_time_ = 0.0;
    std::optional<double> last_fill_time_;  // Simulator time when last fill occurred
    bool done_ = false;
    
    // ========================================================================
    // Market-Derived Rolling Statistics
    // ========================================================================
    std::deque<double> recent_mids_;          // Mid prices for return computation
    std::deque<double> recent_returns_;       // For volatility estimation
    std::optional<double> prev_mid_;          // Previous mid for return calculation
    double last_valid_mid_ = 0.0;             // Fallback when book is empty (floating-point)
    
    // ========================================================================
    // Reward Computation State
    // ========================================================================
    double prev_mtm_ = 0.0;               // Previous mark-to-market for delta PnL
    
    // ========================================================================
    // Simulator (owned)
    // ========================================================================
    std::unique_ptr<Simulator> simulator_;
    
    // Counter for generating unique sub-timestamps within agent order placement
    int64_t event_counter_ = 0;
};

} // namespace rl

#endif // MARKET_MAKING_ENV_H