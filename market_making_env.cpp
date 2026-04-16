#include "market_making_env.h"
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace rl {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MarketMakingEnv::MarketMakingEnv(const EnvConfig& config)
    : config_(config)
{
    // Simulator will be created in reset()
}

MarketMakingEnv::~MarketMakingEnv() = default;

// ============================================================================
// Full Environment Reset
// ============================================================================

Observation MarketMakingEnv::reset() {
    // -------------------------------------------------------------------------
    // 1. Reset all agent financial state
    // -------------------------------------------------------------------------
    cash_ = 0.0;
    inventory_ = 0;
    active_bid_id_ = std::nullopt;
    active_ask_id_ = std::nullopt;
    total_turnover_ = 0.0;
    step_turnover_ = 0.0;
    
    // -------------------------------------------------------------------------
    // 2. Reset timing state (all in simulator time units = seconds)
    // -------------------------------------------------------------------------
    episode_start_time_ = 0.0;
    current_time_ = 0.0;
    last_fill_time_ = std::nullopt;
    done_ = false;
    event_counter_ = 0;
    
    // -------------------------------------------------------------------------
    // 3. Reset market-derived rolling statistics
    // -------------------------------------------------------------------------
    recent_mids_.clear();
    recent_returns_.clear();
    prev_mid_ = std::nullopt;
    last_valid_mid_ = 0.0;
    
    // -------------------------------------------------------------------------
    // 4. Reset the underlying simulator/book
    // -------------------------------------------------------------------------
    reset_simulator();
    warmup_market();
    
    // -------------------------------------------------------------------------
    // 5. Capture simulator time after warmup as episode start
    // -------------------------------------------------------------------------
    episode_start_time_ = simulator_->now();
    current_time_ = episode_start_time_;
    
    // -------------------------------------------------------------------------
    // 6. Compute the first valid mid and store as previous reference
    // -------------------------------------------------------------------------
    double initial_mid = current_mid();
    prev_mid_ = initial_mid;
    last_valid_mid_ = initial_mid;
    recent_mids_.push_back(initial_mid);
    
    // -------------------------------------------------------------------------
    // 7. Compute and store initial mark-to-market baseline
    // -------------------------------------------------------------------------
    prev_mtm_ = mark_to_market();
    
    // -------------------------------------------------------------------------
    // 8. Return the first observation
    // -------------------------------------------------------------------------
    return build_observation();
}

// ============================================================================
// Main Step Function
// ============================================================================

StepResult MarketMakingEnv::step(const Action& action) {
    // -------------------------------------------------------------------------
    // TIMING CONVENTION:
    // - The returned observation is the POST-TRANSITION observation (state s')
    // - The reward corresponds to the transition from previous state s to s'
    // - This matches the standard RL convention: step(a) returns (s', r, done)
    //   where r = R(s, a, s')
    // -------------------------------------------------------------------------
    
    if (done_) {
        throw std::runtime_error("Cannot step: episode is done. Call reset().");
    }
    
    // Reset per-step turnover before market window runs
    step_turnover_ = 0.0;
    
    // =========================================================================
    // Step A: Clamp action and extract fields
    // =========================================================================
    Action clamped = clamp_action(action);
    int bid_offset = clamped.bid_offset;
    int ask_offset = clamped.ask_offset;
    Qty bid_size = static_cast<Qty>(clamped.bid_size);
    Qty ask_size = static_cast<Qty>(clamped.ask_size);
    
    // =========================================================================
    // Step B: Read current mid price from the book
    // =========================================================================
    Price mid = static_cast<Price>(current_mid());
    
    // =========================================================================
    // Step C: Convert offsets into actual quote prices
    // bid_price = mid - bid_offset (prices are integer ticks)
    // ask_price = mid + ask_offset
    // =========================================================================
    Price bid_price = mid - bid_offset;
    Price ask_price = mid + ask_offset;
    
    // =========================================================================
    // Step D: Enforce quote sanity - bid must not cross ask
    // If quotes would cross, widen them symmetrically around mid
    // =========================================================================
    if (bid_price >= ask_price) {
        bid_price = mid - 1;
        ask_price = mid + 1;
    }
    
    // Prices must be positive
    if (bid_price < 1) bid_price = 1;
    if (ask_price < 1) ask_price = 1;
    
    // =========================================================================
    // Step E: Replace agent quotes with computed prices and sizes
    // =========================================================================
    replace_agent_quotes(bid_price, bid_size, ask_price, ask_size);
    
    // Run simulator forward until next decision boundary (time-based)
    run_until_next_decision_time();
    
    // Update volatility with new mid
    double new_mid = current_mid();
    update_volatility(new_mid);
    
    // =========================================================================
    // Episode termination check
    // CONVENTION: Time-based ending is TRUNCATION (not a true terminal state)
    // - done = true means episode ended
    // - truncated = true means it ended due to time limit, not natural termination
    // This distinction matters for value bootstrapping in PPO/TD learning
    // =========================================================================
    double elapsed = current_time_ - episode_start_time_;
    bool time_limit_reached = (elapsed >= config_.episode_duration);
    if (time_limit_reached) {
        done_ = true;
    }
    
    // Compute reward for this transition
    double reward = compute_reward();
    
    // Build result
    StepResult result;
    result.observation = build_observation();  // Post-transition observation (s')
    result.reward = reward;                     // Reward for transition s -> s'
    result.done = done_;
    result.truncated = time_limit_reached;      // True if ended by time, not failure
    result.pnl = mark_to_market();
    result.final_inventory = inventory_;
    result.total_turnover = total_turnover_;
    
    return result;
}

// ============================================================================
// Reward Computation Using Per-Step Quantities
// ============================================================================

double MarketMakingEnv::compute_reward() {
    // 1. Get current portfolio value
    double current_mtm = mark_to_market();
    
    // 2. Delta PnL = current MTM - previous MTM
    double delta_pnl = current_mtm - prev_mtm_;
    
    // 3. Scale delta_pnl to bring it into a reasonable range for RL
    double scaled_pnl = delta_pnl / config_.reward_scale;
    
    // 4. Inventory penalty: convert to double FIRST, then square, then multiply
    double inventory_dbl = static_cast<double>(inventory_);
    double inventory_squared = inventory_dbl * inventory_dbl;
    double inventory_penalty = config_.inventory_penalty_coeff * inventory_squared;
    
    // 5. Turnover penalty using step_turnover (accumulated during process_fills)
    double turnover_penalty = config_.turnover_penalty_coeff * step_turnover_;
    
    // 6. Combine into reward (scaled PnL minus penalties)
    double reward = scaled_pnl - inventory_penalty - turnover_penalty;
    
    // 7. Store current MTM as new baseline for next step
    prev_mtm_ = current_mtm;
    
    // 8. Return the reward
    return reward;
}

// ============================================================================
// Observation Builder
// ============================================================================

Observation MarketMakingEnv::build_observation() const {
    Observation obs;
    
    // -------------------------------------------------------------------------
    // spread_ticks: (best_ask - best_bid), normalized
    // -------------------------------------------------------------------------
    Price spread = best_ask_price() - best_bid_price();
    obs.spread_ticks = std::clamp(static_cast<double>(spread) / config_.max_spread_ticks, 0.0, 1.0);
    
    // -------------------------------------------------------------------------
    // recent_mid_return: most recent step return, scaled
    // Use the last entry in recent_returns_ (populated by update_volatility)
    // -------------------------------------------------------------------------
    if (!recent_returns_.empty()) {
        double raw_return = recent_returns_.back();
        // Scale to roughly [-1, 1]: assume max single-step return ~ 0.5%
        obs.recent_mid_return = std::clamp(raw_return / 0.005, -1.0, 1.0);
    } else {
        obs.recent_mid_return = 0.0;
    }
    
    // -------------------------------------------------------------------------
    // volatility_proxy: rolling std of mid returns, normalized
    // -------------------------------------------------------------------------
    double vol = compute_rolling_volatility();
    obs.volatility_proxy = std::clamp(vol / config_.max_volatility, 0.0, 1.0);
    
    // -------------------------------------------------------------------------
    // top_of_book_imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    // -------------------------------------------------------------------------
    Qty bid_qty = best_bid_quantity();
    Qty ask_qty = best_ask_quantity();
    Qty total_qty = bid_qty + ask_qty;
    if (total_qty > 0) {
        obs.top_of_book_imbalance = static_cast<double>(static_cast<int64_t>(bid_qty) - static_cast<int64_t>(ask_qty)) 
                                    / static_cast<double>(total_qty);
    } else {
        obs.top_of_book_imbalance = 0.0;
    }
    
    // -------------------------------------------------------------------------
    // normalized_inventory = inventory / max_inventory, clipped to [-1, 1]
    // -------------------------------------------------------------------------
    double raw_normalized_inv = static_cast<double>(inventory_) 
                                / static_cast<double>(config_.max_inventory);
    obs.normalized_inventory = std::clamp(raw_normalized_inv, -1.0, 1.0);
    
    // -------------------------------------------------------------------------
    // time_since_last_fill - explicit policy (all in simulator time = seconds):
    // - If a fill has occurred: current_time_ - last_fill_time_
    // - If no fill yet: elapsed episode time (current_time_ - episode_start_time_)
    // - Clipped to max_time_since_fill, then normalized to [0, 1]
    // -------------------------------------------------------------------------
    double time_since_fill;
    if (last_fill_time_.has_value()) {
        // Fill has occurred: compute time since that fill
        time_since_fill = current_time_ - last_fill_time_.value();
    } else {
        // No fill yet: use elapsed episode time as fallback
        time_since_fill = current_time_ - episode_start_time_;
    }
    // Clip to max value
    time_since_fill = std::min(time_since_fill, config_.max_time_since_fill);
    // Normalize to [0, 1]
    obs.time_since_last_fill = time_since_fill / config_.max_time_since_fill;
    
    return obs;
}

// ============================================================================
// Volatility Estimation
// ============================================================================

void MarketMakingEnv::update_volatility(double new_mid) {
    // Compute return if we have a previous mid
    if (prev_mid_.has_value() && prev_mid_.value() > 0.0) {
        double ret = (new_mid - prev_mid_.value()) / prev_mid_.value();
        recent_returns_.push_back(ret);
        
        while (recent_returns_.size() > config_.volatility_window) {
            recent_returns_.pop_front();
        }
    }
    
    // Update mid history
    recent_mids_.push_back(new_mid);
    while (recent_mids_.size() > config_.volatility_window + 1) {
        recent_mids_.pop_front();
    }
    
    // Update previous mid and last valid mid
    prev_mid_ = new_mid;
    if (new_mid > 0.0) {
        last_valid_mid_ = new_mid;
    }
}

double MarketMakingEnv::compute_rolling_volatility() const {
    if (recent_returns_.size() < 2) {
        return 0.0;
    }
    
    double sum = std::accumulate(recent_returns_.begin(), recent_returns_.end(), 0.0);
    double mean = sum / static_cast<double>(recent_returns_.size());
    
    double sq_sum = 0.0;
    for (double r : recent_returns_) {
        double diff = r - mean;
        sq_sum += diff * diff;
    }
    double variance = sq_sum / static_cast<double>(recent_returns_.size());
    
    return std::sqrt(variance);
}

// ============================================================================
// Action Clamping
// ============================================================================

Action MarketMakingEnv::clamp_action(const Action& raw_action) const {
    Action clamped;
    
    // Clamp offsets to valid range [0, MAX_OFFSET]
    clamped.bid_offset = std::clamp(raw_action.bid_offset, 0, Action::MAX_OFFSET);
    clamped.ask_offset = std::clamp(raw_action.ask_offset, 0, Action::MAX_OFFSET);
    
    // Clamp sizes to valid range [0, MAX_SIZE]
    clamped.bid_size = std::clamp(raw_action.bid_size, 0, Action::MAX_SIZE);
    clamped.ask_size = std::clamp(raw_action.ask_size, 0, Action::MAX_SIZE);
    
    // Enforce inventory headroom constraints
    int buy_headroom = config_.max_inventory - static_cast<int>(inventory_);
    int sell_headroom = config_.max_inventory + static_cast<int>(inventory_);
    
    if (clamped.bid_size > buy_headroom) {
        clamped.bid_size = std::max(0, buy_headroom);
    }
    if (clamped.ask_size > sell_headroom) {
        clamped.ask_size = std::max(0, sell_headroom);
    }
    
    return clamped;
}

// ============================================================================
// Quote Management
// ============================================================================

void MarketMakingEnv::replace_agent_quotes(Price bid_price, Qty bid_size,
                                            Price ask_price, Qty ask_size) {
    // Cancel existing orders (this also clears the stored IDs)
    cancel_agent_orders();
    
    // Place new bid if size > 0
    if (bid_size > 0) {
        active_bid_id_ = place_bid(bid_size, bid_price);
    }
    
    // Place new ask if size > 0
    if (ask_size > 0) {
        active_ask_id_ = place_ask(ask_size, ask_price);
    }
}

// ============================================================================
// Fill Processing
// ============================================================================
// Agent identification uses two sources:
// 1. active_bid_id_ / active_ask_id_ for resting orders hit by market events
// 2. submitting_id parameter for immediate fills on order submission
// ============================================================================

void MarketMakingEnv::process_fills(const std::vector<Fill>& fills,
                                     std::optional<OrderId> submitting_id) {
    for (const auto& fill : fills) {
        // ---------------------------------------------------------------------
        // Step 1: Create role variable (none, maker, or taker)
        // ---------------------------------------------------------------------
        enum class Role { None, Maker, Taker };
        Role role = Role::None;
        
        // ---------------------------------------------------------------------
        // Step 2: Check if agent is the taker
        // ---------------------------------------------------------------------
        if (submitting_id.has_value() && fill.taker_id == *submitting_id) {
            role = Role::Taker;
        } else if (active_bid_id_.has_value() && fill.taker_id == *active_bid_id_) {
            role = Role::Taker;
        } else if (active_ask_id_.has_value() && fill.taker_id == *active_ask_id_) {
            role = Role::Taker;
        }
        
        // ---------------------------------------------------------------------
        // Step 3: Only if not taker, check if agent is the maker
        // ---------------------------------------------------------------------
        if (role == Role::None) {
            if (submitting_id.has_value() && fill.maker_id == *submitting_id) {
                role = Role::Maker;
            } else if (active_bid_id_.has_value() && fill.maker_id == *active_bid_id_) {
                role = Role::Maker;
            } else if (active_ask_id_.has_value() && fill.maker_id == *active_ask_id_) {
                role = Role::Maker;
            }
        }
        
        // ---------------------------------------------------------------------
        // Step 4: If no role, agent did not participate - skip this fill
        // ---------------------------------------------------------------------
        if (role == Role::None) {
            continue;
        }
        
        // ---------------------------------------------------------------------
        // Step 5: Determine whether the agent bought or sold
        // ---------------------------------------------------------------------
        bool agent_bought = false;
        
        if (role == Role::Taker) {
            // Agent is taker: taker_side tells us directly
            agent_bought = (fill.taker_side == Side::Buy);
        } else {
            // Agent is maker: opposite of taker_side
            agent_bought = (fill.taker_side == Side::Sell);
        }
        
        // ---------------------------------------------------------------------
        // Step 6: Update inventory, cash, turnover, fill time
        // ---------------------------------------------------------------------
        int64_t qty_signed = static_cast<int64_t>(fill.qty);
        
        if (agent_bought) {
            inventory_ += qty_signed;
        } else {
            inventory_ -= qty_signed;
        }
        
        double notional = static_cast<double>(fill.price) * static_cast<double>(fill.qty);
        
        if (agent_bought) {
            cash_ -= notional;
        } else {
            cash_ += notional;
        }
        
        step_turnover_ += notional;
        total_turnover_ += notional;
        
        last_fill_time_ = fill.ts;
    }
}

// ============================================================================
// Mark-to-Market
// ============================================================================

double MarketMakingEnv::mark_to_market() const {
    double mid = current_mid();
    return cash_ + static_cast<double>(inventory_) * mid;
}

// ============================================================================
// SIMULATOR INTEGRATION
// ============================================================================

double MarketMakingEnv::current_mid() const {
    if (!simulator_) {
        return static_cast<double>(config_.sim_config.initial_mid_ticks);
    }
    
    const OrderBook& ob = simulator_->book();
    
    if (ob.bids.empty() || ob.asks.empty()) {
        if (last_valid_mid_ > 0.0) {
            return last_valid_mid_;
        }
        return static_cast<double>(config_.sim_config.initial_mid_ticks);
    }
    
    double best_bid = static_cast<double>(ob.best_bid());
    double best_ask = static_cast<double>(ob.best_ask());
    double mid = (best_bid + best_ask) / 2.0;
    
    return mid;
}

double MarketMakingEnv::get_mid() const {
    return current_mid();
}

Price MarketMakingEnv::best_bid_price() const {
    if (!simulator_) {
        return config_.sim_config.initial_mid_ticks - 1;
    }
    
    const OrderBook& ob = simulator_->book();
    if (ob.bids.empty()) {
        if (last_valid_mid_ > 0.0) {
            return static_cast<Price>(last_valid_mid_) - 1;
        }
        return config_.sim_config.initial_mid_ticks - 1;
    }
    return ob.best_bid();
}

Price MarketMakingEnv::best_ask_price() const {
    if (!simulator_) {
        return config_.sim_config.initial_mid_ticks + 1;
    }
    
    const OrderBook& ob = simulator_->book();
    if (ob.asks.empty()) {
        if (last_valid_mid_ > 0.0) {
            return static_cast<Price>(last_valid_mid_) + 1;
        }
        return config_.sim_config.initial_mid_ticks + 1;
    }
    return ob.best_ask();
}

Qty MarketMakingEnv::best_bid_quantity() const {
    if (!simulator_) return 0;
    
    const OrderBook& ob = simulator_->book();
    if (ob.bids.empty()) return 0;
    
    const auto& level = ob.bids.begin()->second;
    Qty total = 0;
    for (const auto& order : level) {
        total += order.qty;
    }
    return total;
}

Qty MarketMakingEnv::best_ask_quantity() const {
    if (!simulator_) return 0;
    
    const OrderBook& ob = simulator_->book();
    if (ob.asks.empty()) return 0;
    
    const auto& level = ob.asks.begin()->second;
    Qty total = 0;
    for (const auto& order : level) {
        total += order.qty;
    }
    return total;
}

void MarketMakingEnv::cancel_agent_orders() {
    if (!simulator_) return;
    
    if (active_bid_id_.has_value()) {
        simulator_->cancel_agent_order(active_bid_id_.value());
        active_bid_id_ = std::nullopt;
    }
    
    if (active_ask_id_.has_value()) {
        simulator_->cancel_agent_order(active_ask_id_.value());
        active_ask_id_ = std::nullopt;
    }
}

std::optional<OrderId> MarketMakingEnv::place_bid(Qty quantity, Price price) {
    // Step 1: Guard against invalid inputs
    if (!simulator_ || quantity <= 0 || price <= 0) {
        return std::nullopt;
    }
    
    // Step 2: Create local fills vector
    std::vector<Fill> fills;
    
    // Step 3: Get current simulator time
    TimePoint ts = simulator_->now();
    
    // Step 4: Submit buy limit order with proper bookkeeping
    OrderId id = simulator_->submit_agent_order(Side::Buy, price, quantity, ts, fills);
    
    // Step 5: Process immediate fills
    process_fills(fills, id);
    
    // Step 6: Check whether order still rests in the book
    const OrderBook& ob = simulator_->book();
    if (ob.index.find(id) != ob.index.end()) {
        active_bid_id_ = id;
    } else {
        active_bid_id_ = std::nullopt;
    }
    
    // Step 7: Return the ID
    return id;
}

std::optional<OrderId> MarketMakingEnv::place_ask(Qty quantity, Price price) {
    // -------------------------------------------------------------------------
    // Step 1: Guard against invalid inputs
    if (!simulator_ || quantity <= 0 || price <= 0) {
        return std::nullopt;
    }
    
    // Step 2: Create local fills vector
    std::vector<Fill> fills;
    
    // Step 3: Get current simulator time
    TimePoint ts = simulator_->now();
    
    // Step 4: Submit sell limit order with proper bookkeeping
    OrderId id = simulator_->submit_agent_order(Side::Sell, price, quantity, ts, fills);
    
    // Step 5: Process immediate fills
    process_fills(fills, id);
    
    // Step 6: Check whether order still rests in the book
    const OrderBook& ob = simulator_->book();
    if (ob.index.find(id) != ob.index.end()) {
        active_ask_id_ = id;
    } else {
        active_ask_id_ = std::nullopt;
    }
    
    // Step 7: Return the ID
    return id;
}

// ============================================================================
// TIME-BASED EVENT LOOP
// ============================================================================
// This is the core integration between fixed-interval RL steps and
// event-driven market simulation.
//
// Approach: Simple overshoot
// - Generate and execute events until we cross the target time boundary
// - Accept that we may slightly overshoot the boundary
// - Set current_time_ to the last event's timestamp
//
// This is an approximation, but acceptable for the first integrated version.
// A cleaner design would use peek/lookahead to stop exactly at boundary.
// ============================================================================

void MarketMakingEnv::run_until_next_decision_time() {
    if (!simulator_) {
        current_time_ += config_.decision_interval;
        return;
    }
    
    // Compute target decision boundary
    double target_time = current_time_ + config_.decision_interval;
    
    const OrderBook& ob = simulator_->book();
    
    // Process events until we reach or exceed the target time
    while (true) {
        // Generate next market event (this advances simulator's internal clock)
        SimEvent ev = simulator_->next_event();
        
        // Execute event and collect fills
        std::vector<Fill> fills;
        simulator_->execute(ev, fills);
        
        // Process fills (filters for agent participation internally)
        process_fills(fills);
        
        // Update event counter
        ++event_counter_;
        
        // Clear agent order IDs if they were fully filled (removed from book)
        if (active_bid_id_.has_value()) {
            if (ob.index.find(*active_bid_id_) == ob.index.end()) {
                active_bid_id_ = std::nullopt;
            }
        }
        if (active_ask_id_.has_value()) {
            if (ob.index.find(*active_ask_id_) == ob.index.end()) {
                active_ask_id_ = std::nullopt;
            }
        }
        
        // Update current time to this event's timestamp
        current_time_ = ev.ts;
        
        // Check if we've reached or passed the target time
        if (current_time_ >= target_time) {
            break;
        }
    }
}

void MarketMakingEnv::reset_simulator() {
    simulator_ = std::make_unique<Simulator>(config_.sim_config);
}

void MarketMakingEnv::warmup_market() {
    if (!simulator_) return;
    
    const OrderBook& ob = simulator_->book();
    constexpr size_t MAX_WARMUP_EVENTS = 1000;
    
    for (size_t i = 0; i < MAX_WARMUP_EVENTS; ++i) {
        if (!ob.bids.empty() && !ob.asks.empty()) {
            break;
        }
        
        SimEvent ev = simulator_->next_event();
        std::vector<Fill> fills;
        simulator_->execute(ev, fills);
    }
    
    // Note: episode_start_time_ will be set in reset() after warmup completes
}

} // namespace rl