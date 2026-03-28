#pragma once
#include "matching_engine.hpp"
#include "metrics.hpp"
#include <cstddef>
#include <optional>
#include <random>
#include <vector>
#include <array>
#include <unordered_map>

enum class Regime : uint8_t { Low = 0, High = 1 };

enum class EventType : uint8_t { LimitBuy, LimitSell, MktBuy, MktSell, Cancel };

struct SimEvent {
  EventType type;
  TimePoint ts;
  Side side;
  Qty qty = 0;
  std::optional<Price> px;           // for limit orders
  std::optional<OrderId> cancel_id;  // for cancel events
};

struct RegimeMix {
  // event mix (probabilities); Cancel is implied as 1 - (sum of these four)
  double p_limit_buy  {0.0};
  double p_limit_sell {0.0};
  double p_mkt_buy    {0.0};
  double p_mkt_sell   {0.0};
  double p_cancel     {0.0};
};

struct RegimeParams {
  double    lambda = 1000.0;  // events per second for this regime
  RegimeMix mix{};
};

struct RegimeConfig {
  double      p_LL  = 0.995;         // stay in Low
  double      p_HH  = 0.990;         // stay in High
  RegimeParams low{};                // low-vol regime
  RegimeParams high{};               // high-vol regime
};

struct SimConfig {
  // RNG / runtime
  uint64_t seed{0};
  size_t   max_events{0};
  uint32_t snapshot_every{0};

  // regime switching
  struct {
    double       p_LL{0.0};
    double       p_HH{0.0};
    RegimeParams low{};
    RegimeParams high{};
  } regime;

  // qty distributions
  double mean_limit_qty{0.0};
  double mean_market_qty{0.0};

  // price model
  int    initial_mid_ticks{0};
  int    min_price_ticks{0};
  int    max_offset_ticks{0};
  double geolap_alpha{0.0};    
  double keep_cross_prob{0.0};

  // logging
  bool   log_trades{false};
};

class Simulator {
public:
  explicit Simulator(const SimConfig& cfg);

  void run();

  // =========================================================================
  // Public accessors for RL environment integration
  // =========================================================================
  OrderBook& book() { return ob_; }
  const OrderBook& book() const { return ob_; }
  MatchingEngine& engine() { return me_; }
  const MatchingEngine& engine() const { return me_; }
  TimePoint now() const { return t_curr_; }
  Regime regime() const { return regime_; }

  // =========================================================================
  // Event generation and execution
  // =========================================================================
  SimEvent next_event();
  
  // Execute event and populate fills vector (caller provides, we append)
  // This is the key change: fills are no longer discarded
  void execute(const SimEvent& e, std::vector<Fill>& fills);

  // =========================================================================
  // Agent order helpers (maintain live_ids_ bookkeeping)
  // Use these instead of raw engine/book access for consistent tracking
  // =========================================================================
  
  // Submit a limit order with proper bookkeeping
  // Returns the order ID; fills are appended to the output vector
  OrderId submit_agent_order(Side side, Price price, Qty quantity, 
                              TimePoint ts, std::vector<Fill>& fills);
  
  // Cancel an order with proper bookkeeping
  void cancel_agent_order(OrderId id);

private:
  // State
  SimConfig     cfg_;
  OrderBook     ob_;
  MatchingEngine me_;
  std::mt19937_64 rng_;
  TimePoint     t_curr_{0.0};
  Regime        regime_{Regime::Low};

  // Telemetry (counts and aggregates used by sim.cpp)
  size_t n_events_   = 0;
  size_t n_limits_   = 0;
  size_t n_markets_  = 0;
  size_t n_cancels_  = 0;
  size_t n_trades_   = 0;

  // ---- Limit-order offset / fill-by-distance telemetry ----
  std::array<uint64_t, 5> lim_total_  {0,0,0,0,0};  // how many limits created per distance bucket
  std::array<uint64_t, 5> lim_filled_ {0,0,0,0,0};  // how many of those ever got at least one fill
  std::unordered_map<OrderId, int> lim_bucket_by_id_; // order id -> bucket

  // Optional (if you haven't added these yet) for average absolute offset & histogram:
  uint64_t limit_offset_count_   = 0;
  uint64_t limit_offset_abs_sum_ = 0;
  std::array<uint64_t, 64> limit_offset_hist_{};     // simple histogram of |offset| in ticks

  // Mid tracking & drawdown (if not already present)
  uint64_t mid_samples_ = 0;
  double   sum_mid_     = 0.0;
  int      peak_mid_    = 0;
  int      max_drawdown_ = 0;

  // Market-order slippage accumulators (if not already present)
  double   mo_buy_slip_  = 0.0;  uint64_t mo_buy_qty_  = 0;
  double   mo_sell_slip_ = 0.0;  uint64_t mo_sell_qty_ = 0;

  // optional (you used them; add if not present)
  uint64_t vol_traded_{0};
  double sum_spread_ = 0.0;

  inline int mid_ticks() const {
    if (ob_.bids.empty() || ob_.asks.empty())
      return cfg_.initial_mid_ticks;
    return (ob_.best_bid() + ob_.best_ask()) / 2;
  }

  // helper: bucket by offset k
  static int bucket_for_offset(int k) {
    if (k <= 0)  return 0;
    if (k <= 2)  return 1;
    if (k <= 5)  return 2;
    if (k <= 10) return 3;
    return 4;
  }

  // live-id tracking for Cancel sampling
  std::vector<OrderId>               live_ids_;
  std::unordered_map<OrderId,size_t> pos_;

  // RNG draws
  double draw_exp(double lambda);
  Qty    draw_geometric_mean(double mean);
  int    draw_two_sided_offset();

  // Regime & events
  void             maybe_switch_regime();
  const RegimeMix& mix_for(Regime r) const;

  // Pricing helpers
  Price  current_mid() const;
  Price  decide_limit_price(Side s);

  // Live id helpers
  void   live_add_if_resting(OrderId id);
  void   live_remove(OrderId id);
  OrderId sample_live();
};