#include "sim.hpp"
#include <cmath>
#include <iostream>
#include <limits>

static double mean_to_geom_p(double mean) {
  if (mean <= 1.0) return 1.0;
  return 1.0 / mean; // mean = 1/p for shifted geometric on {1,2,...}
}

static inline int bucket_of(int k) {
  if (k == 0) return 0;   // exactly at mid
  if (k <= 2) return 1;   // 1–2 ticks
  if (k <= 5) return 2;   // 3–5
  if (k <= 10) return 3;  // 6–10
  return 4;               // >10
}

Simulator::Simulator(const SimConfig& cfg)
  : cfg_(cfg),
    ob_(),               // we own the order book
    me_(ob_),            // MatchingEngine requires OrderBook&
    rng_(cfg.seed) {}

double Simulator::draw_exp(double lambda) {
  if (lambda <= 0.0) return 0.0;
  std::uniform_real_distribution<double> U(0.0, 1.0);
  double u = std::max(U(rng_), std::numeric_limits<double>::min());
  return -std::log(u) / lambda;
}

Qty Simulator::draw_geometric_mean(double mean) {
  double p = mean_to_geom_p(mean);
  std::geometric_distribution<int> G(p); // {0,1,2,...}
  return static_cast<Qty>(G(rng_) + 1);  // shift -> {1,2,...}
}

int Simulator::draw_two_sided_offset() {
  double a = cfg_.geolap_alpha;
  if (a <= 0.0) a = 1.0;
  if (a > 1.0)  a = 1.0;
  std::geometric_distribution<int> G(a);
  int k = G(rng_) + 1;
  if (cfg_.max_offset_ticks > 0) k = std::min(k, cfg_.max_offset_ticks);
  std::bernoulli_distribution B(0.5);
  return B(rng_) ? k : -k;
}

void Simulator::maybe_switch_regime() {
  std::bernoulli_distribution stay_low(cfg_.regime.p_LL);
  std::bernoulli_distribution stay_high(cfg_.regime.p_HH);
  regime_ = (regime_ == Regime::Low)
              ? (stay_low(rng_) ? Regime::Low : Regime::High)
              : (stay_high(rng_) ? Regime::High : Regime::Low);
}

const RegimeMix& Simulator::mix_for(Regime r) const {
  return (r == Regime::Low) ? cfg_.regime.low.mix : cfg_.regime.high.mix;
}

Price Simulator::current_mid() const {
  Price m = ob_.mid();
  Price base = (m > 0) ? m : cfg_.initial_mid_ticks;
  // Add price drift from market order impact
  Price adjusted = base + static_cast<Price>(std::round(price_drift_));
  return std::max(adjusted, static_cast<Price>(cfg_.min_price_ticks));
}

Price Simulator::decide_limit_price(Side s) {
  Price mid = current_mid();
  int off = draw_two_sided_offset();
  Price px = mid + off;

  // If would cross, pull it back half the time to keep some resting depth
  if (s == Side::Buy) {
    if (!ob_.asks.empty() && px >= ob_.best_ask()) {
      std::bernoulli_distribution keep_cross(0.5);
      if (!keep_cross(rng_)) px = std::min<Price>(ob_.best_bid(), mid - std::abs(off));
    }
  } else {
    if (!ob_.bids.empty() && px <= ob_.best_bid()) {
      std::bernoulli_distribution keep_cross(0.5);
      if (!keep_cross(rng_)) px = std::max<Price>(ob_.best_ask(), mid + std::abs(off));
    }
  }

  if (px < cfg_.min_price_ticks) px = cfg_.min_price_ticks;
  return px;
}

void Simulator::live_add_if_resting(OrderId id) {
  // Only add if that id is actually resting in the book
  if (ob_.index.find(id) == ob_.index.end()) return;
  if (pos_.count(id)) return;
  size_t idx = live_ids_.size();
  live_ids_.push_back(id);
  pos_[id] = idx;
}

void Simulator::live_remove(OrderId id) {
  auto it = pos_.find(id);
  if (it == pos_.end()) return;
  size_t idx = it->second;
  OrderId last = live_ids_.back();
  live_ids_[idx] = last;
  pos_[last] = idx;
  live_ids_.pop_back();
  pos_.erase(it);
}

OrderId Simulator::sample_live() {
  if (live_ids_.empty()) return 0;
  std::uniform_int_distribution<size_t> U(0, live_ids_.size() - 1);
  return live_ids_[U(rng_)];
}

SimEvent Simulator::next_event() {
  // Advance time by exponential inter-arrival at current regime rate
  const RegimeParams& rp = (regime_ == Regime::Low)
                         ? cfg_.regime.low
                         : cfg_.regime.high;

  // CRITICAL: Actually advance time by sampling inter-arrival delay
  t_curr_ += draw_exp(rp.lambda);

  // Maybe switch regime at the arrival boundary
  maybe_switch_regime();

  // Choose event type with toxicity-adjusted probabilities
  const RegimeMix& mix = mix_for(regime_);
  
  // Start with base probabilities
  double p_mkt_buy = mix.p_mkt_buy;
  double p_mkt_sell = mix.p_mkt_sell;
  
  // Apply toxicity bias if active
  if (toxic_events_remaining_ > 0 && toxic_direction_ != ToxicDirection::None) {
    double shift = rp.toxicity_strength;
    
    if (toxic_direction_ == ToxicDirection::Up) {
      // Bias toward buys (price pressure up)
      double actual_shift = std::min(shift, p_mkt_sell);  // Can't shift more than available
      p_mkt_buy += actual_shift;
      p_mkt_sell -= actual_shift;
    } else {
      // Bias toward sells (price pressure down)
      double actual_shift = std::min(shift, p_mkt_buy);
      p_mkt_sell += actual_shift;
      p_mkt_buy -= actual_shift;
    }
  }
  
  // Compute cumulative thresholds
  double c1 = mix.p_limit_buy;
  double c2 = c1 + mix.p_limit_sell;
  double c3 = c2 + p_mkt_buy;
  double c4 = c3 + p_mkt_sell;
  
  std::uniform_real_distribution<double> U(0.0, 1.0);
  double u = U(rng_);

  SimEvent ev{};
  ev.ts = t_curr_;
  ev.side = Side::Buy;

  if (u < c1) {
    ev.type = EventType::LimitBuy;
    ev.side = Side::Buy;
    ev.qty = draw_geometric_mean(rp.mean_limit_qty);
    ev.px  = decide_limit_price(Side::Buy);
  } else if (u < c2) {
    ev.type = EventType::LimitSell;
    ev.side = Side::Sell;
    ev.qty = draw_geometric_mean(rp.mean_limit_qty);
    ev.px  = decide_limit_price(Side::Sell);
  } else if (u < c3) {
    ev.type = EventType::MktBuy;
    ev.side = Side::Buy;
    ev.qty = draw_geometric_mean(rp.mean_market_qty);
  } else if (u < c4) {
    ev.type = EventType::MktSell;
    ev.side = Side::Sell;
    ev.qty = draw_geometric_mean(rp.mean_market_qty);
  } else {
    ev.type = EventType::Cancel;
    OrderId target = sample_live();
    if (target) ev.cancel_id = target;

    // If nothing to cancel, opportunistically create a limit instead
    if (!ev.cancel_id) {
      std::bernoulli_distribution B(0.5);
      ev.type = B(rng_) ? EventType::LimitBuy : EventType::LimitSell;
      ev.side = (ev.type == EventType::LimitBuy) ? Side::Buy : Side::Sell;
      ev.qty  = draw_geometric_mean(rp.mean_limit_qty);
      ev.px   = decide_limit_price(ev.side);
    }
  }

  return ev;
}

// =============================================================================
// MODIFIED: execute now takes fills by reference instead of discarding them
// =============================================================================
void Simulator::execute(const SimEvent& e, std::vector<Fill>& fills) {
  // NOTE: fills is provided by caller, we append to it
  // The caller keeps the fills after this function returns

  switch (e.type) {
    case EventType::LimitBuy: {
      int k = 0;
      if (!ob_.bids.empty() && !ob_.asks.empty() && e.px) {
        int bb  = ob_.best_bid();
        int ba  = ob_.best_ask();
        int mid = (bb + ba) / 2;
        int off = *e.px - mid;          // positive if above mid
        k = std::abs(off);
        if (cfg_.max_offset_ticks > 0 && k > cfg_.max_offset_ticks) k = cfg_.max_offset_ticks;

        ++limit_offset_count_;
        limit_offset_abs_sum_ += k;
        if (k < (int)limit_offset_hist_.size()) ++limit_offset_hist_[k];
      }

      int bucket = bucket_of(k);
      ++lim_total_[bucket];

      OrderId id = me_.submit_limit(Side::Buy, *e.px, e.qty, e.ts, fills);
      if (ob_.index.count(id)) lim_bucket_by_id_[id] = bucket;
      live_add_if_resting(id);
      break;
    }
    case EventType::LimitSell: {
      int k = 0;
      if (!ob_.bids.empty() && !ob_.asks.empty() && e.px) {
        int bb  = ob_.best_bid();
        int ba  = ob_.best_ask();
        int mid = (bb + ba) / 2;
        int off = mid - *e.px;          // positive if below mid
        k = std::abs(off);
        if (cfg_.max_offset_ticks > 0 && k > cfg_.max_offset_ticks) k = cfg_.max_offset_ticks;

        ++limit_offset_count_;
        limit_offset_abs_sum_ += k;
        if (k < (int)limit_offset_hist_.size()) ++limit_offset_hist_[k];
      }

      int bucket = bucket_of(k);
      ++lim_total_[bucket];

      OrderId id = me_.submit_limit(Side::Sell, *e.px, e.qty, e.ts, fills);
      if (ob_.index.count(id)) lim_bucket_by_id_[id] = bucket;
      live_add_if_resting(id);
      break;
    }
    case EventType::MktBuy: {
      int mid0 = mid_ticks();
      me_.submit_market(Side::Buy, e.qty, e.ts, fills);
      // VWAP of fills
      double vsum = 0.0; uint64_t qsum = 0;
      for (auto& f : fills) { vsum += double(f.price) * f.qty; qsum += f.qty; }
      if (qsum) {
        double vwap = vsum / double(qsum);
        double slip = (vwap - mid0);      // buy pays above mid ⇒ positive
        mo_buy_slip_ += slip * qsum;
        mo_buy_qty_  += qsum;
        
        // Apply price impact: buy pressure pushes price up
        const RegimeParams& rp = (regime_ == Regime::Low) 
                               ? cfg_.regime.low : cfg_.regime.high;
        price_drift_ += rp.impact_coeff * static_cast<double>(qsum);
      }
      break;
    }
    case EventType::MktSell: {
      int mid0 = mid_ticks();
      me_.submit_market(Side::Sell, e.qty, e.ts, fills);
      double vsum = 0.0; uint64_t qsum = 0;
      for (auto& f : fills) { vsum += double(f.price) * f.qty; qsum += f.qty; }
      if (qsum) {
        double vwap = vsum / double(qsum);
        double slip = (mid0 - vwap);      // sell receives below mid ⇒ positive
        mo_sell_slip_ += slip * qsum;
        mo_sell_qty_  += qsum;
        
        // Apply price impact: sell pressure pushes price down
        const RegimeParams& rp = (regime_ == Regime::Low) 
                               ? cfg_.regime.low : cfg_.regime.high;
        price_drift_ -= rp.impact_coeff * static_cast<double>(qsum);
      }
      break;
    }
    case EventType::Cancel:
      if (e.cancel_id) {
        ob_.cancel(*e.cancel_id);     // <— MatchingEngine has no submit_cancel, cancel at book
        live_remove(*e.cancel_id);
      }
      break;
  }

  if (!fills.empty() && cfg_.log_trades) {
    for (const auto& f : fills) {
      std::cout << "TRADE t=" << f.ts
                << " taker=" << f.taker_id
                << " maker=" << f.maker_id
                << " side="  << (f.taker_side == Side::Buy ? 'B' : 'S')
                << " px="    << f.price
                << " qty="   << f.qty
                << "\n";
    }
  }

  for (const auto& f : fills) {
    auto it = lim_bucket_by_id_.find(f.maker_id);
    if (it != lim_bucket_by_id_.end()) {
      ++lim_filled_[it->second];
      // count "order ever got a fill" only once:
      lim_bucket_by_id_.erase(it);
    }
  }

  // ---- Toxicity trigger ----
  // Check if any fill involved an agent order as maker
  // IMPORTANT: Do this BEFORE cleaning up agent_order_ids_
  for (const auto& f : fills) {
    if (agent_order_ids_.count(f.maker_id)) {
      trigger_toxicity_from_fill(f);
    }
  }

  // Remove makers from live set only if fully filled (no longer in book)
  for (const auto& f : fills) {
    if (ob_.index.find(f.maker_id) == ob_.index.end()) {
      live_remove(f.maker_id);
      agent_order_ids_.erase(f.maker_id);  // Clean up agent tracking
    }
  }

  // ---- Telemetry updates ----
  ++n_events_;
  switch (e.type) {
    case EventType::LimitBuy:
    case EventType::LimitSell: ++n_limits_;  break;
    case EventType::MktBuy:
    case EventType::MktSell:  ++n_markets_; break;
    case EventType::Cancel:   ++n_cancels_; break;
  }

  // ---- Price drift decay (from market impact) ----
  const RegimeParams& rp = (regime_ == Regime::Low) 
                         ? cfg_.regime.low : cfg_.regime.high;
  price_drift_ *= (1.0 - rp.impact_decay);

  // ---- Toxicity: apply price drift and countdown ----
  if (toxic_events_remaining_ > 0) {
    // Apply directional drift to price_drift_ (accumulates)
    price_drift_ += toxic_price_drift_;
    
    --toxic_events_remaining_;
    if (toxic_events_remaining_ == 0) {
      toxic_direction_ = ToxicDirection::None;
      toxic_price_drift_ = 0.0;
    }
  }

  // Track spread (only if both sides exist)
  if (!ob_.bids.empty() && !ob_.asks.empty()) {
    int bb  = ob_.best_bid();
    int ba  = ob_.best_ask();
    int mid = (bb + ba) / 2;

    // existing spread metric
    sum_spread_ += (ba - bb);

    // new mid stats
    sum_mid_ += mid;
    ++mid_samples_;

    // max drawdown tracking
    if (mid > peak_mid_) peak_mid_ = mid;
    int dd = peak_mid_ - mid;
    if (dd > max_drawdown_) max_drawdown_ = dd;
  }
  // If you have fills, count trades/volume and drop makers that went to zero
  if (!fills.empty()) {
    for (const auto& f : fills) {
      ++n_trades_;
      vol_traded_ += f.qty;
      live_remove(f.maker_id);
    }
  }
  
  // NOTE: We do NOT clear fills here - caller keeps them
}


void Simulator::run() {
  std::cout << "[sim] start (max_events=" << cfg_.max_events << ")\n";

  for (size_t i = 0; i < cfg_.max_events; ++i) {
    SimEvent e = next_event();
    
    // For standalone run(), we create a local fills vector
    std::vector<Fill> fills;
    execute(e, fills);

    // heartbeat every 10k events so you know it's alive
    if (((i + 1) % 10000) == 0) {
      std::cout << "[sim] processed " << (i + 1) << " events\n";
    }

    // snapshots
    if (cfg_.snapshot_every && ((i + 1) % cfg_.snapshot_every == 0)) {
      std::cout << "\n--- snapshot @" << (i + 1) << " events ---\n";
    }
  }

  double avg_mid     = (mid_samples_ ? (sum_mid_ / double(mid_samples_)) : 0.0);
  double slip_buy_vw = (mo_buy_qty_  ? (mo_buy_slip_  / double(mo_buy_qty_))  : 0.0);
  double slip_sell_vw= (mo_sell_qty_ ? (mo_sell_slip_ / double(mo_sell_qty_)) : 0.0);

  // bucket ratios
  auto pct = [](uint64_t num, uint64_t den)->double {
    return den ? (100.0 * double(num) / double(den)) : 0.0;
  };

  std::cout << "avg_mid=" << avg_mid
            << " max_drawdown_ticks=" << max_drawdown_
            << " mo_slip_buy_vw=" << slip_buy_vw
            << " mo_slip_sell_vw=" << slip_sell_vw
            << "\n";

  static const char* BKT[5] = {"0","1-2","3-5","6-10",">10"};
  for (int i = 0; i < 5; ++i) {
    std::cout << "limit_fill_ratio_bucket[" << BKT[i] << "] "
              << lim_filled_[i] << "/" << lim_total_[i]
              << " (" << pct(lim_filled_[i], lim_total_[i]) << "%)\n";
  }

  // final summary
  double avg_spread = n_events_ ? (static_cast<double>(sum_spread_) / n_events_) : 0.0;
  std::cout << "\n=== SIM DONE ===\n"
            << "events="   << n_events_
            << " limits="  << n_limits_
            << " markets=" << n_markets_
            << " cancels=" << n_cancels_
            << " trades="  << n_trades_
            << " vol="     << vol_traded_
            << " avg_spread=" << avg_spread
            << "\n";
}

// =============================================================================
// Agent Order Helpers
// =============================================================================
// These maintain live_ids_ bookkeeping so agent orders are tracked consistently
// with background market orders. This means agent orders CAN be randomly
// canceled by simulator cancel events, which is realistic behavior.
//
// Agent orders are also tracked in agent_order_ids_ for adverse selection.
// =============================================================================

OrderId Simulator::submit_agent_order(Side side, Price price, Qty quantity,
                                       TimePoint ts, std::vector<Fill>& fills) {
  // Step 1: Submit the limit order through the matching engine
  OrderId id = me_.submit_limit(side, price, quantity, ts, fills);
  
  // Step 2: Register as agent order
  agent_order_ids_.insert(id);
  
  // Step 3: Add to live order tracking if it's resting
  live_add_if_resting(id);
  
  // Step 4: Remove filled makers from live tracking only if fully filled
  // (A partial fill leaves the maker still resting in the book)
  for (const auto& f : fills) {
    if (ob_.index.find(f.maker_id) == ob_.index.end()) {
      live_remove(f.maker_id);
      agent_order_ids_.erase(f.maker_id);  // Clean up if agent order was maker
    }
  }
  
  // Step 5: If this order was fully filled immediately, unregister it
  if (ob_.index.find(id) == ob_.index.end()) {
    agent_order_ids_.erase(id);
  }
  
  // Step 6: Return the order ID
  return id;
}

void Simulator::cancel_agent_order(OrderId id) {
  // Step 1: Cancel the order in the order book
  ob_.cancel(id);
  
  // Step 2: Remove from live order tracking
  live_remove(id);
  
  // Step 3: Unregister as agent order
  agent_order_ids_.erase(id);
}

// =============================================================================
// Toxicity Trigger
// =============================================================================
// Called when an agent order is filled as maker. Triggers a temporary bias
// in market order direction against the agent's new position.
// =============================================================================

void Simulator::trigger_toxicity_from_fill(const Fill& f) {
  const RegimeParams& rp = (regime_ == Regime::Low) 
                         ? cfg_.regime.low : cfg_.regime.high;
  
  // Probabilistically trigger toxicity
  std::bernoulli_distribution trigger(rp.toxicity_prob);
  if (!trigger(rng_)) return;
  
  // If taker bought, agent (maker) sold -> agent is now short
  // We want price to go UP to hurt them (adverse selection)
  // If taker sold, agent (maker) bought -> agent is now long
  // We want price to go DOWN to hurt them (adverse selection)
  
  if (f.taker_side == Side::Buy) {
    // Agent sold -> drift UP to hurt them
    toxic_direction_ = ToxicDirection::Up;
    toxic_price_drift_ = +rp.toxicity_drift;
  } else {
    // Agent bought -> drift DOWN to hurt them
    toxic_direction_ = ToxicDirection::Down;
    toxic_price_drift_ = -rp.toxicity_drift;
  }
  
  toxic_events_remaining_ = rp.toxicity_duration;
}