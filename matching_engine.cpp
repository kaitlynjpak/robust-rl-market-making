#include "matching_engine.hpp"
#include <algorithm>   // std::min
#include <stdexcept>   // std::invalid_argument

OrderId MatchingEngine::submit_market(Side s, Qty q, TimePoint t, std::vector<Fill>& out) {
  if (q <= 0) throw std::invalid_argument("market qty must be > 0");
  OrderId id = next_id++;
  if (s == Side::Buy)  match_buy(id, q, t, out, std::nullopt);
  else                 match_sell(id, q, t, out, std::nullopt);
  return id; // market remainder is discarded
}

OrderId MatchingEngine::submit_limit(Side s, Price px, Qty q, TimePoint t, std::vector<Fill>& out) {
  if (q <= 0) throw std::invalid_argument("limit qty must be > 0");
  if (px <= 0) throw std::invalid_argument("limit price must be > 0");
  OrderId id = next_id++;
  if (s == Side::Buy)  match_buy(id, q, t, out, px);
  else                 match_sell(id, q, t, out, px);
  if (q > 0) {
    Order o{ id, s, OrdType::Limit, px, q, t };
    book.add_limit(o);
  }
  return id;
}

void MatchingEngine::match_buy(OrderId taker_id, Qty& remaining, TimePoint t,
                               std::vector<Fill>& out, std::optional<Price> limit_px) {
  while (remaining > 0 && !book.asks.empty()) {
    auto it = book.asks.begin();           // best ask
    Price ask_px = it->first;
    if (limit_px && *limit_px < ask_px) break; // limit gate

    LevelQueue& q = it->second;            // FIFO at ask_px
    while (remaining > 0 && !q.empty()) {
      Order& maker = q.front();
      Qty traded = std::min(remaining, maker.qty);

      out.push_back(Fill{ taker_id, maker.id, Side::Buy, ask_px, traded, t });

      maker.qty -= traded;
      remaining -= traded;

      if (maker.qty == 0) {
        book.index.erase(maker.id);
        q.pop_front();
        reindex_after_pop_front(q, book, Side::Sell, ask_px);
      }
    }
    if (q.empty()) book.asks.erase(it);
  }
}

void MatchingEngine::match_sell(OrderId taker_id, Qty& remaining, TimePoint t,
                                std::vector<Fill>& out, std::optional<Price> limit_px) {
  while (remaining > 0 && !book.bids.empty()) {
    auto it = book.bids.begin();           // best bid
    Price bid_px = it->first;
    if (limit_px && *limit_px > bid_px) break; // limit gate

    LevelQueue& q = it->second;            // FIFO at bid_px
    while (remaining > 0 && !q.empty()) {
      Order& maker = q.front();
      Qty traded = std::min(remaining, maker.qty);

      out.push_back(Fill{ taker_id, maker.id, Side::Sell, bid_px, traded, t });

      maker.qty -= traded;
      remaining -= traded;

      if (maker.qty == 0) {
        book.index.erase(maker.id);
        q.pop_front();
        reindex_after_pop_front(q, book, Side::Buy, bid_px);
      }
    }
    if (q.empty()) book.bids.erase(it);
  }
}