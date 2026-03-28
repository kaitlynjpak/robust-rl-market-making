#pragma once
#include "order_book.hpp"
#include <vector>
#include <optional>

struct Fill {
  OrderId taker_id;  // active, incoming order
  OrderId maker_id;  // resting order that got hit
  Side taker_side;   // buy if taker is buying, sell if taker is selling
  Price price;       // execution price (maker's level)
  Qty qty;           // traded qty
  TimePoint ts;      // trade time   
};

struct MatchingEngine {
  explicit MatchingEngine(OrderBook& ob) : book(ob) {}

  // Public API
  OrderId submit_market(Side s, Qty q, TimePoint t, std::vector<Fill>& out);
  OrderId submit_limit (Side s, Price px, Qty q, TimePoint t, std::vector<Fill>& out);

private:
  // MUST match src/matching_engine.cpp exactly:
  void match_buy (OrderId taker_id, Qty& remaining, TimePoint t,
                  std::vector<Fill>& out, std::optional<Price> limit_px);
  void match_sell(OrderId taker_id, Qty& remaining, TimePoint t,
                  std::vector<Fill>& out, std::optional<Price> limit_px);

  // Inline helper to reindex positions after pop_front
  static void reindex_after_pop_front(LevelQueue& q, OrderBook& book, Side /*side*/, Price /*px*/) {
    for (std::size_t p = 0; p < q.size(); ++p) {
      auto it = book.index.find(q[p].id);
      if (it != book.index.end()) it->second.pos = p;
    }
  }

  OrderId next_id{1};
  OrderBook& book;
};