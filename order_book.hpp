#pragma once
#include "order.hpp"

// A flexible list that holds Orders, where I can add/remove from both ends
using LevelQueue = std::deque<Order>;

struct OrderBook {
  std::map<Price, LevelQueue, std::greater<Price>> bids;
  std::map<Price, LevelQueue, std::less<Price>> asks;

  struct IndexEntry { 
    Side side; // buy or sell
    Price px;  // price of order
    size_t pos; // position in list
  };
  std::unordered_map<OrderId, IndexEntry> index; // hash table

  // ternary operator (one-line if)
    // condition ? value_if_true : value_if_false
  Price best_bid() const { 
    return bids.empty() ? 0 : bids.begin()->first; 
  }
  Price best_ask() const { 
    return asks.empty () ? 0 : asks.begin()->first; 
  }
  Price mid() const {
    if (bids.empty() || asks.empty()) return 0;
    return (best_bid() + best_ask()) / 2;
  }

  // &: don't copy whole thing, just refer to it
  // o: name of parameter inside function
  void add_limit(const Order& o);
  void cancel(OrderId id);

  bool self_check() const;
};