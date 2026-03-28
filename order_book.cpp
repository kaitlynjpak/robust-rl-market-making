#include "order_book.hpp"
#include <stdexcept> // throw "thats not allowed" errors
#include <cstddef> // std::size_t, std::ptrdiff_t, etc
#include <iterator> 
#include <cassert>


/*
When a limit order shows up:

Make sure it really is a Limit (not Market/Cancel). If not, we throw.

Make sure no duplicate IDs.

Make sure quantity and price are positive.

If Buy, put the order at the back of the buy queue.
If it’s a Sell, do the same on the sell side.

Remember where it was in a hash map: for this OrderId, store its side, price, and position in the queue at that price level.

So the order book is like shelves by price, each shelf is a line (deque) of orders in FIFO. We put a new order on the right shelf, at the end of the line, and write down exactly where we put it so we can find it later.
*/

void OrderBook::add_limit(const Order& o) {
  if (o.type != OrdType::Limit) {
    throw std::invalid_argument("add_limit expects OrdType::Limit");
  }
  if (index.find(o.id) != index.end()) {
    throw std::invalid_argument("Duplicate OrderId");
  }
  if (o.qty <= 0) {
    throw std::invalid_argument("qty must be positive");
  }
  if (o.limit_price <= 0) {
    throw std::invalid_argument("limit_price must be > 0");
  }

  // Choose the side explicitly; avoid mixing comparator types
  if (o.side == Side::Buy) {
    auto& q = bids[o.limit_price];   // creates level if missing
    q.push_back(o);
    const std::size_t pos = q.size() - 1;
    index.emplace(o.id, IndexEntry{ Side::Buy, o.limit_price, pos });
  } else {
    auto& q = asks[o.limit_price];
    q.push_back(o);
    const std::size_t pos = q.size() - 1;
    index.emplace(o.id, IndexEntry{ Side::Sell, o.limit_price, pos });
  }
}


/*
Want to cancel an order by its ID:

Look up the order in index. If don’t know, just stop.

From the index card, learn:
  Which side (Buy/Sell) it’s on,
  Which price shelf (px),
  Which spot in the line (pos).

Go to the correct shelf (buy or sell) at that price.
  If somehow that shelf doesn’t exist anymore, clean up the index and stop.
  If the position is nonsense (out of range), clean up the index and stop.

Remove the order from the line at that position.

Because we pulled a order out of the middle of the line, everyone behind it moves up by one. Update the index positions for those orders.

If the shelf became empty, remove the shelf entirely.

Finally, remove this order’s card from the index.
*/

void OrderBook::cancel(OrderId id) {
  auto it = index.find(id);
  if (it == index.end()) return; // not found

  const Side side = it->second.side;
  const Price px  = it->second.px;
  std::size_t pos = it->second.pos;

  // Get the right side’s map
  if (side == Side::Buy) {
    auto lvl_it = bids.find(px);
    if (lvl_it == bids.end()) { index.erase(it); return; }

    LevelQueue& q = lvl_it->second;
    if (pos >= q.size()) { index.erase(it); return; }

    auto erase_it = q.begin() + static_cast<std::ptrdiff_t>(pos);
    q.erase(erase_it);

    // Re-index subsequent orders at the same price level
    for (std::size_t p = pos; p < q.size(); ++p) {
      auto idx_it = index.find(q[p].id);
      if (idx_it != index.end()) idx_it->second.pos = p;
    }

    if (q.empty()) bids.erase(lvl_it);
  } else {
    auto lvl_it = asks.find(px);
    if (lvl_it == asks.end()) { index.erase(it); return; }

    LevelQueue& q = lvl_it->second;
    if (pos >= q.size()) { index.erase(it); return; }

    auto erase_it = q.begin() + static_cast<std::ptrdiff_t>(pos);
    q.erase(erase_it);

    for (std::size_t p = pos; p < q.size(); ++p) {
      auto idx_it = index.find(q[p].id);
      if (idx_it != index.end()) idx_it->second.pos = p;
    }

    if (q.empty()) asks.erase(lvl_it);
  }

index.erase(it);
}


/*
Ensures bid/ask book state and the index state are perfectly synchronized, catching any data corruption or stale references that may occur during operations like add_limit or cancel.
*/

bool OrderBook::self_check() const {
  // 1) Every order in the book must appear in index with correct side/px/pos
  auto check_side = [&](const auto& bookSide, Side side) -> bool {
    for (const auto& [px, q] : bookSide) {
      for (std::size_t p = 0; p < q.size(); ++p) {
        const auto& o = q[p];
        auto it = index.find(o.id);
        if (it == index.end()) return false;
        const auto& e = it->second;
        if (e.side != side || e.px != px || e.pos != p) return false;
      }
    }
    return true;
  };

  if (!check_side(bids, Side::Buy))  return false;
  if (!check_side(asks, Side::Sell)) return false;

  // 2) Every index entry must point to a real order in the right place
  for (const auto& [id, e] : index) {
    if (e.side == Side::Buy) {
      auto lvl = bids.find(e.px);
      if (lvl == bids.end()) return false;
      const auto& q = lvl->second;
      if (e.pos >= q.size()) return false;
      if (q[e.pos].id != id) return false;
    } else { // Side::Sell
      auto lvl = asks.find(e.px);
      if (lvl == asks.end()) return false;
      const auto& q = lvl->second;
      if (e.pos >= q.size()) return false;
      if (q[e.pos].id != id) return false;
    }
  }

  return true;
}
