#pragma once
#include "types.hpp"

struct Order {
    OrderId id;
    Side side;
    OrdType type;
    Price limit_price; // ignored for market orders
    Qty qty;
    TimePoint ts;
};

