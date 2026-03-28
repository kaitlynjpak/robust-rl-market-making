#pragma once              // instructions to compiler: don't duplicate
#include <cstdint>        // fixed-size number types
#include <deque>          // add/remove from both front and back
#include <map>            // like a dictionary
#include <unordered_map>  // faster map
#include <optional>       
#include <string>

// using is like a nickname
using Price = int64_t;    // ticks
using Qty   = int64_t;    // units
using OrderId = uint64_t;
using TimePoint = double; // simulation time in seconds (decimals)

// like choosing from a menu
enum class Side : uint8_t { Buy, Sell };
enum class OrdType : uint8_t { Limit, Market, Cancel };