// python_bindings.cpp
// 
// Python bindings for the C++ market-making RL environment.
// Uses pybind11 to expose a Gym-like interface to Python.
//
// =============================================================================
// ACTION SPACE DESIGN DECISION (v1 - SAC Training)
// =============================================================================
// The Python wrapper INTENTIONALLY restricts the agent's action space to a
// conservative subset of the full simulator bounds for easier initial training.
//
// Simulator supports:
//   - offsets: [0, 10] ticks
//   - sizes: [1, unlimited]
//
// RL wrapper exposes (v1):
//   - offsets: [0, 5] ticks  (rl_max_offset_)
//   - sizes: [1, 3] lots     (rl_max_size_)
//
// This is deliberate, not accidental. Wider bounds can be enabled later.
// =============================================================================
//
// Build:
//   c++ -O3 -shared -std=c++17 -fPIC -undefined dynamic_lookup \
//       $(python3 -m pybind11 --includes) \
//       python_bindings.cpp market_making_env.cpp sim.cpp matching_engine.cpp order_book.cpp \
//       -o lob_env$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "market_making_env.h"

#include <vector>
#include <stdexcept>
#include <cmath>
#include <string>

namespace py = pybind11;

// ============================================================================
// PyMarketMakingEnv: Thin Python-facing wrapper
// ============================================================================
// Responsibilities:
//   1. Action conversion: normalized [-1, 1] -> discrete C++ Action
//   2. Observation conversion: C++ Observation -> numpy array
//   3. Step result packaging: (obs, reward, done, truncated, info)
//
// Does NOT contain: replay buffer, actor/critic networks, learning logic
// ============================================================================

class PyMarketMakingEnv {
public:
    // ========================================================================
    // Constructor
    // ========================================================================
    // Parameters:
    //   seed: Random seed for reproducibility
    //   episode_duration: Episode length in seconds
    //   decision_interval: Time between decisions in seconds
    //   inventory_penalty: Coefficient for inventory penalty in reward
    //   regime_mode: "high", "low", or "mixed" (default: "high")
    //   toxicity_prob: Probability of toxicity trigger after agent-maker fill
    //   toxicity_duration: Duration of toxicity in events
    //   toxicity_strength: Probability shift during toxicity
    //   toxicity_drift: Price drift per event during toxicity
    //   rl_max_offset: Max offset for RL actions (default: 5, can widen later)
    //   rl_max_size: Max size for RL actions (default: 3, can widen later)
    //   reward_scale: Divide delta_pnl by this (default: 1000.0 for RL training)
    // ========================================================================
    PyMarketMakingEnv(
        int seed = 42,
        double episode_duration = 10.0,
        double decision_interval = 0.1,
        double inventory_penalty = 0.01,
        const std::string& regime_mode = "high",
        double toxicity_prob = 1.0,
        int toxicity_duration = 10,
        double toxicity_strength = 0.15,
        double toxicity_drift = 0.05,
        int rl_max_offset = 5,
        int rl_max_size = 3,
        double reward_scale = 1000.0
    ) : rl_max_offset_(rl_max_offset), rl_max_size_(rl_max_size) {
        
        // ====================================================================
        // Environment config
        // ====================================================================
        config_.episode_duration = episode_duration;
        config_.decision_interval = decision_interval;
        config_.inventory_penalty_coeff = inventory_penalty;
        config_.reward_scale = reward_scale;
        
        // These are the ENV bounds (what the simulator supports)
        // The RL bounds (rl_max_offset_, rl_max_size_) are separate
        config_.max_offset = 10;  // Simulator supports up to 10
        config_.max_size = 10;    // Simulator supports larger sizes
        
        // ====================================================================
        // Simulator config
        // ====================================================================
        config_.sim_config.initial_mid_ticks = 1000;
        config_.sim_config.min_price_ticks = 1;
        config_.sim_config.max_offset_ticks = 10;
        config_.sim_config.geolap_alpha = 0.5;
        config_.sim_config.seed = seed;
        
        // ====================================================================
        // Regime mode
        // ====================================================================
        if (regime_mode == "high") {
            // Force high regime
            config_.sim_config.regime.p_LL = 0.0;
            config_.sim_config.regime.p_HH = 1.0;
        } else if (regime_mode == "low") {
            // Force low regime
            config_.sim_config.regime.p_LL = 1.0;
            config_.sim_config.regime.p_HH = 0.0;
        } else if (regime_mode == "mixed") {
            // Allow regime switching
            config_.sim_config.regime.p_LL = 0.995;
            config_.sim_config.regime.p_HH = 0.990;
        } else {
            throw std::runtime_error("regime_mode must be 'high', 'low', or 'mixed'");
        }
        
        // ====================================================================
        // Low regime parameters
        // ====================================================================
        config_.sim_config.regime.low.lambda = 1000.0;
        config_.sim_config.regime.low.mix = {0.35, 0.35, 0.10, 0.10, 0.10};
        config_.sim_config.regime.low.mean_limit_qty = 5.0;
        config_.sim_config.regime.low.mean_market_qty = 3.0;
        config_.sim_config.regime.low.toxicity_prob = 0.0;  // No toxicity in low
        config_.sim_config.regime.low.toxicity_duration = 0;
        config_.sim_config.regime.low.toxicity_strength = 0.0;
        config_.sim_config.regime.low.toxicity_drift = 0.0;
        config_.sim_config.regime.low.impact_decay = 0.0;
        
        // ====================================================================
        // High regime parameters
        // ====================================================================
        config_.sim_config.regime.high.lambda = 2000.0;
        config_.sim_config.regime.high.mix = {0.15, 0.15, 0.30, 0.30, 0.10};
        config_.sim_config.regime.high.mean_limit_qty = 5.0;
        config_.sim_config.regime.high.mean_market_qty = 8.0;
        
        // Toxicity settings (only in high regime)
        config_.sim_config.regime.high.toxicity_prob = toxicity_prob;
        config_.sim_config.regime.high.toxicity_duration = toxicity_duration;
        config_.sim_config.regime.high.toxicity_strength = toxicity_strength;
        config_.sim_config.regime.high.toxicity_drift = toxicity_drift;
        config_.sim_config.regime.high.impact_decay = 0.05;
        
        // ====================================================================
        // Create environment
        // ====================================================================
        env_ = std::make_unique<rl::MarketMakingEnv>(config_);
    }
    
    // ========================================================================
    // reset() -> numpy array of shape (6,)
    // ========================================================================
    py::array_t<double> reset() {
        rl::Observation obs = env_->reset();
        return obs_to_numpy(obs);
    }
    
    // ========================================================================
    // reset_with_seed(seed) -> numpy array
    // ========================================================================
    py::array_t<double> reset_with_seed(int seed) {
        config_.sim_config.seed = seed;
        env_ = std::make_unique<rl::MarketMakingEnv>(config_);
        return reset();
    }
    
    // ========================================================================
    // step(action) -> (obs, reward, done, truncated, info)
    // ========================================================================
    // Action format: numpy array of 4 floats in [-1, 1]
    //   action[0]: bid_offset  [-1, 1] -> [0, rl_max_offset_]
    //   action[1]: ask_offset  [-1, 1] -> [0, rl_max_offset_]
    //   action[2]: bid_size    [-1, 1] -> [1, rl_max_size_]
    //   action[3]: ask_size    [-1, 1] -> [1, rl_max_size_]
    // ========================================================================
    py::tuple step(py::array_t<double> action_array) {
        auto buf = action_array.request();
        if (buf.ndim != 1 || buf.shape[0] != 4) {
            throw std::runtime_error("Action must be a 1D array of length 4");
        }
        
        double* action_ptr = static_cast<double*>(buf.ptr);
        std::vector<double> action_vec(action_ptr, action_ptr + 4);
        
        rl::Action action = map_action(action_vec);
        rl::StepResult result = env_->step(action);
        
        py::dict info;
        info["pnl"] = result.pnl;
        info["inventory"] = result.final_inventory;
        info["turnover"] = result.total_turnover;
        
        return py::make_tuple(
            obs_to_numpy(result.observation),
            result.reward,
            result.done,
            result.truncated,
            info
        );
    }
    
    // ========================================================================
    // Gym-compatible properties
    // ========================================================================
    
    py::tuple observation_shape() const {
        return py::make_tuple(6);
    }
    
    py::tuple action_shape() const {
        return py::make_tuple(4);
    }
    
    // Normalized action bounds (always [-1, 1])
    py::tuple action_bounds() const {
        std::vector<double> low = {-1.0, -1.0, -1.0, -1.0};
        std::vector<double> high = {1.0, 1.0, 1.0, 1.0};
        return py::make_tuple(low, high);
    }
    
    // Expose the actual RL action ranges (for documentation/debugging)
    py::dict action_ranges() const {
        py::dict ranges;
        ranges["bid_offset"] = py::make_tuple(0, rl_max_offset_);
        ranges["ask_offset"] = py::make_tuple(0, rl_max_offset_);
        ranges["bid_size"] = py::make_tuple(1, rl_max_size_);
        ranges["ask_size"] = py::make_tuple(1, rl_max_size_);
        return ranges;
    }
    
    // Debugging accessors
    double get_mid() const { return env_->get_mid(); }
    int64_t get_inventory() const { return env_->current_inventory(); }
    int get_rl_max_offset() const { return rl_max_offset_; }
    int get_rl_max_size() const { return rl_max_size_; }

private:
    rl::EnvConfig config_;
    std::unique_ptr<rl::MarketMakingEnv> env_;
    
    // ========================================================================
    // RL action bounds (deliberately restricted for v1 training)
    // These are SEPARATE from simulator bounds (config_.max_offset, etc.)
    // ========================================================================
    int rl_max_offset_;  // Default: 5 (simulator supports 10)
    int rl_max_size_;    // Default: 3 (simulator supports more)
    
    // ========================================================================
    // Action mapping: [-1, 1] -> C++ Action using RL bounds
    // ========================================================================
    rl::Action map_action(const std::vector<double>& a) const {
        rl::Action out;
        
        double a0 = clamp(a[0], -1.0, 1.0);
        double a1 = clamp(a[1], -1.0, 1.0);
        double a2 = clamp(a[2], -1.0, 1.0);
        double a3 = clamp(a[3], -1.0, 1.0);
        
        // Use RL bounds, not simulator bounds
        out.bid_offset = map_to_range(a0, 0, rl_max_offset_);
        out.ask_offset = map_to_range(a1, 0, rl_max_offset_);
        out.bid_size = map_to_range(a2, 1, rl_max_size_);
        out.ask_size = map_to_range(a3, 1, rl_max_size_);
        
        return out;
    }
    
    static double clamp(double x, double lo, double hi) {
        return (x < lo) ? lo : (x > hi) ? hi : x;
    }
    
    static int map_to_range(double x, int lo, int hi) {
        double u = 0.5 * (x + 1.0);  // [-1,1] -> [0,1]
        double y = static_cast<double>(lo) + u * static_cast<double>(hi - lo);
        return static_cast<int>(std::round(y));
    }
    
    static py::array_t<double> obs_to_numpy(const rl::Observation& obs) {
        auto arr = obs.to_array();
        py::array_t<double> result(arr.size());
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        for (size_t i = 0; i < arr.size(); ++i) {
            ptr[i] = arr[i];
        }
        return result;
    }
};

// ============================================================================
// Python module definition
// ============================================================================

PYBIND11_MODULE(lob_env, m) {
    m.doc() = R"doc(
Python bindings for the market-making RL environment.

Action Space (normalized to [-1, 1]):
    action[0]: bid_offset  -> [0, rl_max_offset] ticks below mid
    action[1]: ask_offset  -> [0, rl_max_offset] ticks above mid
    action[2]: bid_size    -> [1, rl_max_size] lots
    action[3]: ask_size    -> [1, rl_max_size] lots

Default RL bounds (v1 - conservative for initial training):
    rl_max_offset = 5  (simulator supports 10)
    rl_max_size = 3    (simulator supports more)

Observation Space (6 features):
    obs[0]: spread_ticks (normalized)
    obs[1]: recent_mid_return (scaled)
    obs[2]: volatility_proxy [0, 1]
    obs[3]: top_of_book_imbalance [-1, 1]
    obs[4]: normalized_inventory [-1, 1]
    obs[5]: time_since_last_fill (normalized)
)doc";
    
    py::class_<PyMarketMakingEnv>(m, "MarketMakingEnv")
        .def(py::init<int, double, double, double, const std::string&, 
                      double, int, double, double, int, int, double>(),
             py::arg("seed") = 42,
             py::arg("episode_duration") = 10.0,
             py::arg("decision_interval") = 0.1,
             py::arg("inventory_penalty") = 0.01,
             py::arg("regime_mode") = "high",
             py::arg("toxicity_prob") = 1.0,
             py::arg("toxicity_duration") = 10,
             py::arg("toxicity_strength") = 0.15,
             py::arg("toxicity_drift") = 0.05,
             py::arg("rl_max_offset") = 5,
             py::arg("rl_max_size") = 3,
             py::arg("reward_scale") = 1000.0,
             R"doc(
Create a market-making environment.

Args:
    seed: Random seed (default: 42)
    episode_duration: Episode length in seconds (default: 10.0)
    decision_interval: Time between decisions (default: 0.1)
    inventory_penalty: Inventory penalty coefficient (default: 0.01)
    regime_mode: 'high', 'low', or 'mixed' (default: 'high')
    toxicity_prob: Probability of toxicity trigger (default: 1.0)
    toxicity_duration: Duration of toxicity in events (default: 10)
    toxicity_strength: Probability shift during toxicity (default: 0.15)
    toxicity_drift: Price drift per event during toxicity (default: 0.05)
    rl_max_offset: Max offset for RL actions (default: 5)
    rl_max_size: Max size for RL actions (default: 3)
    reward_scale: Divide delta_pnl by this for RL training (default: 1000.0)
)doc")
        .def("reset", &PyMarketMakingEnv::reset,
             "Reset the environment and return initial observation.")
        .def("reset_with_seed", &PyMarketMakingEnv::reset_with_seed,
             py::arg("seed"),
             "Reset the environment with a new random seed.")
        .def("step", &PyMarketMakingEnv::step,
             py::arg("action"),
             "Take a step with normalized action [-1, 1]^4. Returns (obs, reward, done, truncated, info).")
        .def("observation_shape", &PyMarketMakingEnv::observation_shape,
             "Return observation space shape: (6,)")
        .def("action_shape", &PyMarketMakingEnv::action_shape,
             "Return action space shape: (4,)")
        .def("action_bounds", &PyMarketMakingEnv::action_bounds,
             "Return normalized action bounds: ([-1,-1,-1,-1], [1,1,1,1])")
        .def("action_ranges", &PyMarketMakingEnv::action_ranges,
             "Return actual RL action ranges as dict: {bid_offset: (0, max), ...}")
        .def("get_mid", &PyMarketMakingEnv::get_mid,
             "Get current mid price (for debugging)")
        .def("get_inventory", &PyMarketMakingEnv::get_inventory,
             "Get current inventory (for debugging)")
        .def("get_rl_max_offset", &PyMarketMakingEnv::get_rl_max_offset,
             "Get RL max offset bound")
        .def("get_rl_max_size", &PyMarketMakingEnv::get_rl_max_size,
             "Get RL max size bound");
}