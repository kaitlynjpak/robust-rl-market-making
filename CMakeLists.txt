cmake_minimum_required(VERSION 3.14)
project(lob_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
# Option 1: If installed via pip (pip install pybind11)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Source files for the simulator and environment
set(ENV_SOURCES
    order_book.cpp
    matching_engine.cpp
    sim.cpp
    market_making_env.cpp
)

# Create the Python module
pybind11_add_module(lob_env
    python_bindings.cpp
    ${ENV_SOURCES}
)

# Include directories (adjust if your headers are elsewhere)
target_include_directories(lob_env PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optimization flags for release builds
target_compile_options(lob_env PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

# Installation (optional)
install(TARGETS lob_env DESTINATION .)