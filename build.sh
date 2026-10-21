#!/bin/sh
# Builds the C++ test suite and the Python extension module into build/.
set -e
mkdir -p build
g++ -O2 -std=c++17 -Wall -Wextra -Isrc tests/test_qlite.cpp -o build/test_qlite
g++ -O3 -std=c++17 -shared -fPIC -Isrc $(python -m pybind11 --includes) src/bindings.cpp \
    -o build/qlite$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo built
