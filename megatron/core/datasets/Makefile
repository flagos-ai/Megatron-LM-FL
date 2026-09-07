CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python3 -m pybind11 --includes 2>/dev/null | tail -n 1)
LIBNAME = helpers_cpp
LIBEXT = $(shell python3-config --extension-suffix)

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
