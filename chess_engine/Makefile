# Makefile for AIChessbot C++ engine.
#
#   make        — builds optimized release binary (chess_engine.exe / chess_engine).
#   make debug  — debug build with -O0 -g.
#   make clean  — removes objects + binary.
#   make perft  — builds, then runs `perft 5` from startpos.
#   make bench  — builds, then runs the standard bench (depth 13).

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -DNDEBUG -Wall -Wextra -Wno-unused-parameter -pthread
LDFLAGS  ?= -pthread

ifeq ($(OS),Windows_NT)
    EXE := chess_engine.exe
else
    EXE := chess_engine
endif

SRC_DIR := src
SRCS    := $(wildcard $(SRC_DIR)/*.cpp) $(SRC_DIR)/fathom/tbprobe.cpp
OBJS    := $(SRCS:.cpp=.o)

# Fathom (Syzygy) compile flags: silence its many -Wall warnings, allow
# its C-style casts and unused returns without polluting our own build noise.
$(SRC_DIR)/fathom/tbprobe.o: CXXFLAGS += -Wno-all -Wno-extra -Wno-unused-but-set-variable \
    -Wno-sign-compare -Wno-unused-function -Wno-unused-variable \
    -Wno-strict-aliasing -Wno-format -Wno-pointer-arith \
    -Wno-implicit-fallthrough -Wno-deprecated-declarations \
    -Wno-class-memaccess -Wno-conversion-null -Wno-cast-function-type

.PHONY: all debug clean perft bench tuner

all: $(EXE)

# Texel tuner — links engine objects sans main.o.
TUNER_OBJS := $(SRC_DIR)/board.o $(SRC_DIR)/movegen.o $(SRC_DIR)/evaluate.o \
              $(SRC_DIR)/nnue.o $(SRC_DIR)/tt.o
tuner: $(TUNER_OBJS) tuner/tuner.cpp
	$(CXX) $(CXXFLAGS) -o tuner/tuner.exe tuner/tuner.cpp $(TUNER_OBJS) $(LDFLAGS)

$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

debug: CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra -pthread
debug: clean $(EXE)

perft: $(EXE)
	echo "position startpos\nperft 5\nquit" | ./$(EXE)

bench: $(EXE)
	./$(EXE) bench 13

clean:
	rm -f $(SRC_DIR)/*.o $(EXE)
