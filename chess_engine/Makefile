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
SRCS    := $(wildcard $(SRC_DIR)/*.cpp)
OBJS    := $(SRCS:.cpp=.o)

.PHONY: all debug clean perft bench

all: $(EXE)

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
