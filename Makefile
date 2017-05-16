CXX	= g++
CXXFLAGS= -m64 -pipe -g -std=c++11 -Wall -W -fPIC -O3 -rdynamic -D_GNU_SOURCE
LFLAGS	= -m64
LIBS	= -lm -lgsl -lgslcblas -lpython3.5m
BIN	= Neural_Net

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all:
	$(CXX) $(SRC) $(LIBS) $(CXXFLAGS) -o $(BIN)

clean:
	rm -f $(BIN) *.o
