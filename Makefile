CXX := g++
CXXFLAGS := -Wall -std=c++2a -MD -MP -fopenmp -O2
LDFLAGS := -lstdc++ -lOpenCL -fopenmp
TARGET := executable

$(TARGET): $(patsubst %.cpp,%.o,$(wildcard *.cpp))
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

-include $(wildcard *.d)

clean:
	@ rm *.o
	@ rm *.d

run:
	@ ./$(TARGET)
