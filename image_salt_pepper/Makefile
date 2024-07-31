
CXX = icpx

CXXFLAGS=-O3 -fsycl

LIBS=


main: main.cpp kernels.cpp 
	$(CXX) $^ $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f main
