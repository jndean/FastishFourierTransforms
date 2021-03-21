
TARGET = main
INCDIR = -I/usr/local/cuda-11.0/targets/x86_64-linux/include/ \
	 -I/usr/local/cuda-11.0/samples/common/inc/
CFLAGS = -O3 ${INCDIR} -std=c++11 -g

default: FiFT.o main.o
	nvcc ${CFLAGS}  $^ -o ${TARGET}

%.o: %.cu FiFT.h
	nvcc $(CFLAGS) -c $< -o $@

%.o: %.cpp FiFT.h
	g++ $(CFLAGS) -c $< -o $@


FiFT.so: FiFT.cu FiFT.h
	nvcc $(CFLAGS)  --compiler-options="-fPIC -shared" FiFT.cu -o FiFT.so


clean:
	rm *.o *~ *.so
