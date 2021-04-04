
TARGET = main
INCDIR = -I/usr/local/cuda-9.0/include/ \
	 -I/usr/local/cuda-9.0/samples/common/inc/
NVCC = /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -O3 ${INCDIR} -std=c++11 -g -lineinfo

default: FiFT.o main.o
	${NVCC} ${CFLAGS}  $^ -o ${TARGET}

%.o: %.cu FiFT.h
	${NVCC} $(CFLAGS) -c $< -o $@

%.o: %.cpp FiFT.h
	g++ $(CFLAGS) -c $< -o $@


FiFT.so: FiFT.cu FiFT.h
	${NVCC} $(CFLAGS)  --compiler-options="-fPIC -shared" FiFT.cu -o FiFT.so


clean:
	rm *.o *~ *.so
