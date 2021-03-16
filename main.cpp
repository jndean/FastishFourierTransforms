#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "FiFT.h"





int main(int argc, char** argv) {

    const size_t burst_size = 512;
    const size_t batch_size = 50;

    int num_elts = burst_size * batch_size;
    

    // Allocate memory //
    REAL_T *h_input, *d_input;
    COMPLEX_T *h_output, *d_output;
    h_input = new REAL_T[num_elts];
    h_output = new COMPLEX_T[num_elts];
    checkCudaErrors(cudaMalloc((void**)&d_input, num_elts * sizeof(REAL_T)));
    checkCudaErrors(cudaMalloc((void**)&d_output, num_elts * sizeof(COMPLEX_T)));

    
    // Create and transfer input data //
    for (size_t t = 0; t < burst_size; ++t) {
	for (size_t b = 0; b < batch_size; ++b) {
	    h_input[t * batch_size + b] = 128 + 100 * sin(2 * 3.14159 * t / (float)(b + 1));
	}
    }
    cudaMemcpy(d_input, h_input, num_elts * sizeof(REAL_T), cudaMemcpyHostToDevice);


    // Run the FFT //
    FiFT fift(burst_size, batch_size);
    fift.run(d_input, d_output);

    
    cudaMemcpy(h_output, d_output, num_elts * sizeof(COMPLEX_T), cudaMemcpyDeviceToHost);

    
    cudaDeviceSynchronize();
    
    // Cleanup //
    delete[] h_input, h_output;
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors( cudaFree(d_output));
    
    return EXIT_SUCCESS;
}
