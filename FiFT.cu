#include "FiFT.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>


#define FFT_BASE_BLOCK 16



FiFT::FiFT(const size_t burst_size, const size_t batch_size)
    : m_burst_size(burst_size)
    , m_batch_size(batch_size)
    , m_num_elts(batch_size * burst_size)
{
    size_t buf_size = burst_size * batch_size * sizeof(COMPLEX_T);
    checkCudaErrors(cudaMalloc(&m_workspace, buf_size));
};

FiFT::~FiFT() {
    checkCudaErrors(cudaFree(m_workspace));
};



/*
  TODO: One possible issue with this step1 approach is the high shared memory per thread, which 
  might reduce occupancy (and I think it might have high compute usage relative to bandwidth?).
  Could try a version which does the loads across batches but then does the matmul 
  as a cooperative warp. Would  the increased need for sync hurt that?
*/

#define FFT_STEP1_THREADBLOCK 32

__global__ static void FFT_step1(const REAL_T* input,
				 COMPLEX_T* output,
				 const size_t burst_size,
				 const size_t batch_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;

    const int num_blocks = burst_size / FFT_BASE_BLOCK;

    // TODO: store as real or complex? i.e. cast ost or shared mem cost?
    __shared__ REAL_T shared_mem[FFT_BASE_BLOCK * FFT_STEP1_THREADBLOCK];

    // Multiply each block by the base case twiddle matrix
    for (int block = 0; block < num_blocks; ++block) {

	// Read the whole block into shared mem in a loop
	// Store it transposed, so adjacent threads aren't getting bank conflicts
	const REAL_T* global_block = &input[burst + block * FFT_BASE_BLOCK * batch_size];
	REAL_T* local_block = &shared_mem[threadIdx.x];
	for (int k = 0; k < FFT_BASE_BLOCK; ++k) {
	    local_block[k * FFT_STEP1_THREADBLOCK] = global_block[k * batch_size];
	}
	
	// Each element in the output block
	for (int k = 0; k < FFT_BASE_BLOCK; ++k) {
	    COMPLEX_T y_k = {0, 0};

	    // Multiply the block by a row from the twiddle matrix
	    for (int n = 0; n < FFT_BASE_BLOCK; ++n) {
		// TODO: do the multiplication by hand, removing the zero terms
		COMPLEX_T term = {(float) local_block[n * FFT_STEP1_THREADBLOCK], 0};
		float exponent = 2 * 3.14159 * n * k / (float) FFT_BASE_BLOCK;
		COMPLEX_T twiddle = {cos(exponent), sin(exponent)};
		y_k = cuCaddf(y_k, cuCmulf(term, twiddle));
	    }

	    output[burst + (block * FFT_BASE_BLOCK + k) * batch_size] = y_k;
	}
    }
    
}
    

__global__ void copy_kernel(const REAL_T* input, COMPLEX_T* output, const size_t n) {
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    output[i] = make_cuComplex (input[i], 0.f);
}

void FiFT::run(const REAL_T* input, COMPLEX_T* output) {
    int num_blocks = (m_batch_size + FFT_STEP1_THREADBLOCK - 1) / FFT_STEP1_THREADBLOCK;
    FFT_step1<<<num_blocks, FFT_STEP1_THREADBLOCK>>>(input,
						     output,
						     m_burst_size,
						     m_batch_size);
};



// ------------------------------ VALIDATION WRAPPER ------------------------------ //

extern "C"
void test(const REAL_T* input,
	  COMPLEX_T* output,
	  const size_t burst_size,
	  const size_t batch_size)
{

    const size_t input_size = burst_size * batch_size * sizeof(REAL_T);
    const size_t output_size = burst_size * batch_size * sizeof(COMPLEX_T);
    REAL_T *d_input;
    COMPLEX_T *d_output;
    checkCudaErrors(cudaMalloc((void**)&d_input, input_size));
    checkCudaErrors(cudaMalloc((void**)&d_output, output_size));
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    
    FiFT fift(burst_size, batch_size);
    fift.run(d_input, d_output);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}