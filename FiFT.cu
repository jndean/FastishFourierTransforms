#include "FiFT.h"

#include <algorithm>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>


#define BASE_BLOCK 4



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

#define STEP1_THREADBLOCK 32

__global__ static void step1_kernel(const REAL_T* input,
				    COMPLEX_T* output,
				    const size_t burst_size,
				    const size_t batch_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;

    const int num_blocks = burst_size / BASE_BLOCK;

    // TODO: store as real or complex? i.e. cast ost or shared mem cost?
    __shared__ REAL_T shared_mem[BASE_BLOCK * STEP1_THREADBLOCK];

    // Multiply each block by the base case twiddle matrix
    for (int block = 0; block < num_blocks; ++block) {

	// Read the whole block into shared mem in a loop
	// Store it transposed, so adjacent threads aren't getting bank conflicts
	const REAL_T* global_block = &input[burst + block * batch_size];
	REAL_T* local_block = &shared_mem[threadIdx.x];
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    local_block[k * STEP1_THREADBLOCK] = global_block[k * num_blocks * batch_size];
	}
	
	// Each element in the output block
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    COMPLEX_T y_k = {0.0f, 0.0f};

	    // Multiply the block by a row from the twiddle matrix
	    for (int n = 0; n < BASE_BLOCK; ++n) {
		// TODO: do the multiplication by hand, removing the zero terms
		COMPLEX_T term = {(float) local_block[n * STEP1_THREADBLOCK], 0};
		float exponent = -2.0 * 3.141592653589793 * n * k / (float) BASE_BLOCK;
		COMPLEX_T twiddle = {cos(exponent), sin(exponent)};
		y_k = cuCaddf(y_k, cuCmulf(term, twiddle));
	    }
	    
	    output[burst + (block + k * num_blocks) * batch_size] = y_k;
	}
    } 
}


void FiFT::run_step1(const REAL_T* input, COMPLEX_T* output)
{
    int num_blocks = (m_batch_size + STEP1_THREADBLOCK - 1) / STEP1_THREADBLOCK;
    step1_kernel<<<num_blocks, STEP1_THREADBLOCK>>>(input,
						    output,
						    m_burst_size,
						    m_batch_size);
}


#define STEP2_THREADBLOCK 32

__global__ static void step2_kernel(const COMPLEX_T* input,
				    COMPLEX_T* output,
				    const int block_size,
				    const int num_blocks,
				    const int batch_size,
				    const int burst_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;
    
    const int half_block_size = block_size >> 1;
    
    for (int block = 0; block < num_blocks; ++block) {

	float exponent = -2.0 * 3.141592653589793 * block * half_block_size / (float)burst_size;
	COMPLEX_T twiddle = {cos(exponent), sin(exponent)};
	
	for (int i = 0; i < half_block_size; ++i) {
	    COMPLEX_T odd = input[(block * block_size + half_block_size + i) * batch_size + burst];
	    COMPLEX_T even = input[(block * block_size + i) * batch_size + burst];
	    odd = cuCmulf(odd, twiddle);
	    output[(block * half_block_size + i) * batch_size + burst] = cuCaddf(even, odd);
	    output[((num_blocks + block) * half_block_size + i) * batch_size + burst] = cuCsubf(even, odd);
	}
    }
} 


void FiFT::run_step2(COMPLEX_T* input, COMPLEX_T* output)
{
    int num_FFT_blocks = BASE_BLOCK;
    int FFT_block_size = m_burst_size / num_FFT_blocks;

    COMPLEX_T *read = input, *write = output;

    while (num_FFT_blocks < m_burst_size) {
	int num_thread_blocks = (m_batch_size + STEP2_THREADBLOCK - 1) / STEP2_THREADBLOCK;
	step2_kernel<<<num_thread_blocks, STEP2_THREADBLOCK>>>(read,
							       write,
							       FFT_block_size,
							       num_FFT_blocks,
							       m_batch_size,
							       m_burst_size);
	std::swap(read, write);
	num_FFT_blocks *= 2;
	FFT_block_size /= 2;
    }
}



__global__ static void step2_transposed_kernel(const COMPLEX_T* input,
					       COMPLEX_T* output,
					       const int batch_size,
					       const int burst_size)
{
    const int burst = blockIdx.x;
    const int element = threadIdx.x;

    extern __shared__ COMPLEX_T local[];
    local[element] = input[burst * burst_size + element];
    local[element * 2] = input[burst * burst_size + element * 2];
    
    int num_blocks = BASE_BLOCK;
    int block_size = burst_size / num_blocks;
    int half_block_size = block_size >> 1;
    
    while (num_blocks < burst_size) {
	int block = element / half_block_size;
	int block_elt = element % half_block_size;
	
	// would like TODO an async sharedmem read here, but my GPU is too old
	COMPLEX_T even = local[block * block_size + block_elt];
	COMPLEX_T odd = local[block * block_size + half_block_size + block_elt];
	
	float exponent = -2.0 * 3.141592653589793 * block * half_block_size / (float)burst_size;
	COMPLEX_T twiddle = {cos(exponent), sin(exponent)};
	odd = cuCmulf(odd, twiddle);

	__syncthreads();
	
	local[block * half_block_size + block_elt] = cuCaddf(even, odd);
	local[(block + num_blocks) * half_block_size + block_elt] = cuCsubf(even, odd);
	
	num_blocks <<= 1;
	block_size = half_block_size;
	half_block_size >>= 1;	
    }

    __syncthreads();
    output[burst * burst_size + element] = local[element];
    output[burst * burst_size + element * 2] = local[element * 2];
} 


void FiFT::run_step2_transposed(COMPLEX_T* input, COMPLEX_T* output)
{
    const int num_blocks = m_batch_size;
    const int threads_per_block = m_burst_size / 2;
    const int sharedmem = sizeof(COMPLEX_T) * m_burst_size;
    
    step2_transposed_kernel<<<num_blocks, threads_per_block, sharedmem>>>
	(input,
	 output,
	 m_batch_size,
	 m_burst_size);
}



void FiFT::run(const REAL_T* input, COMPLEX_T* output) {
    COMPLEX_T *step2_input = m_workspace;
    COMPLEX_T *step2_output = output;
    int N = m_burst_size / BASE_BLOCK, log2N = 1;
    while (N >>= 1) log2N++;
    if (log2N & 1) std::swap(step2_input, step2_output);
    
    run_step1(input, step2_input);
    run_step2(step2_input, step2_output);
};



// ------------------------------ VALIDATION WRAPPER ------------------------------ //

extern "C"
void test_step1(const REAL_T* input,
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
    fift.run_step1(d_input, d_output);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}


extern "C"
void test_run(const REAL_T* input,
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


extern "C"
void test_step2_transpose(COMPLEX_T* input,
			  COMPLEX_T* output,
			  const size_t burst_size,
			  const size_t batch_size)
{

    const size_t input_size = burst_size * batch_size * sizeof(COMPLEX_T);
    const size_t output_size = burst_size * batch_size * sizeof(COMPLEX_T);
    COMPLEX_T *d_input;
    COMPLEX_T *d_output;
    checkCudaErrors(cudaMalloc((void**)&d_input, input_size));
    checkCudaErrors(cudaMalloc((void**)&d_output, output_size));
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    
    FiFT fift(burst_size, batch_size);
    fift.run_step2_transposed(d_input, d_output);
    
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}