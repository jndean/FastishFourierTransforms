#include "FiFT.h"

#include <algorithm>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>


#define PI 3.141592653589793

#define BASE_BLOCK 32
#define BASE_BLOCK_LOG2 5
#define BASE_BLOCK_MASK (BASE_BLOCK - 1)


__device__ __constant__ COMPLEX_T s1_twiddles[BASE_BLOCK][BASE_BLOCK];
__device__ __constant__ COMPLEX_T s2_twiddles[1024];


FiFT::FiFT(const size_t burst_size, const size_t batch_size)
    : m_burst_size(burst_size)
    , m_batch_size(batch_size)
    , m_num_elts(batch_size * burst_size)
{
    // Allocate workspace
    size_t buf_size = burst_size * batch_size * sizeof(COMPLEX_T);
    checkCudaErrors(cudaMalloc(&m_workspace, buf_size));

    // Precompute FFT twiddles and put them in constant memory
    COMPLEX_T h_s1_twiddles[BASE_BLOCK][BASE_BLOCK];
    COMPLEX_T h_s2_twiddles[burst_size/2];
    for (int k = 0; k < burst_size/2; ++k) {
	double exponent = -2.0 * PI * k / (double)burst_size;
	h_s2_twiddles[k] = {(float) cos(exponent), (float) sin(exponent)};
    }
    for (int n = 0; n < BASE_BLOCK; ++n) {
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    double exponent = -2.0 * PI * n * k / (double) BASE_BLOCK;
	    h_s1_twiddles[n][k] = {(float) cos(exponent), (float) sin(exponent)};
	}
    }
    cudaMemcpyToSymbol(s1_twiddles, h_s1_twiddles, sizeof(s1_twiddles));
    cudaMemcpyToSymbol(s2_twiddles, h_s2_twiddles, sizeof(s2_twiddles));
};

FiFT::~FiFT() {
    checkCudaErrors(cudaFree(m_workspace));
};



// ----------------------------------- NORMAL VERSION ----------------------------------- //



#define STEP1_THREADBLOCK 32

__global__ static void step1_kernel(const REAL_T* input,
				    COMPLEX_T* output,
				    const size_t burst_size,
				    const size_t batch_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;

    const int num_blocks = burst_size >> BASE_BLOCK_LOG2;

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
		//COMPLEX_T twiddle = {0.0f, 0.0f};
		COMPLEX_T twiddle = s1_twiddles[n][k];
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

	float exponent = -2.0 * PI * block * half_block_size / (float)burst_size;
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


void FiFT::run(const REAL_T* input, COMPLEX_T* output) {
    COMPLEX_T *step2_input = m_workspace;
    COMPLEX_T *step2_output = output;
    int N = m_burst_size >> BASE_BLOCK_LOG2, log2N = 1;
    while (N >>= 1) log2N++;
    if (log2N & 1) std::swap(step2_input, step2_output);
    
    run_step1(input, step2_input);
    run_step2(step2_input, step2_output);
};



// ----------------------------------- TRANSPOSED VERSION ----------------------------------- //



__global__ static void step1_transpose_kernel(const REAL_T* input,
					      COMPLEX_T* output,
					      const size_t burst_size,
					      const size_t batch_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;

    const int num_blocks = burst_size >> BASE_BLOCK_LOG2;

    // Store as float not int8. Saves on later casts, and means there's one
    // element per shared memory bank
    __shared__ float tile[BASE_BLOCK * BASE_BLOCK];

    for (int block = 0; block < num_blocks; ++block) {

	// Read the whole block into shared memory
	// Store it transposed, so adjacent threads aren't getting bank conflicts
	const REAL_T* global_block = &input[burst + block * batch_size];
	float* local_block = &tile[threadIdx.x];
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    local_block[k * BASE_BLOCK] = global_block[k * num_blocks * batch_size];
	}
	
	// Each element in the output block
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    COMPLEX_T y_k = {0.0f, 0.0f};

	    // Multiply the block by a row from the twiddle matrix
	    #pragma unroll
	    for (int n = 0; n < BASE_BLOCK; ++n) {
		// Seems it's still faster to do complex multiplication with a zero complex term than
		// do the mul separately with fewer terms (does it compile to use double instrucitons maybe?)
		COMPLEX_T term = {(float)local_block[n * BASE_BLOCK], 0};
		
		// Twiddle matrix is symmetric, so can just put [k][n] the way round that means
		// __constant__ memory access is done with broadcasts :D
		COMPLEX_T twiddle = s1_twiddles[k][n];
		y_k = cuCaddf(y_k, cuCmulf(term, twiddle));
	    }
	    
	    //output[burst + (block + k * num_blocks) * batch_size] = y_k;
	    // These writes aren't coalesced, and it's a big deal...
	    output[burst * burst_size + k + block * BASE_BLOCK] = y_k;
	}
    }

}


void FiFT::run_step1_transpose(const REAL_T* input, COMPLEX_T* output)
{
    int num_blocks = (m_batch_size + BASE_BLOCK - 1) >> BASE_BLOCK_LOG2;
    step1_transpose_kernel<<<num_blocks, BASE_BLOCK>>>(input,
						       output,
						       m_burst_size,
						       m_batch_size);
}


__global__ static void step2_transpose_kernel(const COMPLEX_T* input,
					      COMPLEX_T* output,
					      const int burst_size)
{
    const int burst = blockIdx.x;
    const int element = threadIdx.x;

    extern __shared__ COMPLEX_T local[];
    // Load elements accounting for how they're twisted within the burst by step1
    int num_base_blocks = burst_size >> BASE_BLOCK_LOG2;
    int step1_block = element >> BASE_BLOCK_LOG2;
    int step1_block_element = element & BASE_BLOCK_MASK;
    int local_idx = step1_block + step1_block_element * num_base_blocks;
    local[local_idx] = input[burst * burst_size + element];
    local_idx += num_base_blocks >> 1;
    local[local_idx] = input[burst * burst_size + element + burst_size/2];

    int num_blocks = BASE_BLOCK;
    int block_size = num_base_blocks;
    int half_block_size = block_size >> 1;

    while (num_blocks < burst_size) {
	
	int block = element / half_block_size;
	int block_elt = element % half_block_size;
	
	// This is where I might try an async shared mem load? If my GPU supported it...
	int idx = block * block_size + block_elt;
	COMPLEX_T even = local[idx];
	COMPLEX_T odd = local[idx + half_block_size];

	idx = block * half_block_size;
	COMPLEX_T twiddle = s2_twiddles[idx];
	odd = cuCmulf(odd, twiddle);
	idx += block_elt;
	
	__syncthreads();
	
	local[idx] = cuCaddf(even, odd);
	local[idx + burst_size/2] = cuCsubf(even, odd);
	
	num_blocks <<= 1;
	block_size = half_block_size;
	half_block_size >>= 1;	
    }

    __syncthreads();
    
    output[burst * burst_size + element] = local[element];
    output[burst * burst_size + element + burst_size/2] = local[element + burst_size/2];
} 


void FiFT::run_step2_transpose(COMPLEX_T* input, COMPLEX_T* output)
{
    const int num_blocks = m_batch_size;
    const int threads_per_block = m_burst_size / 2;
    const int sharedmem = sizeof(COMPLEX_T) * m_burst_size;

    step2_transpose_kernel<<<num_blocks, threads_per_block, sharedmem>>>
	(input,
	 output,
	 m_burst_size);
}


void FiFT::run_transposed(const REAL_T* input, COMPLEX_T* output) {
    run_step1_transpose(input, m_workspace);
    run_step2_transpose(m_workspace, output);
};


// ----------------------------------- PACKED VERSION ----------------------------------- //


__global__ static void step1_packed_kernel(const COMPLEX_T* input,
					   COMPLEX_T* output,
					   const size_t burst_size,
					   const size_t batch_size)
{
    const int burst =  blockDim.x * blockIdx.x + threadIdx.x;
    if (burst >= batch_size) return;

    const int num_blocks = burst_size >> BASE_BLOCK_LOG2;

    // Store as float not int8. Saves on later casts, and means there's one
    // element per shared memory bank
    __shared__ COMPLEX_T tile[BASE_BLOCK * BASE_BLOCK];

    for (int block = 0; block < num_blocks; ++block) {

	// Read the whole block into shared memory
	// Store it transposed, so adjacent threads aren't getting bank conflicts
	const COMPLEX_T* global_block = &input[burst + block * batch_size];
	// TODO: Now that this is COMPLEX_T, get 2-way bank conflicts on every access...
	// Could split into real components in a column to avoid this
	COMPLEX_T* local_block = &tile[threadIdx.x];
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    local_block[k * BASE_BLOCK] = global_block[k * num_blocks * batch_size];
	}
	
	// Each element in the output block
	for (int k = 0; k < BASE_BLOCK; ++k) {
	    COMPLEX_T y_k = {0.0f, 0.0f};

	    // Multiply the block by a row from the twiddle matrix
	    // #pragma unroll
	    for (int n = 0; n < BASE_BLOCK; ++n) {
		y_k = cuCaddf(y_k, cuCmulf(s1_twiddles[k][n], local_block[n * BASE_BLOCK]));
	    }
	    
	    //output[burst + (block + k * num_blocks) * batch_size] = y_k;
	    // These writes aren't coalesced, and it's a big deal...
	    output[burst * burst_size + k + block * BASE_BLOCK] = y_k;
	}
    }

}


void FiFT::run_step1_packed(const REAL_T* input, COMPLEX_T* output)
{
    int num_blocks = (m_batch_size + BASE_BLOCK - 1) >> BASE_BLOCK_LOG2;
    step1_packed_kernel<<<num_blocks, BASE_BLOCK>>>
	((COMPLEX_T*) input,
	 output,
	 m_burst_size / 2,
	 m_batch_size);
}


__global__ static void step2_packed_kernel(const COMPLEX_T* input,
					   COMPLEX_T* output,
					   const int burst_size)
{
    const int burst = blockIdx.x;
    const int element = threadIdx.x;

    extern __shared__ COMPLEX_T local[];
    // Load elements accounting for how they're twisted within the burst by step1
    int num_base_blocks = burst_size >> BASE_BLOCK_LOG2;
    int step1_block = element >> BASE_BLOCK_LOG2;
    int step1_block_element = element & BASE_BLOCK_MASK;
    int local_idx = step1_block + step1_block_element * num_base_blocks;
    local[local_idx] = input[burst * burst_size + element];
    local_idx += num_base_blocks >> 1;
    local[local_idx] = input[burst * burst_size + element + burst_size/2];

    int num_blocks = BASE_BLOCK;
    int block_size = num_base_blocks;
    int half_block_size = block_size >> 1;

    while (num_blocks < burst_size) {
	
	int block = element / half_block_size;
	int block_elt = element % half_block_size;
	
	// This is where I might try an async shared mem load? If my GPU supported it...
	int idx = block * block_size + block_elt;
	COMPLEX_T even = local[idx];
	COMPLEX_T odd = local[idx + half_block_size];

	idx = block * half_block_size;
	COMPLEX_T twiddle = s2_twiddles[idx];
	odd = cuCmulf(odd, twiddle);
	idx += block_elt;
	
	__syncthreads();
	
	local[idx] = cuCaddf(even, odd);
	local[idx + burst_size/2] = cuCsubf(even, odd);
	
	num_blocks <<= 1;
	block_size = half_block_size;
	half_block_size >>= 1;	
    }

    __syncthreads();

    // W^nk_N = e^(-i.2.PI.n.k/N)
    // A(k) = (1 - j.WK2n) / 2
    // B(k) = (1 + j.WK2n) / 2
    int k = element;
    int Nmink = (burst_size - k) % burst_size;
    COMPLEX_T XNmink_conj = {local[Nmink].x, -local[Nmink].y};
    float exponent = -2 * PI * k / (float) burst_size;
    COMPLEX_T jWk2N = cuCmulf({0, 1}, {cos(exponent), sin(exponent)});
    COMPLEX_T Ak = cuCmulf({0.5, 0}, cuCsubf({1, 0}, jWk2N));
    COMPLEX_T Bk = cuCmulf({0.5, 0}, cuCaddf({1, 0}, jWk2N));
    COMPLEX_T Gk = cuCmulf(local[k], Ak);
    Gk = cuCaddf(Gk, cuCmulf(XNmink_conj, Bk));  
    output[burst * burst_size + element] = Gk;

    k = element + burst_size/2;
    Nmink = (burst_size - k) % burst_size;
    XNmink_conj = {local[Nmink].x, -local[Nmink].y};
    exponent = -2 * PI * k / (float) burst_size;
    jWk2N = cuCmulf({0, 1}, {cos(exponent), sin(exponent)});
    Ak = cuCmulf({0.5, 0}, cuCsubf({1, 0}, jWk2N));
    Bk = cuCmulf({0.5, 0}, cuCaddf({1, 0}, jWk2N));
    Gk = cuCmulf(local[k], Ak);
    Gk = cuCaddf(Gk, cuCmulf(XNmink_conj, Bk));  
    output[burst * burst_size + element + burst_size/2] = Gk;
} 


void FiFT::run_step2_packed(COMPLEX_T* input, COMPLEX_T* output)
{
    const int num_blocks = m_batch_size;
    const int threads_per_block = m_burst_size / 4;
    const int sharedmem = sizeof(COMPLEX_T) * m_burst_size / 2;

    step2_packed_kernel<<<num_blocks, threads_per_block, sharedmem>>>
	(input,
	 output,
	 m_burst_size/2);
}


void FiFT::run_packed(const REAL_T* input, COMPLEX_T* output) {
    run_step1_packed(input, m_workspace);
    run_step2_packed(m_workspace, output);
};



// ------------------------------ ONESHOT VERSION ------------------------------ //



#define WARP 32

__global__ static void oneshot_kernel(const REAL_T* input,
				      COMPLEX_T* output,
				      const int burst_size)
{
    
}


void FiFT::run_oneshot(const REAL_T* input, COMPLEX_T* output) {
    const int thread_block = WARP;
    const int thread_grid = (m_burst_size + WARP - 1) / WARP;
    const int shmem = m_burst_size * (sizeof(COMPLEX_T) + WARP * sizeof(REAL_T));
    oneshot_kernel<<<thread_grid, thread_block, shmem>>>
	(input, output, m_burst_size);
};



// ------------------------------ VALIDATION WRAPPERS ------------------------------ //

extern "C"
const int base_block = BASE_BLOCK;

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
void test_step1_transpose(const REAL_T* input,
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
    fift.run_step1_transpose(d_input, d_output);

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
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    FiFT fift(burst_size, batch_size);
    fift.run_step2_transpose(d_input, d_output);
    
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}




extern "C"
void test_step1_packed(const REAL_T* input,
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
    fift.run_step1_packed(d_input, d_output);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}


extern "C"
void test_step2_packed(COMPLEX_T* input,
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
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    FiFT fift(burst_size, batch_size);
    fift.run_step2_packed(d_input, d_output);
    
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}


