#include "FiFT.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <helper_cuda.h>



FiFT::FiFT(const size_t burst_size, const size_t batch_size)
    : m_burst_size(burst_size)
    , m_batch_size(batch_size)
    , m_num_elts(batch_size * burst_size)
{
    size_t buf_size = burst_size * batch_size * sizeof(COMPLEX_T);
    checkCudaErrors(cudaMalloc(&m_workspace,buf_size));
};

FiFT::~FiFT() {
    checkCudaErrors(cudaFree(m_workspace));
};


__global__ void copy_kernel(REAL_T* input, COMPLEX_T* output, size_t n) {
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    output[i] = make_cuComplex (input[i], 0.f);
}

void FiFT::run(REAL_T* input, COMPLEX_T* output) {
    int block_size = 32;
    int num_blocks = (m_num_elts + block_size - 1) / block_size;
    copy_kernel<<<num_blocks, block_size>>>(input, output, m_num_elts);
};

