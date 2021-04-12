#pragma once

#include <memory>

//#include <cuda_fp16.h>
#include <cuComplex.h>


#define REAL_T uint8_t
#define COMPLEX_T cuFloatComplex



class FiFT {
 public:
    FiFT(const size_t burst_size, const size_t batch_size);
    ~FiFT();

    void run(const REAL_T* input, COMPLEX_T* out);
    void run_transposed(const REAL_T* input, COMPLEX_T* output);
    void run_packed(const REAL_T* input, COMPLEX_T* output);
    void run_oneshot(const REAL_T* input, COMPLEX_T* output);
    
    //private:
    void run_step1(const REAL_T* input, COMPLEX_T* out);
    void run_step2(COMPLEX_T* input, COMPLEX_T* output);
    
    void run_step2_transpose(COMPLEX_T* input, COMPLEX_T* output);
    void run_step1_transpose(const REAL_T* input, COMPLEX_T* output);
    
    void run_step2_packed(COMPLEX_T* input, COMPLEX_T* output);
    void run_step1_packed(const REAL_T* input, COMPLEX_T* output);
	
    const size_t m_burst_size;
    const size_t m_batch_size;
    const size_t m_num_elts;
    COMPLEX_T* m_workspace;
};

