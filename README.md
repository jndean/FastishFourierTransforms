# Fast-ish Fourier Transforms

GPUs are pretty good at FFTs, and the Nvidia-provided (closed-source) cuFFT library does a smashing job of implementing efficient R2C FFTs on CUDA devices.

**BUT CAN WE DO BETTER?!?**

Almost certainly not. In this repo I will try anyway, targeting a very specific use case I encountered which *may* be handled suboptimally by cuFFT:

- cuFFT expects the data in "burst-major" form (adjacent elements from each burst are adjacent in memory) whereas our data is in "batch-major" form (the same elements from adjacent bursts in the batch are adjacent in memory), thus requiring a pre-processing step that transposes the data using good coalesced memory accesses. This project will instead have adjacent threads in a warp directly accessing adjacent batch elements. 
- cuFFT expects floats (or __halfs) as input, but our incoming data is real-valued uint8s. This either requires another pre-processing step (folded into the transposition kernel mentioned above) and hits the global memory bandwidth 4 (or 2) times harder than necessary, or requires using cufft call-backs to translate the data on read. I found the call-backs to have an unexpectedly high performance impact (I assume due to the overhead of setting up the function call for each read?). In this project we can compile the cast into the FFT!
- cuFFT outputs only the non-redundant complex-valued frequency terms in the R2C operation, saving on some VRAM and write bandwidth. But for our use case we're only interested in the magnitude of the responses, and we're also about to immediately apply a (convolution kernel and then a) bandpass filter to throw away half the results anyway. With this knowledge our project can save on these output costs!  

Also of interest in this repo: how small does your bandpass filter range need to be before it becomes more efficient not to use an FFT at all, and instead calculate a few frequency bands directly in a big old dot product?

