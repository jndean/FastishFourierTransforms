# Fast-ish Fourier Transforms

GPUs are pretty good at FFTs, and the Nvidia-provided (closed-source) cuFFT library does a smashing job of implementing efficient 1D R2C (Real2Complex) FFTs on CUDA devices.

**BUT CAN WE DO BETTER?!?**

I definitely cannot. In this repo I try anyway, targeting a very specific use case I am interested in which doesn't play perfectly to cuFFT's tune. My advantages, compared to a team of professional Nvidia engineers, are:

- they don't know what the incoming data looks like - in particular cuFFT expects floats (or __halfs if you forgo callback support) in "burst-major" form. Our data arrives as uint8s arranged in "batch-major" form, requiring an extra preprocessing kernel to perform a massive transposition and a type cast with good coalesced accesses. Plus it hits the global memory bandwidth 4 (or 2) times harder than is necessary. cuFFT allows inserting arbitrary code on each load using callbacks, but I found the callbacks to have an unexpectedly high impact on throughput (I assume due to the overhead of setting up each function call), and there's no way for them to overcome the uncoalesced data arrangement. In this project these considerations can be compiled into the FFT design.

- cuFFT doesn't understand what parts of the output are important . It already throws away the redundant negative frequency terms for R2C outputs, saving on some VRAM and write bandwidth. But for our use case we're about to apply a bandpass filter to drop half the outputs anyway, and additionally we're only interested in the magnitude of the responses. These will reduce our output bandwidth costs!
- We only need to support a couple of values for the FFT burst length, so we can template these away and also safely do some things entirely in shared memory. Honestly this might end up being our biggest advantage...

Also of interest: how small does your bandpass filter range need to be before it becomes more efficient not to use an FFT at all, and instead calculate a few frequency bands directly in a big old dot product?

