Running the tConvolveSSE benchmark
==================================

The tConvolveSSE benchmark program measures the performance of a convolutional resampling
algorithm with SSE SIMD Tiled SoA

The benchmark distributes work to multiple cores or multiple nodes via Message Passing
Interface (MPI) much like the ASKAP software, and while it is possible to benchmark an
entire cluster the aim of the benchmark is primarily to benchmark a single compute node.

Platform Requirements
---------------------
Building and execution of the benchmark requires:

* A host system with 512MB of RAM per CPU core
* A C++ compiler (e.g. GCC)
* Make
* MPI (e.g. OpenMPI, MPICH)

By default, the Makefile uses GNU C++ compiler and flags.
