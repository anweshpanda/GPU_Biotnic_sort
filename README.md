# GPU_Biotnic_sort
The array in GPU global memory is divided into the shared memory of each block.
Parallel bitonic sort is first performed block-wise on GPU using shared memory.
Then the sorted blockes are merged using parallel merge-sort to form the final sorted array.
