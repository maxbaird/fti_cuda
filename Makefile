PROJECT						= fti_cuda.out
NVCC							= nvcc
NVCC_FLAGS				= -c -I/usr/include/openmpi-x86_64 -I${FTI_HOME}/include -Xcompiler -Wall,-Wextra,-Werror
MPICC							= mpicc
MPICC_LD_FLAGS		= -L${CUDA_HOME}/lib64/ -lcudart -L${FTI_HOME}/lib -lfti -lm -lstdc++

.PHONY : $(PROJECT) clean

all : $(PROJECT)

$(PROJECT) : fti_cuda.o
	$(MPICC) $(MPICC_LD_FLAGS) -o $@ $<

fti_cuda.o : fti_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf *.o *.out $(PROJECT)
