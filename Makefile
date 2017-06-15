all: hook vectorAdd

CUDA_HOME=/usr/local/cuda-7.5
CUPTI_HOME=$(CUDA_HOME)/extras/CUPTI
PROG_HOME=/home/Test863/CudaProf

export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(CUPTI_HOME)/lib64
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(CUDA_HOME)/lib64
export LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(PROG_HOME)/lib

INCLUDES=-I$(CUPTI_HOME)/include
LIBS= -L $(CUDA_HOME)/lib64 -lcuda -lcudart

FLAGS= -v  -O0

vectorAdd: vectorAdd.o 
	nvcc -o $@ vectorAdd.o $(LIBS) -arch sm_35 -lcuda -lcudart ${FLAGS} -g -G 

vectorAdd.o: vectorAdd.cu
	nvcc -c $< -arch sm_35 -lcuda -lcudart ${FLAGS} -g -G 

hook: cudart_wrapper.c cudart_wrapper.h cudaProf.c
	g++ -fPIC -I. -fpermissive -I${CUDA_HOME}/include -c cudart_wrapper.c -o cudart_wrapper.o -I$(CUPTI_HOME)/include -lm -I$(PROG_HOME)/include ${FLAGS}
	g++ -shared -o cudart_wrapper.so cudart_wrapper.o -L${CUDA_HOME}/lib64 -lbfd -Wl,--export-dynamic -lrt -ldl -L $(CUPTI_HOME)/lib64 -lcupti -pthread -rdynamic ${FLAGS}
	g++ -o cudaProf cudaProf.c -lrt -g ${FLAGS} 

run: vectorAdd
	cudaProf vectorAdd

clean:
	rm -f vectorAdd vectorAdd.o cudart_wrapper.so cudart_wrapper.o cudaProf

