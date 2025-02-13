SRC := src/poisson_icing
NVCC_ARCH	?= sm_75
CUPY_CUDA_VERSION ?= cuda11x

LIB	:= -L$(CUDA_HOME)/lib64 -lcudart
NVCCFLAGS	:= -arch=$(NVCC_ARCH) --ptxas-options=-v --use_fast_math

.PHONY: all clean

all:	cuda

cuda:	$(SRC)/poissonicing.ptx

$(SRC)/poissonicing.ptx: $(SRC)/poissonicing.cu Makefile
	nvcc -ptx $< -o $@ $(NVCCFLAGS) $(LIB)

clean:
	rm -f $(SRC)/poissonicing.ptx

python: cuda
	pip install .
	pip install .[$(CUPY_CUDA_VERSION)]
	pip install .[test]

clean-python: clean
	pip uninstall -y poisson_icing
	pip uninstall -y cupy-$(CUPY_CUDA_VERSION)
	pip uninstall -y pytest
	rm -rf dist
	rm -rf build
	rm -rf $(SRC).egg-info
