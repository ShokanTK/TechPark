#include <iostream>
#include <numeric>
#include <stdlib.h>



void initializeTemp(float *data, unsigned size)
{
	for (unsigned i = 0; i < size-1; ++i){
		data[i] = 0;
	}
	data[size-1] = 5;
}

void initializeCoeff(float* data, unsigned size, float coeff1, float coeff2)
{
	for (unsigned i = 1; i < size; ++i)
		for (unsigned j = 0; j < size; ++j)
			data[j + i * size] = 0;
	data[0] = 1;
	data[(size - 1) * size + size - 1] = 1;
	for (unsigned i = 1; i < size; ++i)
	{	for (unsigned j = 0; j < size; ++j)
		{	
			data[i + i * size] = coeff2;
			data[i - 1 + i * size] = coeff1;
			data[i + 1 + i * size] = ceoff1;
		}
	}
}

__global__ void KernelCompute(float* dataP, float* dataN, unsigned SIZE, float* dataTemp, float* coeff)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int sum = 0;

	for (unsigned i = 1; i < SIZE; ++i)
		sum += coeff[idx * SIZE + i] * dataTemp[idx];

	dataN[idx] = dataP[idx] + (dataTemp[idx] - sum) /coeff[idx * SIZE + idx];
}

__global__ void KernelEq(float* X, float* Y)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	Y[idx] = X[idx];
}


bool Error(float* dataX, float* dataY, unsigned size, float eps)
{
	unsigned count = 0;
	for(unsigned i = 0; i < size; ++i)
	{
		if (abs(dataX[i] - dataY[i]) > eps)
			count += 1;
	}
	return count == 0 ? true : false;
}

void show(float* data, unsigned size)
{
	for (unsigned i; i < size; ++i)
		std::cout << data[i] << " ";
}

int main(void)
{
	static const float lengthStick = 10.0;
	static const float Time = 5.0;

	static int SIZE = 10;
	size_t memSize = SIZE * sizeof(float);

	static const float stepX = lengthStick / SIZE;
	static const float stepT = 0.02;
	
	float *dataFirst, *dataSecond, *dataThird, *dataX, *dataY, *dataT, *Coeff, *coeffMatrix;

	float coeffA_1_3 = (stepX * stepX) / stepT;
	float coeffA_2 = 1 + 2 * stepT / (stepX * stepX);

	dataFirst = (float*)malloc(memSize);
	dataSecond = (float*)malloc(memSize);
	dataThird = (float*)malloc(memSize);
	coeffMatrix = (float*)malloc(memSize * memSize);

	int BLOCKSIZE = 512;
	int nBlocks = SIZE * SIZE / BLOCKSIZE + 1;

	int tBLOCKSIZE = 512;
	int tnBlocks = SIZE / BLOCKSIZE + 1;

	initialCoeff(coeffMatrix, SIZE, coeffA_1_3, coeff_2);
	initializeTemp(dataThird, SIZE);
	for(unsigned i = 0; i < SIZE; ++i)
		dataFirst[i] = 0;

	cudaMalloc((void**)&dataX, memSize);
	cudaMalloc((void**)&dataY, memSize);
	cudaMalloc((void**)&dataT, memSize);
	cudaMalloc((void**)&Coeff, memSize * memSize);

	cudaMemcpy(dataX, dataFirst, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dataY, dataSecond, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dataT, dataThird, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(Coeff, coeffMatrix, memSize * memSize, cudaMemcpyHostToDevice);

	Kernel <<< nBlocks, BLOCKSIZE >>> (dataX, dataY, SIZE, dataT, Coeff);
	float i = 0.0; int j = 0;
	for (; i < Time; i += stepT)
	{
		while(Error(dataFirst, dataSecond, SIZE, eps))
		{
			if (j % 2 == 0)
				KernelCompute <<< nBlocks, BLOCKSIZE >>> (dataY, dataX, SIZE, dataT, Coeff);
			else
				KernelCompute <<< nBlocks, BLOCKSIZE >>> (dataX, dataY, SIZE, dataT, Coeff);
			cudaMemcpy(dataFirst, dataX, memSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(dataSecond, dataY, memSize, cudaMemcpyDeviceToDevice);
		}
		KernelEq <<< tnBlocks, tBLOCKSIZE >>> (dataX, dataT);
	}


	cudaMemcpy(dataFirst, dataX, memSize, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dataSecond, dataT, memSize, cudaMemcpyDeviceToDevice);

	cudaFree(dataT);
	cudaFree(dataY);
	cudaFree(dataX);
	cudaFree(Coeff);
	
	free(coeffMatrix);
	free(dataThird);
	free(dataSecond);
	free(dataFirst);

	return 0;
}



