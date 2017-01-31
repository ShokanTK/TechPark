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

__global__ void Kernel(float *dataX, float* dataY, unsigned size, float stepT, float stepX)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x + 1;
	if (idx < size - 1)
		dataY[idx] = (dataX[idx+1] - 2*dataX[idx] + dataX[idx-1])*stepT /
		(stepX*stepX) + dataX[idx];

}

int main(void)
{

	static const float lengthStick = 10.0;
	static const float Time = 5.0;

	static const int SIZE = 10;
	size_t const memSize = SIZE * sizeof(float);

	static const float stepX = lengthStick / SIZE;
	static const float stepT = 0.02;
	float *dataFirst, *dataSecond, *dataX, *dataY;


	dataFirst = (float*)malloc(memSize);
	dataSecond = (float*)malloc(memSize);

	initializeTemp(dataFirst, SIZE);

	static const int BLOCKSIZE = 512;
	static const int nBlocks = SIZE / BLOCKSIZE + 1;

	HANDLE_ERROR(cudaMalloc((void**)&dataX, memSize));
	HANDLE_ERROR(cudaMalloc((void**)&dataY, memSize));

	HANDLE_ERROR(cudaMemcpy(dataX, dataFirst, memSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dataY, dataSecond, memSize, cudaMemcpyHostToDevice));
	float i;
	int j;
	for (i = 0.0, j = 0; i < Time; i += stepT, ++j)
	{
		std::cout << j << std::endl;
		if (j % 2 == 0)
			Kernel <<< nBlocks, BLOCKSIZE >>> (dataX, dataY, SIZE, stepT, stepX);
		else
			Kernel <<< nBlocks, BLOCKSIZE >>> (dataY, dataX, SIZE, stepT, stepX);
	}
	HANDLE_ERROR(cudaMemcpy(dataSecond, dataY, memSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(dataFirst, dataX, memSize, cudaMemcpyDeviceToHost));
	std::cout << "First array:" << std::endl;
	for (int i = 0; i < SIZE; i++)
		std::cout << dataFirst[i] << " ";
	std::cout << std::endl;
	std::cout << "Second array:" << std::endl;
	for (int i = 0; i < SIZE; i++)
			std::cout << dataSecond[i] << " ";


	cudaFree(dataY);
	cudaFree(dataX);
	free(dataSecond);
	free(dataFirst);
	return 0;
}



