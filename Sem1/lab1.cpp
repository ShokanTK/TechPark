#include <stdio.h>
#include <stdlib.h>


void initializeTemp(float*, float*);
void show(float*, float*);
void gcpp(float*, float*, float, float, size_t);
void showIdx(float*, int);
// MAIN FUNCTION

int main(int argc, char* argv[])
{
	FILE *f = fopen("outfile.txt", "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}	
	const float lengthStick = 10.0;
	const float Time = 5.0;
	
	size_t arraySize = 10;
	size_t memSIZE = sizeof(float) * arraySize;
	
	const float stepX = lengthStick / arraySize; // 1.0
	const float stepT = 0.02; // 0.02 std
	
	float* dataA = (float*)malloc(memSIZE);
	float* dataB = (float*)malloc(memSIZE);
    
	initializeTemp(dataA, dataA + arraySize);
	initializeTemp(dataB, dataB + arraySize);
  	
	show(dataA, dataA + arraySize);
	show(dataB, dataB + arraySize);

	float i; int j;
	for (i = 0.0, j = 0; i < Time; i += stepT, ++j)
	{
		if (j % 2 == 0)
		{
			gcpp(dataA, dataB, stepT, stepX, arraySize);
		}
		else
		{
			gcpp(dataB, dataA, stepT, stepX, arraySize);
		}
		printf("%d: ", j);
		showIdx(dataA, arraySize);
		showIdx(dataB, arraySize);
	} 
	
	for (unsigned i = 0; i < arraySize; ++i)
	{
		fprintf(f, "%d\t%f\n", i, dataA[i]);
	}
	fclose(f);
	printf("\n");
	free(dataB);
	free(dataA);
	 
 return 0;
}


void initializeTemp(float* data, float* last) 
{
 	float* curr = data;
	for (; curr != last; ++curr)
	{ 
  		*curr = 0;
	}
	last--;
	*last = 5;
}


void show(float* data, float* last)
{
	float* curr = data;
    printf("Array: ");
	for (; curr != last; ++curr)
	{
		printf("%f ", *curr);
	}
	printf("\n");
}

void gcpp(float *dataX, float* dataY, float stepT, float stepX, size_t arraySize)
{
	for (unsigned i = 1; i < arraySize - 1; ++i)
	{
		dataY[i] = (dataX[i + 1] - 2 * dataX[i] + dataX[i - 1]) * stepT / (stepX * stepX) + dataX[i];
	}
}


void showIdx(float* data, int size)
{
	for(int i = 0; i < size; ++i)
		printf("%f ", data[i]);
	printf("\n");
}





