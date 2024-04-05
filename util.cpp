#include "util.hpp"

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <sys/time.h>

#include "lodepng.h"

//Matrix product for two square matrices.
void sqMatrixProduct(
	const float* a,
	const float* b,
	const uint32_t side,
	float* out
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	for(uint32_t i=0;i<side;i++){
		for(uint32_t j=0;j<side;j++){
			out[j + i * side] = 0.0;
			for(uint32_t k=0;k<side;k++){
				out[j + i * side] += a[k + i * side] * b[j + k * side];
			}
		}
	}

	gettimeofday(&end, NULL);
	double elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);

	printf("---Host matrix calculation---\nTotal execution time: %f S.\n\n", elapsed);
}

//Print the contents of a square matrix.
void printSqMatrix(
	const float* a,
	const uint32_t side
){
	for(uint32_t i=0;i<side;i++){
		std::cout<<"[ ";
		for(uint32_t j=0;j<side;j++){
			std::cout<<a[j + i * side]<<" ";
		}
		std::cout<<"]\n";
	}
	std::cout<<std::endl;
}

//Decodes an png image using lodepng.
void imgLoad(
	const char* filename,
	uint32_t* width,
	uint32_t* height,
	unsigned char** image
){
	unsigned error = lodepng_decode32_file(image, width, height, filename);
	if(error){
		std::cout<<lodepng_error_text(error)<<"\n";
		exit(EXIT_FAILURE);
	}
}

//Write image to disk as png.
void imgWrite(
	const char* filename,
	const uint32_t width,
	const uint32_t height,
	const unsigned char* image
){
	unsigned error = lodepng_encode32_file(filename, image, width, height);
	if(error){
		std::cout<<lodepng_error_text(error)<<"\n";
		exit(EXIT_FAILURE);
	}
}
