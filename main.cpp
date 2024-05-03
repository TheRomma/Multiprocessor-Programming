
#include <iostream>
#include <string>
#include <vector>
#include <cinttypes>
#include <time.h>

#include "util.hpp"
#include "compute.hpp"
#include "simpleDepthEstimator.hpp"
#include "OMPDepthEstimator.hpp"
#include "CLDepthEstimator.hpp"
#include "CLDepthEstimator2.hpp"

/*--------------------------------------------------
Constructor arguments:
	1: Factor by which to downsample the original image.
	2: Radius of the window patch in the disparity map calculation ((windowSide - 1) / 2 aka 4 means a 9x9 window).
	3: Maximum disparity value for the disparity maps.
	4: Maximum permitted difference in the cross check calculation.
	5: Radius of the window patch in the occlusion fill calculation.
--------------------------------------------------*/

int main(int argc, char** argv){
	//Square matrix multiplication.
	/*
	{
		float* mat_a = (float*)malloc(100*100*sizeof(float));
		float* mat_b = (float*)malloc(100*100*sizeof(float));
		float* mat_c = (float*)malloc(100*100*sizeof(float));

		srand(time(NULL));
		for(uint32_t i=0;i<100*100;i++){
			mat_a[i] = rand()%10;
			mat_b[i] = rand()%10;
			mat_c[i] = rand()%10;
		}

		ComputeApp matrixCalculator;

		sqMatrixProduct(mat_a, mat_b, 100, mat_c);
		matrixCalculator.sqMatrixProduct(mat_a, mat_b, 100, mat_c);

		free(mat_a);
		free(mat_b);
		free(mat_c);
	}
	*/

	//Stereo image depth estimators.
	SimpleDepthEstimator sde(4, 4, 64, 8, 8);
	OMPDepthEstimator mpd(4, 4, 64, 8, 8);
	CLDepthEstimator cld(4, 4, 64, 8, 8);
	CLDepthEstimator2 cld2(4, 4, 64, 8, 8); //In order for the optimizations to work properly, the arguments for CLD2 should not be changed.
	//cld.printInfo();
	//cld2.printInfo();
	//sde.createDepthMap("im0.png", "im1.png", "simple_out.png");
	//mpd.createDepthMap("im0.png", "im1.png", "openmp_out.png");
	cld.createDepthMap("im0.png", "im1.png", "opencl_out.png");
	cld2.createDepthMap("im0.png", "im1.png", "opencl2_out.png");
}	
