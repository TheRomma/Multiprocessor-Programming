#include "simpleDepthEstimator.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "util.hpp"

/*-------------------------------------------
This is the single threaded implementation 
of the depth estimator for phase 3.
-------------------------------------------*/

//Stereo image depth estimator implemented using OpenMP multithreading.
SimpleDepthEstimator::SimpleDepthEstimator(
	const uint32_t downsampleFactor,
	const uint32_t windowRadius,
	const unsigned char maxDisparity,
	const unsigned char maxCrossDifference,
	const uint32_t occlusionRadius
){
	this->downsampleFactor = downsampleFactor;
	this->windowRadius = windowRadius;
	this->maxDisparity = maxDisparity;
	this->maxCrossDifference = maxCrossDifference;
	this->occlusionRadius = occlusionRadius;
}

//Create a depth map from left and right source images.
void SimpleDepthEstimator::createDepthMap(
	const char* left_name,
	const char* right_name,
	const char* out_name
){
	//Load images.
	unsigned char* img[2];
	uint32_t w, h;

	imgLoad(left_name, &w, &h, &img[0]);
	imgLoad(right_name, &w, &h, &img[1]);

	uint32_t W = w / downsampleFactor;
	uint32_t H = h / downsampleFactor;

	//Allocate memory for images.
	unsigned char* grey[2];
	unsigned char* down[2];
	unsigned char* mean[2];

	grey[0] = (unsigned char*)malloc(w*h*sizeof(unsigned char));
	grey[1] = (unsigned char*)malloc(w*h*sizeof(unsigned char));
	down[0] = (unsigned char*)malloc(W*H*sizeof(unsigned char));
	down[1] = (unsigned char*)malloc(W*H*sizeof(unsigned char));
	mean[0] = (unsigned char*)malloc(W*H*sizeof(unsigned char));
	mean[1] = (unsigned char*)malloc(W*H*sizeof(unsigned char));

	double times[11];

	//Start measuring execution time.
	struct timeval time_start, time_end;
	gettimeofday(&time_start, NULL);

	//Prepare left and right images.
	for(uint32_t i=0;i<2;i++){
		makeImgGrey(img[i], w, h, grey[i], &times[0+i*3]);
		downsampleImg(grey[i], w, h, downsampleFactor, down[i], &times[1+i*3]);
		filterImg(down[i], W, H, windowRadius, mean[i], &times[2+i*3]);
	}

	//Create left and right disparity maps.
	for(uint32_t i=0;i<2;i++){
		calcDisparity(down[i], down[1-i], mean[i], mean[1-i], W, H, windowRadius, maxDisparity, -1+i*2, grey[i], &times[6+i]);
	}

	//Combine images and apply post processing.
	crossCheck(grey[0], grey[1], W, H, maxCrossDifference, &times[8]);
	occlusionFill(grey[0], W, H, occlusionRadius, mean[0], &times[9]);

	//Finish measuring execution time.
	gettimeofday(&time_end, NULL);

	//Write the final image into a file.
	makeImgRGBA(mean[0], W, H, grey[1], &times[10]);
	imgWrite(out_name, W, H, grey[1]);

	free(img[0]);
	free(img[1]);
	free(grey[0]);
	free(grey[1]);
	free(down[0]);
	free(down[1]);
	free(mean[0]);
	free(mean[1]);

	//Print total execution time.
	double elapsed = (double)(time_end.tv_usec - time_start.tv_usec) / 1000000 +
		(double)(time_end.tv_sec - time_start.tv_sec);
	printf("---Simple Depth Estimator---\nTotal execution time: %f S.\n", elapsed);

	printf("Left greyscale      : %f S.\n", times[0]);
	printf("Left downsample     : %f S.\n", times[1]);
	printf("Left filter         : %f S.\n", times[2]);
	printf("Right greyscale     : %f S.\n", times[3]);
	printf("Right downsample    : %f S.\n", times[4]);
	printf("Right filter        : %f S.\n", times[5]);
	printf("Left disparity      : %f S.\n", times[6]);
	printf("Right disparity     : %f S.\n", times[7]);
	printf("Cross check         : %f S.\n", times[8]);
	printf("Occlusion fill      : %f S.\n", times[9]);
	printf("Convert rgba        : %f S.\n\n", times[10]);
}

//Create a greyscale image based on source 8bit rgba image.
void SimpleDepthEstimator::makeImgGrey(
	const unsigned char* img,
	const uint32_t width,
	const uint32_t height,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	uint32_t N = width * height;
	for(uint32_t i=0;i<N;i++){
		out[i] = (unsigned int)(
			img[i*4  ] * 0.2126f +
			img[i*4+1] * 0.7152f +
			img[i*4+2] * 0.0722f
		);
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Make an rgba image based on source greyscale image.
void SimpleDepthEstimator::makeImgRGBA(
	const unsigned char* img,
	const uint32_t width,
	const uint32_t height,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	uint32_t N = width * height;
	for(uint32_t i=0;i<N;i++){
		uint32_t I = i * 4;
		out[I  ] = img[i];
		out[I+1] = img[i];
		out[I+2] = img[i];
		out[I+3] = 255;
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Downsample the image by averaging pixel intensities.
void SimpleDepthEstimator::downsampleImg(
	const unsigned char* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t factor,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	uint32_t w = width / factor;
	uint32_t h = height / factor;
	for(uint32_t i=0;i<h;i++){
		for(uint32_t j=0;j<w;j++){
			uint32_t val = 0;
			uint32_t I = i * factor;
			uint32_t J = j * factor;
			for(uint32_t m=I;m<I+factor;m++){
				for(uint32_t n=J;n<J+factor;n++){
					val += img[n+m*width];
				}
			}
			out[j+i*w] = val / (factor * factor);
		}
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Apply a mean filter to the image.
void SimpleDepthEstimator::filterImg(
	const unsigned char* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t radius,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	for(int32_t i=0;i<(int32_t)height;i++){
		for(int32_t j=0;j<(int32_t)width;j++){
			uint32_t val = 0;
			for(int32_t m=i-(int32_t)radius;m<=i+(int32_t)radius;m++){
				for(int32_t n=j-(int32_t)radius;n<=j+(int32_t)radius;n++){
					if(0<=m&&m<(int32_t)height&&0<=n&&n<(int32_t)width){
						val += img[n+m*width];
					}
				}
			}
			uint32_t d = radius*2+1;
			out[j+i*width] = val / (d*d);
		}
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Create a disparity map from source images.
void SimpleDepthEstimator::calcDisparity(
	const unsigned char* img_0,
	const unsigned char* img_1,
	const unsigned char* mean_0,
	const unsigned char* mean_1,
	const uint32_t width,
	const uint32_t height,
	const uint32_t radius,
	const uint32_t maxDisparity,
	const int32_t direction,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	for(int32_t i=0;i<(int32_t)height;i++){
		for(int32_t j=0;j<(int32_t)width;j++){

			float top_zncc = -1.0f;
			float temp_zncc = -1.0f;
			unsigned char disparity = 0;

			float std_0 = 0.0f;
			float std_1 = 0.0f;
			float numer = 0.0f;
			float denom_0 = 0.0f;
			float denom_1 = 0.0f;

			for(int32_t d=0;d<(int32_t)maxDisparity;d++){
				if((j+direction*d)<0||(int32_t)width<=(j+direction*d)){break;}
				numer = 0.0f;
				denom_0 = 0.0f;
				denom_1 = 0.0f;

				for(int32_t m=i-(int32_t)radius;m<=i+(int32_t)radius;m++){
					for(int32_t n=j-(int32_t)radius;n<=j+(int32_t)radius;n++){
						if(0<=m&&m<(int32_t)height&&0<=(n+direction*d)&&(n+direction*d)<(int32_t)width&&0<=n&&n<(int32_t)width){
							std_0 = img_0[n+m*width] - mean_0[j+i*width];
							std_1 = img_1[n+m*width+direction*d] - mean_1[j+i*width+direction*d];
							numer += std_0 * std_1;
							denom_0 += std_0 * std_0;
							denom_1 += std_1 * std_1;
						}
					}
				}

				temp_zncc = numer / (sqrt(denom_0) * sqrt(denom_1));
				if(temp_zncc > top_zncc){
					top_zncc = temp_zncc;
					disparity = d;
				}
			}
			out[j+i*width] = disparity;
		}
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Compare and combine left and right images. Resulting image will be saved to "left".
void SimpleDepthEstimator::crossCheck(
	unsigned char* left,
	unsigned char* right,
	const uint32_t width,
	const uint32_t height,
	const unsigned char maxDifference,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	uint32_t N = width * height;
	for(uint32_t i=0;i<N;i++){
		if(abs(left[i] - right[i]) > maxDifference){left[i] = 0;}
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}

//Fill blank spaces left by cross check.
void SimpleDepthEstimator::occlusionFill(
	const unsigned char* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t radius,
	unsigned char* out,
	double* elapsed
){
	struct timeval start, end;
	gettimeofday(&start, NULL);

	for(int32_t i=0;i<(int32_t)height;i++){
		for(int32_t j=0;j<(int32_t)width;j++){
			if(img[j+i*width] > 0){
				out[j+i*width] = img[j+i*width];
			}else{
				float numer = 0.0f;
				uint32_t denom = 0;
				for(int32_t m=i-(int32_t)radius;m<=i+(int32_t)radius;m++){
					for(int32_t n=j-(int32_t)radius;n<=j+(int32_t)radius;n++){
						if(0<=m&&m<(int32_t)height&&0<=n&&n<(int32_t)width){
							if(img[n+m*width] > 0){
								numer += img[n+m*width];
								denom++;
							}
						}
					}
				}
				out[j+i*width] = (unsigned char)(numer/denom);
			}
		}
	}

	gettimeofday(&end, NULL);
	*elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);
}
