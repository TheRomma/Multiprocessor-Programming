#pragma once

#include <cinttypes>

struct OMPDepthEstimator{
	OMPDepthEstimator(
		const uint32_t downsampleFactor,
		const uint32_t windowRadius,
		const unsigned char maxDisparity,
		const unsigned char maxCrossDifference,
		const uint32_t occlusionRadius
	);
	~OMPDepthEstimator(){};

	void createDepthMap(
		const char* left_name,
		const char* right_name,
		const char* out_name
	);

	uint32_t downsampleFactor;
	uint32_t windowRadius;
	unsigned char maxDisparity;
	unsigned char maxCrossDifference;
	uint32_t occlusionRadius;

	private:

	void makeImgGrey(
		const unsigned char* img,
		const uint32_t width,
		const uint32_t height,
		unsigned char* out,
		double* elapsed
	);

	void makeImgRGBA(
		const unsigned char* img,
		const uint32_t width,
		const uint32_t height,
		unsigned char* out,
		double* elapsed
	);

	void downsampleImg(
		const unsigned char* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t factor,
		unsigned char* out,
		double* elapsed
	);

	void filterImg(
		const unsigned char* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t radius,
		unsigned char* out,
		double* elapsed
	);

	void calcDisparity(
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
	);

	void crossCheck(
		unsigned char* left,
		unsigned char* right,
		const uint32_t width,
		const uint32_t height,
		const unsigned char maxDifference,
		double* elapsed
	);

	void occlusionFill(
		const unsigned char* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t radius,
		unsigned char* out,
		double* elapsed
	);
};
