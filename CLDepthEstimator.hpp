#pragma once

#define CL_TARGET_OPENCL_VERSION 220

#include <cinttypes>
#include <CL/cl.h>

struct CLDepthEstimator{
	CLDepthEstimator(
		const uint32_t downsampleFactor,
		const uint32_t windowRadius,
		const unsigned char maxDisparity,
		const unsigned char maxCrossDifference,
		const uint32_t occlusionRadius
	);
	~CLDepthEstimator();

	void createDepthMap(
		const char* left_name,
		const char* right_name,
		const char* out_name
	);

	void printInfo();

	uint32_t downsampleFactor;
	uint32_t windowRadius;
	unsigned char maxDisparity;
	unsigned char maxCrossDifference;
	uint32_t occlusionRadius;

	private:
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue[2];

	void profileEvent(
		const char* eventName,
		cl_event event
	);

	cl_platform_id findPlatform();
	
	cl_device_id findDevice(
		cl_platform_id platform
	);

	cl_context createContext(
		cl_device_id* device
	);

	cl_command_queue createQueue(
		cl_context context,
		cl_device_id device
	);

	cl_kernel createKernel(
		const char* name,
		const char* source
	);

	cl_kernel k_greyscale;
	cl_kernel k_downsample;
	cl_kernel k_filter;
	cl_kernel k_disparity;
	cl_kernel k_cross;
	cl_kernel k_occlusion;
	cl_kernel k_rgba;

	void prepareKernels();

	cl_mem createBuffer(
		cl_mem_flags flags,
		uint32_t size,
		void* copy
	);

	void loadImage(
		cl_command_queue queue,
		const char* filename,
		uint32_t* width,
		uint32_t* height,
		cl_mem* image
	);

	void writeImage(
		cl_command_queue queue,
		const char* filename,
		uint32_t width,
		uint32_t height,
		cl_mem* image
	);

	void makeImgGrey(
		cl_command_queue queue,
		cl_mem* img,
		const uint32_t width,
		const uint32_t height,
		cl_mem* out,
		cl_event* event
	);

	void makeImgRGBA(
		cl_command_queue queue,
		cl_mem* img,
		const uint32_t width,
		const uint32_t height,
		cl_mem* out,
		cl_event* event
	);

	void downsampleImg(
		cl_command_queue queue,
		cl_mem* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t factor,
		cl_mem* out,
		cl_event* event
	);

	void filterImg(
		cl_command_queue queue,
		cl_mem* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t radius,
		cl_mem* out,
		cl_event* event
	);

	void calcDisparity(
		cl_command_queue queue,
		cl_mem* img_0,
		cl_mem* img_1,
		cl_mem* mean_0,
		cl_mem* mean_1,
		const uint32_t width,
		const uint32_t height,
		const uint32_t radius,
		const uint32_t maxDisparity,
		const int32_t direction,
		cl_mem* out,
		cl_event* event
	);

	void crossCheck(
		cl_command_queue queue,
		cl_mem* left,
		cl_mem* right,
		const uint32_t width,
		const uint32_t height,
		const uint32_t maxDifference,
		cl_event* event
	);

	void occlusionFill(
		cl_command_queue queue,
		cl_mem* img,
		const uint32_t width,
		const uint32_t height,
		const uint32_t radius,
		cl_mem* out,
		cl_event* event
	);
};
