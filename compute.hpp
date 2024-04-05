#pragma once

#define CL_TARGET_OPENCL_VERSION 220

#include <cinttypes>
#include <CL/cl.h>

struct ComputeApp{
	ComputeApp();
	~ComputeApp();

	void sqMatrixProduct(
		float* a,
		float* b,
		const uint32_t side,
		float* out
	);

	cl_kernel createKernel(
		const char* name,
		const char* source
	);

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;

	cl_kernel k_sqmProduct;
};
