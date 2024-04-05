#include "compute.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

ComputeApp::ComputeApp(){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Get platforms.
	uint32_t numPlatforms = 0;
	err = clGetPlatformIDs(0, nullptr, &numPlatforms);
	if(err != CL_SUCCESS){
		std::cout<<"Could not get platform count!\n";
		exit(EXIT_FAILURE);
	}
	
	if(numPlatforms <= 0){
		std::cout<<"No platforms found!\n";
		exit(EXIT_FAILURE);
	}

	std::vector<cl_platform_id> platforms(numPlatforms);
	err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
	if(err != CL_SUCCESS){
		std::cout<<"Could not get platforms!\n";
		exit(EXIT_FAILURE);
	}

	//Use the first available platform.
	platform = platforms[0];

	//Get devices.
	uint32_t numDevices = 0;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
	if(err != CL_SUCCESS){
		std::cout<<"Could not get device count!\n";
		exit(EXIT_FAILURE);
	}

	if(numDevices <= 0){
		std::cout<<"No supported devices found!\n";
		exit(EXIT_FAILURE);
	}

	std::vector<cl_device_id> devices(numDevices);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
	if(err != CL_SUCCESS){
		std::cout<<"Could not get devices!\n";
		exit(EXIT_FAILURE);
	}

	//Use the first suitable device.
	device = devices[0];

	//Create context.
	context = clCreateContext(0, 1, &device, nullptr, nullptr, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not create context!\n";
		exit(EXIT_FAILURE);
	}

	//Create command queue.
	const cl_queue_properties properties[] = {
		CL_QUEUE_PROPERTIES,
		CL_QUEUE_PROFILING_ENABLE,
		0
	};
	queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not create command queue!\n";
		exit(EXIT_FAILURE);
	}

	//Sources for kernel programs.
	const char* source_sqmProduct = R"(
		__kernel void sqmProduct(
			__global const float* a,
			__global const float* b,
			const unsigned int side,
			__global float* out
		){
			int m = get_global_id(0);
			int n = get_global_id(1);

			if((m<side)&&(n<side)){
				float result = 0.0f;
				for(int k=0;k<side;k++){
					result += a[k + n * side] * b[m + k * side];
				}

				out[m + n * side] = result;
			}
		}
	)";
	k_sqmProduct = createKernel("sqmProduct", source_sqmProduct);
}

ComputeApp::~ComputeApp(){
	//Cleanup.
	clReleaseKernel(k_sqmProduct);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void ComputeApp::sqMatrixProduct(float* a, float* b, const uint32_t side, float* out){
	//Error handle.
	cl_int err = CL_SUCCESS;

	struct timeval start, end;
	gettimeofday(&start, NULL);

	//Constants.
	uint32_t count = side * side;
	size_t bufLen = count * sizeof(float);

	//Allocate buffers and copy data.
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bufLen, a, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not allocate buffer!\n";
		exit(EXIT_FAILURE);
	}

	cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bufLen, b, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not allocate buffer!\n";
		exit(EXIT_FAILURE);
	}

	cl_mem d_Out = clCreateBuffer(context, CL_MEM_READ_WRITE, bufLen, nullptr, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not allocate buffer!\n";
		exit(EXIT_FAILURE);
	}

	//Set kernel arguments.
	err = clSetKernelArg(k_sqmProduct, 0, sizeof(cl_mem), &d_A);
	err |= clSetKernelArg(k_sqmProduct, 1, sizeof(cl_mem), &d_B);
	err |= clSetKernelArg(k_sqmProduct, 2, sizeof(uint32_t), &side);
	err |= clSetKernelArg(k_sqmProduct, 3, sizeof(cl_mem), &d_Out);
	if(err != CL_SUCCESS){
		std::cout<<"Could not set kernel program arguments!\n";
		exit(EXIT_FAILURE);
	}

	//Submit work.
	cl_event kernel_event;
	const size_t global[2] = {side, side};
	err = clEnqueueNDRangeKernel(queue, k_sqmProduct, 2, 0, global, NULL, 0, NULL, &kernel_event);
	if(err != CL_SUCCESS){
		std::cout<<"Could not submit work!\n";
		exit(EXIT_FAILURE);
	}

	//Copy results to host.
	cl_event copy_event;
	err = clEnqueueReadBuffer(queue, d_Out, CL_TRUE, 0, bufLen, out, 0, NULL, &copy_event);
	if(err != CL_SUCCESS){
		std::cout<<"Could not read results!\n";
		exit(EXIT_FAILURE);
	}

	//Free buffers.
	clReleaseMemObject(d_Out);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_A);

	cl_event events[] = {
		kernel_event,
		copy_event
	};
	clFinish(queue);
	clWaitForEvents(2, events);

	//Print execution times.
	gettimeofday(&end, NULL);
	double elapsed = (double)(end.tv_usec - start.tv_usec) / 1000000 +
		(double)(end.tv_sec - start.tv_sec);

	printf("---OpenCL matrix calculation---\nTotal execution time: %f S.\n", elapsed);

	cl_ulong event_start, event_end;

	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, NULL);
	printf("Kernel: %f S.\n", (double)(event_end - event_start)/1000000000);

	clGetEventProfilingInfo(copy_event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, NULL);
	clGetEventProfilingInfo(copy_event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, NULL);
	printf("Copy  : %f S.\n\n", (double)(event_end - event_start)/1000000000);
}

cl_kernel ComputeApp::createKernel(const char* name, const char* source){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create program.
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not create program: "<<name<<"!\n";
		exit(EXIT_FAILURE);
	}

	//Build program.
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS){
		size_t buildLogLen = 0;
		char buildLog[512];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024, buildLog, &buildLogLen);
		std::cout<<buildLog<<"\n";
	}

	//Create kernel.
	cl_kernel kernel = clCreateKernel(program, name, &err);
	if(err != CL_SUCCESS){
		std::cout<<"Could not create kernel!\n";
		exit(EXIT_FAILURE);
	}

	clReleaseProgram(program);

	return kernel;
}
