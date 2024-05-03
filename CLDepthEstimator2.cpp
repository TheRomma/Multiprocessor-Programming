#include "CLDepthEstimator2.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sys/time.h>

#include "util.hpp"

#define LOCAL_SIZE 64
#define LOCAL_SIZE_X 8
#define LOCAL_SIZE_Y 8

/*-------------------------------------------
This is the GPU compute implementation of 
the depth estimator for phase 5. OpenCL is 
used for this implementation.

Workgroups are dispatched to parallelize 
image calculations and two command queues 
are used for concurrent resource preparation.
-------------------------------------------*/

//Stereo image depth estimator implemented with OpenCL.
CLDepthEstimator2::CLDepthEstimator2(
	const uint32_t downsampleFactor,
	const uint32_t windowRadius,
	const unsigned char maxDisparity,
	const unsigned char maxCrossDifference,
	const uint32_t occlusionRadius
){
	//Save arguments.
	this->downsampleFactor = downsampleFactor;
	this->windowRadius = windowRadius;
	this->maxDisparity = maxDisparity;
	this->maxCrossDifference = maxCrossDifference;
	this->occlusionRadius = occlusionRadius;

	//Prepare CL.
	platform = findPlatform();
	device = findDevice(platform);
	context = createContext(&device);
	queue[0] = createQueue(context, device);
	queue[1] = createQueue(context, device);

	//Prepare kernels.
	prepareKernels();
}

//Cleanup.
CLDepthEstimator2::~CLDepthEstimator2(){
	clReleaseKernel(k_occlusion);
	clReleaseKernel(k_cross);
	clReleaseKernel(k_disparity);
	clReleaseKernel(k_filter);
	clReleaseKernel(k_downsample);
	clReleaseKernel(k_rgba);
	clReleaseKernel(k_greyscale);
	clReleaseCommandQueue(queue[1]);
	clReleaseCommandQueue(queue[0]);
	clReleaseContext(context);
}

//Create a depth map from left and right source images.
void CLDepthEstimator2::createDepthMap(
	const char* left_name,
	const char* right_name,
	const char* out_name
){
	//Load images.
	cl_mem img[2];
	uint32_t w, h;
	loadImage(queue[0], left_name, &w, &h, &img[0]);
	loadImage(queue[1], right_name, &w, &h, &img[1]);

	uint32_t W = w / downsampleFactor;
	uint32_t H = h / downsampleFactor;

	//Allocate buffers.
	cl_mem grey[2];
	cl_mem down[2];
	cl_mem mean[2];

	grey[0] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, w*h*sizeof(unsigned char), nullptr);
	grey[1] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, w*h*sizeof(unsigned char), nullptr);
	down[0] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, W*H*sizeof(unsigned char), nullptr);
	down[1] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, W*H*sizeof(unsigned char), nullptr);
	mean[0] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, W*H*sizeof(unsigned char), nullptr);
	mean[1] = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_WRITE, W*H*sizeof(unsigned char), nullptr);

	//Create a list of events for profiling.
	cl_event events[11];

	//Sync queues.
	clFinish(queue[0]);
	clFinish(queue[1]);

	//Start measuring execution time.
	struct timeval time_start, time_end;
	gettimeofday(&time_start, NULL);

	//Prepare left and right images.
	for(uint32_t i=0;i<2;i++){
		makeImgGrey(queue[i], &img[i], w, h, &grey[i], &events[0+i*3]);
		downsampleImg(queue[i], &grey[i], w, h, downsampleFactor, &down[i], &events[1+i*3]);
		filterImg(queue[i], &down[i], W, H, windowRadius, &mean[i], &events[2+i*3]);
	}

	//Sync queues.
	clFinish(queue[0]);
	clFinish(queue[1]);

	//Create disparity maps.
	for(uint32_t i=0;i<2;i++){
		calcDisparity(queue[i], &down[i], &down[1-i], &mean[i], &mean[1-i], W, H, windowRadius, maxDisparity, -1+i*2, &grey[i], &events[6+i]);
	}

	//Sync queues.
	clFinish(queue[0]);
	clFinish(queue[1]);

	//Combine images and do post processing.
	crossCheck(queue[0], &grey[0], &grey[1], W, H, maxCrossDifference, &events[8]);
	occlusionFill(queue[0], &grey[0], W, H, occlusionRadius, &mean[0], &events[9]);

	//Finish measuring execution time.
	clFinish(queue[0]);
	gettimeofday(&time_end, NULL);

	//Make final image into 8bit rgba and write as png.
	makeImgRGBA(queue[0], &mean[0], W, H, &grey[1], &events[10]);
	writeImage(queue[0], out_name, W, H, &grey[1]);

	//Cleanup.
	clReleaseMemObject(img[0]);
	clReleaseMemObject(img[1]);
	clReleaseMemObject(grey[0]);
	clReleaseMemObject(grey[1]);
	clReleaseMemObject(down[0]);
	clReleaseMemObject(down[1]);
	clReleaseMemObject(mean[0]);
	clReleaseMemObject(mean[1]);

	//Print execution times.
	clFinish(queue[0]);
	clWaitForEvents(11, events);

	double elapsed = (double)(time_end.tv_usec - time_start.tv_usec) / 1000000 +
		(double)(time_end.tv_sec - time_start.tv_sec);
	printf("---OpenCL Depth Estimator 2---\nTotal execution time: %f S.\n", elapsed);

	profileEvent("Left greyscale      ", events[0]);
	profileEvent("Left downsample     ", events[1]);
	profileEvent("Left filter         ", events[2]);
	profileEvent("Right greyscale     ", events[3]);
	profileEvent("Right downsample    ", events[4]);
	profileEvent("Right filter        ", events[5]);
	profileEvent("Left disparity      ", events[6]);
	profileEvent("Right disparity     ", events[7]);
	profileEvent("Cross check         ", events[8]);
	profileEvent("Occlusion fill      ", events[9]);
	profileEvent("Convert rgba        ", events[10]);
}

//Print OpenCL information.
void CLDepthEstimator2::printInfo(){
	//Error handle.
	cl_int err = CL_SUCCESS;

	char info[128];
	size_t infoSize = 0;

	printf("\n---OpenCL Platform---\n");

	err |= clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &infoSize);
	err |= clGetPlatformInfo(platform, CL_PLATFORM_NAME, infoSize, info, nullptr);
	printf("Name: %s\n", info);

	err |= clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &infoSize);
	err |= clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, infoSize, info, nullptr);
	printf("Vendor: %s\n", info);

	err |= clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, 0, nullptr, &infoSize);
	err |= clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, infoSize, info, nullptr);
	printf("Profile: %s\n", info);

	err |= clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &infoSize);
	err |= clGetPlatformInfo(platform, CL_PLATFORM_VERSION, infoSize, info, nullptr);
	printf("Version: %s\n", info);

	printf("\n---OpenCL Device---\n");

	err |= clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_NAME, infoSize, info, nullptr);
	printf("Name: %s\n", info);

	cl_device_local_mem_type mem_type;
	cl_uint size = 0;

	err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, infoSize, &mem_type, nullptr);
	if(mem_type == 1){
		printf("Local memory type: LOCAL\n");
	}else{
		printf("Local memory type: GLOBAL\n");
	}

	err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, infoSize, &size, nullptr);
	printf("Local memory size: %u\n", size);

	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, infoSize, &size, nullptr);
	printf("Max compute units: %u\n", size);

	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, infoSize, &size, nullptr);
	printf("Max clock frequency: %u\n", size);

	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, infoSize, &size, nullptr);
	printf("Max constant buffer size: %u\n", size);

	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, infoSize, &size, nullptr);
	printf("Max workgroup size: %u\n", size);

	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, nullptr, &infoSize);
	err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, infoSize, &size, nullptr);
	printf("Max item sizes: %u\n", size);

	printf("\n");

	if(err != CL_SUCCESS){
		printf("Could not get OpenCL info!\n");
		exit(EXIT_FAILURE);
	}
}

void CLDepthEstimator2::profileEvent(
	const char* eventName,
	cl_event event
){
	cl_ulong event_start, event_end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, NULL);
	
	printf("%s: %f S.\n", eventName, (double)(event_end - event_start)/1000000000);
}

//Find a suitable platform.
cl_platform_id CLDepthEstimator2::findPlatform(){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Get platforms.
	uint32_t numPlatforms = 0;
	err = clGetPlatformIDs(0, nullptr, &numPlatforms);
	if(err != CL_SUCCESS){
		printf("Could not get platform count!\n");
		exit(EXIT_FAILURE);
	}
	
	if(numPlatforms <= 0){
		//Platform count 0.
		printf("No supported platforms found!\n");
		exit(EXIT_FAILURE);
	}

	std::vector<cl_platform_id> platforms(numPlatforms);
	err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
	if(err != CL_SUCCESS){
		printf("Could not get supported platforms!\n");
		exit(EXIT_FAILURE);
	}

	//Just use first available platform.
	return platforms[0];
}

//Find a suitable device.
cl_device_id CLDepthEstimator2::findDevice(
	cl_platform_id platform
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Get devices.
	uint32_t numDevices = 0;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
	if(err != CL_SUCCESS){
		printf("Could not get device count!\n");
		exit(EXIT_FAILURE);
	}

	if(numDevices <= 0){
		//Device count 0.
		printf("No supported devices found!\n");
		exit(EXIT_FAILURE);
	}

	std::vector<cl_device_id> devices(numDevices);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
	if(err != CL_SUCCESS){
		printf("Could not get supported devices!\n");
		exit(EXIT_FAILURE);
	}

	//Just use the first available device.
	return devices[0];
}

//Create a CL context.
cl_context CLDepthEstimator2::createContext(
	cl_device_id* device
){
	//Error handle.
	cl_int err = CL_SUCCESS;
	
	//Create context.
	cl_context context = clCreateContext(0, 1, device, nullptr, nullptr, &err);
	if(err != CL_SUCCESS){
		printf("Could not create a context!\n");
		exit(EXIT_FAILURE);
	}

	return context;
}

//Create a command queue.
cl_command_queue CLDepthEstimator2::createQueue(
	cl_context context,
	cl_device_id device
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create queue.
	const cl_queue_properties properties[] = {
		CL_QUEUE_PROPERTIES,
		CL_QUEUE_PROFILING_ENABLE,
		0
	};
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
	if(err != CL_SUCCESS){
		printf("Could not create a command queue!\n");
		exit(EXIT_FAILURE);
	}

	return queue;
}

//Create a kernel program from source.
cl_kernel CLDepthEstimator2::createKernel(
	const char* name,
	const char* source
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create program.
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
	if(err != CL_SUCCESS){
		printf("Could not create program: %s!\n", name);
		exit(EXIT_FAILURE);
	}

	//Build program.
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS){
		size_t logLen = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
		char log[logLen];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLen, log, nullptr);
		printf("%s BUILD ERROR! : %s\n", name, log);
		exit(EXIT_FAILURE);
	}

	//Create kernel.
	cl_kernel kernel = clCreateKernel(program, name, &err);
	if(err != CL_SUCCESS){
		printf("Could not create kernel: %s!\n", name);
		exit(EXIT_FAILURE);
	}

	clReleaseProgram(program);

	return kernel;
}

//Create all the needed kernel programs.
void CLDepthEstimator2::prepareKernels(){
	{
		//Create an 8bit greyscale image based on a source 8bit rgba image.
		const char* source = R"(
			__kernel void greyscale(
				__global const uchar4* img,
				const uint width,
				const uint height,
				__global uchar* out,
				__local float4* temp
			){
				int m = get_global_id(0);
				int lm = get_local_id(0);

				if(m<width*height){
					temp[lm] = convert_float4(img[m]);
					float4 vec = {0.2126f, 0.7152f, 0.0722f, 0.0f};

					out[m] = convert_uchar(dot(temp[lm], vec));

				}
			}

		)";
		k_greyscale = createKernel("greyscale", source);
	}

	{
		//Create an 8bit rgba image based on a source 8bit greyscale image.
		const char* source = R"(
			__kernel void rgba(
				__global const uchar* img,
				const uint width,
				const uint height,
				__global uchar* out
			){
				int m = get_global_id(0);
				
				if(m<width*height){
					unsigned int out_i = m*4;

					out[out_i  ] = img[m];
					out[out_i+1] = img[m];
					out[out_i+2] = img[m];
					out[out_i+3] = 255;
				}
			}
		)";
		k_rgba = createKernel("rgba", source);
	}

	{
		//Downsample a greyscale image.
		const char* source = R"(
			__kernel void downsample(
				__global const uchar4* img,
				const uint width,
				const uint height,
				const uint factor,
				__global uchar* out,
				__local float4* temp
			){
				int m = get_global_id(0);
				int n = get_global_id(1);
				int lm = get_local_id(0);
				int ln = get_local_id(1);

				unsigned int w = width/4;
				unsigned int h = height/4;

				if((m<w)&&(n<h)){
					int M = m * 4;
					int N = n * 4;
					float4 vec = {0.0625f, 0.0625f, 0.0625f, 0.0625f};
					float val = 0.0f;

					for(int i=0;i<4;i++){
						temp[(lm*4+i)+ln*8] = convert_float4(img[m+(N+i)*w]);
					}

					for(int i=0;i<4;i++){
						val += dot(temp[(lm*4+i)+ln*8], vec);
					}

					out[m+n*w] = convert_uchar(val);
				}
			}
		)";
		k_downsample = createKernel("downsample", source);
	}

	{
		//A mean filter with an adjustable radius.
		const char* source = R"(
			__kernel void filter(
				__global const uchar* img,
				const uint width,
				const uint height,
				const uint radius,
				__global uchar* out,
				__local float* temp
			){
				int m = get_global_id(0);
				int n = get_global_id(1);
				int lm = get_local_id(0);
				int ln = get_local_id(1);
				int gm = get_group_id(0);
				int gn = get_group_id(1);

				if((m<width)&&(n<height)){
					float val = 0.0f;

					for(int i=0;i<=2;i++){
						for(int j=0;j<=2;j++){
							int x = lm*2+j-4+gm*8;
							int y = ln*2+i-4+gn*8;
							int lx = lm*2+j;
							int ly = ln*2+i;
							if(0<=x&&x<width&&0<=y&&y<height){
								temp[lx+ly*16] = convert_float(img[x+y*width]);
							}else{
								temp[lx+ly*16] = 0.0f;
							}
						}
					}
					barrier(CLK_LOCAL_MEM_FENCE);

					for(int i=-4;i<=4;i++){
						for(int j=-4;j<=4;j++){
							int lx = lm+4+j;
							int ly = ln+4+i;
							val += temp[lx+ly*16];
						}
					}

					float d = 4*2+1;
					out[m+n*width] = convert_uchar(val / (d*d));
				}
			}
		)";
		k_filter = createKernel("filter", source);
	}

	{
		//Calculate disparity from two greyscale images.
		const char* source = R"(
			__kernel void disparity(
				__global const uchar* img_0,
				__global const uchar* img_1,
				__global const uchar* mean_0,
				__global const uchar* mean_1,
				const uint width,
				const uint height,
				const uint radius,
				const uint maxDisparity,
				const int direction,
				__global uchar* out
			){
				int m = get_global_id(0);
				int n = get_global_id(1);

				if((m<width)&&(n<height)){
					float top_zncc = -1.0f;
					float temp_zncc = -1.0f;
					unsigned char disparity = 0;

					float std_0 = 0.0f;
					float std_1 = 0.0f;
					float numer = 0.0f;
					float denom_0 = 0.0f;
					float denom_1 = 0.0f;

					for(int d=0;d<maxDisparity;d++){
						if((m+direction*d)<0||width<=(m+direction*d)){break;}
						numer = 0.0f;
						denom_0 = 0.0f;
						denom_1 = 0.0f;

						for(int i=n-radius;i<=n+radius;i++){
							for(int j=m-radius;j<=m+radius;j++){
								if(0<=i&&i<height&&0<=(j+direction*d)&&(j+direction*d)<width&&0<=j&&j<width){
									std_0 = img_0[j+i*width] - mean_0[m+n*width];
									std_1 = img_1[j+i*width+direction*d] - mean_1[m+n*width+direction*d];
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

					out[m+n*width] = disparity;
				}
			}
		)";
		k_disparity = createKernel("disparity", source);
	}

	{
		//Combine two disparity maps together.
		const char* source = R"(
			__kernel void crosscheck(
				__global uchar* left,
				__global const uchar* right,
				const uint width,
				const uint height,
				const uint maxDifference
			){
				int m = get_global_id(0);
				
				if(m < width*height){
					if(abs(left[m] - right[m]) > maxDifference){
						left[m] = 0;
					}
				}
			}
		)";
		k_cross = createKernel("crosscheck", source);
	}

	{
		//Fill blank spaces left by cross check.
		const char* source = R"(
			__kernel void occlusion(
				__global const uchar* img,
				const uint width,
				const uint height,
				const uint radius,
				__global uchar* out
			){
				int m = get_global_id(0);
				int n = get_global_id(1);
				
				if((m<width)&&(n<height)){
					if(img[m+n*width] > 0){
						out[m+n*width] = img[m+n*width];
					}else{
						float numer = 0.0f;
						int denom = 0;
						for(int i=n-radius;i<=n+radius;i++){
							for(int j=m-radius;j<=m+radius;j++){
								if(0<=i&&i<height&&0<=j&&j<width){
									if(img[j+i*width] > 0){
										numer += img[j+i*width];
										denom++;
									}
								}
							}
						}
						out[m+n*width] = numer / denom;
					}
				}
			}
		)";
		k_occlusion = createKernel("occlusion", source);
	}
}

//Creates an OpenCL buffer and returns the handle.
cl_mem CLDepthEstimator2::createBuffer(
	cl_mem_flags flags,
	uint32_t size,
	void* copy
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	cl_mem buf = clCreateBuffer(context, flags, size, copy, &err);
	if(err != CL_SUCCESS){
		printf("Could not create a buffer!\n");
		exit(EXIT_FAILURE);
	}

	return buf;
}

//Loads an image from a file and sends it to the GPU via a staging buffer in order to utilize faster local device memory.
void CLDepthEstimator2::loadImage(
	cl_command_queue queue,
	const char* filename,
	uint32_t* width,
	uint32_t* height,
	cl_mem* image
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Load image.
	unsigned char* img;
	uint32_t w, h;

	imgLoad(filename, &w, &h, &img);
	uint32_t len = w * h * 4 * sizeof(unsigned char);

	*width = w;
	*height = h;

	//Create a staging buffer.
	cl_mem d_staging = createBuffer(CL_MEM_COPY_HOST_PTR, len, img);
	free(img);

	//Create a buffer for the image.
	*image = createBuffer(CL_MEM_HOST_NO_ACCESS|CL_MEM_READ_ONLY, len, nullptr);

	//Copy image from the staging buffer.
	err = clEnqueueCopyBuffer(queue, d_staging, *image, 0, 0, len, 0, nullptr, nullptr);
	if(err != CL_SUCCESS){
		printf("Could not copy contents from the staging buffer!\n");
		exit(EXIT_FAILURE);
	}

	clReleaseMemObject(d_staging);
}

//Copies the image into a staging buffer and back onto host memory for writing.
void CLDepthEstimator2::writeImage(
	cl_command_queue queue,
	const char* filename,
	uint32_t width,
	uint32_t height,
	cl_mem* image
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	uint32_t len = width * height * 4 * sizeof(unsigned char);

	//Create a staging buffer.
	cl_mem d_staging = createBuffer(CL_MEM_HOST_READ_ONLY, len, nullptr);

	//Copy image to the staging buffer.
	err = clEnqueueCopyBuffer(queue, *image, d_staging, 0, 0, len, 0, nullptr, nullptr);
	if(err != CL_SUCCESS){
		printf("Could not copy contents to the staging buffer!\n");
		exit(EXIT_FAILURE);
	}

	//Read contents from the staging buffer.
	unsigned char* img = (unsigned char*)malloc(len);

	err = clEnqueueReadBuffer(queue, d_staging, CL_TRUE, 0, len, img, 0, NULL, NULL);
	if(err != CL_SUCCESS){
		printf("Could not read results!\n");
		exit(EXIT_FAILURE);
	}

	//Write image to a file.
	imgWrite(filename, width, height, img);

	clReleaseMemObject(d_staging);
	free(img);
}

//Executes a kernel program that creates a new 8bit greyscale image from a source 8bit/channel rgba image.
void CLDepthEstimator2::makeImgGrey(
	cl_command_queue queue,
	cl_mem* img,
	const uint32_t width,
	const uint32_t height,
	cl_mem* out,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_greyscale, 0, sizeof(cl_mem), img);
	err |= clSetKernelArg(k_greyscale, 1, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_greyscale, 2, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_greyscale, 3, sizeof(cl_mem), out);
	err |= clSetKernelArg(k_greyscale, 4, LOCAL_SIZE*4*sizeof(float), NULL);
	if(err != CL_SUCCESS){
		printf("Could not set greyscale kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[1] = {LOCAL_SIZE};
	const size_t global[1] = {(size_t)ceil((width*height)/local[0])*local[0]};
	err = clEnqueueNDRangeKernel(queue, k_greyscale, 1, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit greyscale work!\n");
		exit(EXIT_FAILURE);
	}
}

//Reverse of the greyscale operation. Creates an 8bit rgba image from a source 8bit greyscale image.
void CLDepthEstimator2::makeImgRGBA(
	cl_command_queue queue,
	cl_mem* img,
	const uint32_t width,
	const uint32_t height,
	cl_mem* out,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_rgba, 0, sizeof(cl_mem), img);
	err |= clSetKernelArg(k_rgba, 1, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_rgba, 2, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_rgba, 3, sizeof(cl_mem), out);
	if(err != CL_SUCCESS){
		printf("Could not set rgba kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[1] = {LOCAL_SIZE};
	const size_t global[1] = {(size_t)ceil((width*height)/local[0])*local[0]};
	err = clEnqueueNDRangeKernel(queue, k_rgba, 1, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit rgba work!\n");
		exit(EXIT_FAILURE);
	}
}

//Downsamples an image by a given factor. Resulting pixels are the means of corresponding image patches with size factor*factor.
void CLDepthEstimator2::downsampleImg(
	cl_command_queue queue,
	cl_mem* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t factor,
	cl_mem* out,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_downsample, 0, sizeof(cl_mem), img);
	err |= clSetKernelArg(k_downsample, 1, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_downsample, 2, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_downsample, 3, sizeof(uint32_t), &factor);
	err |= clSetKernelArg(k_downsample, 4, sizeof(cl_mem), out);
	err |= clSetKernelArg(k_downsample, 5, LOCAL_SIZE*4*4*sizeof(float), NULL);
	if(err != CL_SUCCESS){
		printf("Could not set downsample kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
	const size_t global[2] = {
		(size_t)(ceil((width/factor)/local[0])*local[0]),
		(size_t)(ceil((height/factor)/local[1])*local[1])
	};
	err = clEnqueueNDRangeKernel(queue, k_downsample, 2, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit downsample work!\n");
		exit(EXIT_FAILURE);
	}
}

//Applies a mean filter to a greyscale image with a given radius. Out of bound pixels in the window are considered as 0.
void CLDepthEstimator2::filterImg(
	cl_command_queue queue,
	cl_mem* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t radius,
	cl_mem* out,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_filter, 0, sizeof(cl_mem), img);
	err |= clSetKernelArg(k_filter, 1, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_filter, 2, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_filter, 3, sizeof(uint32_t), &radius);
	err |= clSetKernelArg(k_filter, 4, sizeof(cl_mem), out);
	err |= clSetKernelArg(k_filter, 5, 16*16*sizeof(float), NULL);
	if(err != CL_SUCCESS){
		printf("Could not set filter kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
	const size_t global[2] = {
		(size_t)(ceil(width/local[0])*local[0]),
		(size_t)(ceil(height/local[1])*local[1])
	};
	err = clEnqueueNDRangeKernel(queue, k_filter, 2, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit filter work!\n");
		exit(EXIT_FAILURE);
	}
}

//Creates a disparity map from source greyscale images and their mean filtered images.
void CLDepthEstimator2::calcDisparity(
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
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_disparity, 0, sizeof(cl_mem), img_0);
	err |= clSetKernelArg(k_disparity, 1, sizeof(cl_mem), img_1);
	err |= clSetKernelArg(k_disparity, 2, sizeof(cl_mem), mean_0);
	err |= clSetKernelArg(k_disparity, 3, sizeof(cl_mem), mean_1);
	err |= clSetKernelArg(k_disparity, 4, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_disparity, 5, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_disparity, 6, sizeof(uint32_t), &radius);
	err |= clSetKernelArg(k_disparity, 7, sizeof(uint32_t), &maxDisparity);
	err |= clSetKernelArg(k_disparity, 8, sizeof(int32_t), &direction);
	err |= clSetKernelArg(k_disparity, 9, sizeof(cl_mem), out);
	if(err != CL_SUCCESS){
		printf("Could not set disparity kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
	const size_t global[2] = {
		(size_t)(ceil(width/local[0])*local[0]),
		(size_t)(ceil(height/local[1])*local[1])
	};
	err = clEnqueueNDRangeKernel(queue, k_disparity, 2, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit disparity work!\n");
		exit(EXIT_FAILURE);
	}
}

//Combines two disparity maps with a given difference threshold. Pixels deemed too dissimilar are assigned as 0.
void CLDepthEstimator2::crossCheck(
	cl_command_queue queue,
	cl_mem* left,
	cl_mem* right,
	const uint32_t width,
	const uint32_t height,
	const uint32_t maxDifference,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_cross, 0, sizeof(cl_mem), left);
	err |= clSetKernelArg(k_cross, 1, sizeof(cl_mem), right);
	err |= clSetKernelArg(k_cross, 2, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_cross, 3, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_cross, 4, sizeof(uint32_t), &maxDifference);
	if(err != CL_SUCCESS){
		printf("Could not set cross kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[1] = {LOCAL_SIZE};
	const size_t global[1] = {(size_t)ceil((width*height)/local[0])*local[0]};
	err = clEnqueueNDRangeKernel(queue, k_cross, 1, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit cross work!\n");
		exit(EXIT_FAILURE);
	}
}

//Fixes the blank spaces left by crossCheck by assiging them the mean of neighbouring pixels from a given radius.
void CLDepthEstimator2::occlusionFill(
	cl_command_queue queue,
	cl_mem* img,
	const uint32_t width,
	const uint32_t height,
	const uint32_t radius,
	cl_mem* out,
	cl_event* event
){
	//Error handle.
	cl_int err = CL_SUCCESS;

	//Create the greyscale image.
	err = clSetKernelArg(k_occlusion, 0, sizeof(cl_mem), img);
	err |= clSetKernelArg(k_occlusion, 1, sizeof(uint32_t), &width);
	err |= clSetKernelArg(k_occlusion, 2, sizeof(uint32_t), &height);
	err |= clSetKernelArg(k_occlusion, 3, sizeof(uint32_t), &radius);
	err |= clSetKernelArg(k_occlusion, 4, sizeof(cl_mem), out);
	if(err != CL_SUCCESS){
		printf("Could not set occlusion kernel arguments!\n");
		exit(EXIT_FAILURE);
	}
	const size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
	const size_t global[2] = {
		(size_t)(ceil(width/local[0])*local[0]),
		(size_t)(ceil(height/local[1])*local[1])
	};
	err = clEnqueueNDRangeKernel(queue, k_occlusion, 2, 0, global, local, 0, NULL, event);
	if(err != CL_SUCCESS){
		printf("Could not submit occlusion work!\n");
		exit(EXIT_FAILURE);
	}
}
