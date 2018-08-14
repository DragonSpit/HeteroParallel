#include <stdlib.h>
#include <string.h>

#include "clRNG.h"
#include "mrg31k3p.h"

#include "TimerCycleAccurateArray.h"

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

int openClAttributes(cl_device_id * devices, cl_uint& num_devices, cl_platform_id * platforms, cl_uint max_num_platforms, cl_uint& num_platforms)
{
	cl_int err;
	//cl_uint num_platforms = 0;
	//cl_platform_id platform = 0;
	//cl_platform_id platforms[100];

	/* Setup OpenCL environment. */
	err = clGetPlatformIDs(max_num_platforms, platforms, &num_platforms);

	printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
	for (int i = 0; i<num_platforms; i++)
	{
		char buffer[10240];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}

	if (num_platforms == 0)
		return 1;

	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, &num_devices);

	printf("%u OpenCL device(s) found on platform\n", num_devices);
	for (int i = 0; i < num_devices; i++)
	{
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		printf("  -- %d --\n", i);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_NAME = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VENDOR = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DRIVER_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		//CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_uint), &buf_uint, NULL));
		//printf("  DEVICE_MAX_WORK_GROUP_SIZE = %u\n", (unsigned int)buf_uint);
		//CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_uint), &buf_uint, NULL));
		//printf("  DEVICE_MAX_WORK_ITEM_SIZES = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}

	return 0;
}

int clRngExample(void)
{
	cl_int err;
	const unsigned MaxNumPlatforms = 100;
	cl_platform_id platforms[MaxNumPlatforms];
	cl_uint num_platforms = 0;

	const unsigned NumDevices = 100;
	cl_uint num_devices = NumDevices;
	cl_device_id devices[NumDevices];

	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	cl_program program = 0;
	cl_kernel kernel = 0;
	cl_event event = 0;
	cl_mem bufIn, bufOut;
	float *out;
	char *clrng_root;
	char include_str[1024];
	char build_log[4096];
	size_t i = 0;
	size_t numWorkItems = 64;		// WorkItems are executed in parallel, as tasks on each GPU core? What is the optimal number of tasks/threads to run on GPU in parallel?
	clrngMrg31k3pStream *streams = 0;
	size_t streamBufferSize = 0;
	size_t kernelLines = 0;

	/* Sample kernel that calls clRNG device-side interfaces to generate random numbers */
	const char *kernelSrc[] = {
		"    #define CLRNG_SINGLE_PRECISION                                   \n",
		"    #include <cl/include/mrg31k3p.clh>                               \n",
		"                                                                     \n",
		"    __kernel void example(__global clrngMrg31k3pHostStream *streams, \n",
		"                          __global float *out)                       \n",
		"    {                                                                \n",
		"        int gid = get_global_id(0);                                  \n",
		"                                                                     \n",
		"        clrngMrg31k3pStream workItemStream;                          \n",
		"        clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream,   \n",
		"                                                     &streams[gid]); \n",
		"                                                                     \n",
		"        out[gid] = clrngMrg31k3pRandomU01(&workItemStream);          \n",
		"    }                                                                \n",
		"                                                                     \n",
	};

	/* Setup OpenCL environment. */
	err = clGetPlatformIDs(MaxNumPlatforms, platforms, &num_platforms);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, &num_devices);

	openClAttributes(devices, num_devices, platforms, MaxNumPlatforms, num_platforms);

	props[1] = (cl_context_properties)platforms[0];
	ctx = clCreateContext(props, 1, &devices[0], NULL, NULL, &err);
	queue = clCreateCommandQueue(ctx, devices[0], 0, &err);

	/* Make sure CLRNG_ROOT is specified to get library path */
	clrng_root = getenv("CLRNG_ROOT");
	if (clrng_root == NULL) printf("\nSpecify environment variable CLRNG_ROOT as described\n");
	strcpy(include_str, "-I ");
	strcat(include_str, clrng_root);
	strcat(include_str, "/include");

	/* Create sample kernel */
	kernelLines = sizeof(kernelSrc) / sizeof(kernelSrc[0]);
	program = clCreateProgramWithSource(ctx, kernelLines, kernelSrc, NULL, &err);
	err = clBuildProgram(program, 1, &devices[0], include_str, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("\nclBuildProgram has failed\n");
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 4096, build_log, NULL);
		printf("%s", build_log);
	}
	kernel = clCreateKernel(program, "example", &err);

	/* Create streams */
	streams = clrngMrg31k3pCreateStreams(NULL, numWorkItems, &streamBufferSize, (clrngStatus *)&err);

	/* Create buffers for the kernel */
	bufIn  = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,  streamBufferSize, streams, &err);
	bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, numWorkItems * sizeof(cl_float), NULL, &err);

	/* Setup the kernel */
	err = clSetKernelArg(kernel, 0, sizeof(bufIn), &bufIn);
	err = clSetKernelArg(kernel, 1, sizeof(bufOut), &bufOut);

	TimerCycleAccurateArray	timer;
	timer.reset();
	timer.timeStamp();
	/* Execute the kernel and read back results */
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &numWorkItems, NULL, 0, NULL, &event);
	err = clWaitForEvents(1, &event);	// Wait for all work items to finish
	timer.timeStamp();
	printf("doWork: OpenCL took %f seconds and ran at %zd floats/second\n", timer.getAverageDeltaInSeconds(), (size_t)((double)numWorkItems / timer.getAverageDeltaInSeconds()));
	out = (float *)malloc(numWorkItems * sizeof(out[0]));
	err = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, numWorkItems * sizeof(out[0]), out, 0, NULL, NULL);

	float sum = 0.0;
	for (unsigned i = 0; i < numWorkItems; i++)
	{
		//printf("%f\n", out[i]);
		sum += out[i];
	}
	printf("Average = %f\n", sum / numWorkItems);

	/* Release allocated resources */
	clReleaseEvent(event);
	free(out);
	clReleaseMemObject(bufIn);
	clReleaseMemObject(bufOut);

	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	printf("Ran successfully!\n");

	return 0;
}