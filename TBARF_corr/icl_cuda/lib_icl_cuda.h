#pragma once
//#define DEBUG_MODE

/// XXX TODO not portable right now!
//#ifndef NVCC_PATH
//	#define NVCC_PATH "%CUDA_PATH%/bin/nvcc.exe"
//#endif


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#ifndef _WIN32
	#include <stdbool.h>
#else
//	typedef int bool;
//	#define false 0
//	#define true 1
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
	
	/*#ifndef __APPLE__
	#include <CL/cl.h>
#else
	#include <OpenCL/opencl.h>
#endif
*/


/* OpenCL types re-definition */
#define CL_FALSE                                    0
#define CL_TRUE                                     1
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_USE_HOST_PTR                         (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)
#define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define CL_SUCCESS CUDA_SUCCESS


/* Define alignment keys */
#if (defined( __GNUC__ ) || defined( __IBMC__ ))
	#define CL_ALIGNED(_x)          __attribute__ ((aligned(_x)))
#elif defined( _WIN32) && (_MSC_VER)
	/* Alignment keys neutered on windows because MSVC can't swallow function arguments with alignment requirements     */
	/* http://msdn.microsoft.com/en-us/library/373ak2y1%28VS.71%29.aspx                                                 */
	/* #include <crtdefs.h>                                                                                             */
	/* #define CL_ALIGNED(_x)          _CRT_ALIGN(_x)                                                                   */
	#define CL_ALIGNED(_x)
#else
   #warning  Need to implement some method to align data here
   #define  CL_ALIGNED(_x)
#endif


typedef int cl_event;	
typedef int cl_bool;
//typedef bool cl_bool;
typedef int cl_map_flags;
typedef unsigned long cl_ulong;
typedef unsigned cl_uint;
typedef int cl_int;
typedef float cl_float;
typedef union { cl_float  CL_ALIGNED(16) s[4]; } cl_float4;
typedef double cl_double;
typedef union { cl_double  CL_ALIGNED(16) s[4]; } cl_double4;
typedef unsigned cl_device_type;
typedef int cl_device_id;
typedef int cl_device_fp_config;
typedef int cl_device_mem_cache_type;
typedef int cl_device_local_mem_type;
typedef CUmodule cl_program;

typedef cl_ulong            cl_bitfield;
//typedef cl_bitfield         cl_device_type;


#define ICL_ASSERT(__condition, __message, ...) \
if(!(__condition)) { \
	fprintf(stderr, "Water Assertion failure in %s#%d:\n", __FILE__, __LINE__); \
	printf(__message, ##__VA_ARGS__); \
	printf("\n"); \
	exit(-1); \
}

#define ICL_INFO(__message, ...) { \
	printf(__message, ##__VA_ARGS__); fflush(stdout);\
}

#ifdef DEBUG_MODE
		#define ICL_DEBUG(__message, ...) { \
				printf(__message, ##__VA_ARGS__); \
		}
#else
		#define ICL_DEBUG(__message, ...)
#endif

#define ICL_INIT_DEVS		(1 << 1)
#define ICL_RELEASE_DEVS	(1 << 2)
#define ICL_PRINT_DEV_SHORT	(1 << 3)
#define ICL_PRINT_DEV_INFOS	(1 << 4)

#define ICL_CREATE_BUFFER	(1 << 5)
#define ICL_WRITE_BUFFER	(1 << 6)
#define ICL_READ_BUFFER		(1 << 7)
#define ICL_RELEASE_BUFFER	(1 << 8)

#define ICL_CREATE_KERNEL	(1 << 9)
#define ICL_RUN_KERNEL		(1 << 10)
#define ICL_RELEASE_KERNEL	(1 << 11)

typedef uint64_t	icl_bitfield;
typedef icl_bitfield	icl_device_type;
#define ICL_CPU		(1 << 1)
#define ICL_GPU		(1 << 2)	
#define ICL_ACL		(1 << 3)
#define ICL_ALL		0xFFFFFFFF

#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
typedef icl_bitfield		icl_mem_flag;
#define ICL_MEM_READ_WRITE		(1 << 0)
#define ICL_MEM_WRITE_ONLY		(1 << 1)
#define ICL_MEM_READ_ONLY		(1 << 2)
#define ICL_MEM_USE_HOST_PTR	(1 << 3)
#define ICL_MEM_ALLOC_HOST_PTR	(1 << 4)
#define ICL_MEM_COPY_HOST_PTR	(1 << 5)
// reserved			(1 << 6)    
#define ICL_MEM_HOST_WRITE_ONLY	(1 << 7)
#define ICL_MEM_HOST_READ_ONLY	(1 << 8)
#define ICL_MEM_HOST_NO_ACCESS	(1 << 9)

typedef icl_bitfield	icl_create_kernel_flag;
#define ICL_SOURCE		(1 << 1)
#define ICL_BINARY		(1 << 2)
#define ICL_STRING		(1 << 3)
#define ICL_NO_CACHE	(1 << 4)

typedef uint32_t	icl_blocking_flag;
#define ICL_BLOCKING		1
#define ICL_NON_BLOCKING	0

typedef struct _icl_local_device {
	// internal info
	CUdevice device;
	CUcontext context;
	CUstream  queue; // the closest approximation to OpenCL queue is a CUDA stream

	// buffers info
//	cl_ulong mem_size; // memory of the device
//	cl_ulong mem_available; // memory still available, reduced after each buffer allocation
//	cl_ulong max_buffer_size; // max size of a buffer
	
	// device info
	char name[128];
	cl_device_type type;
	char vendor[128];
	char version[128];
	CUdevprop property;

	cl_uint max_compute_units;
/*	cl_uint max_clock_frequency;
	cl_uint max_work_item_dimensions;
	size_t* max_work_item_sizes;
	size_t max_work_group_size;


	cl_bool image_support;
	cl_device_fp_config single_fp_config;
	cl_bool endian_little;
	char *extensions;
	
	cl_device_mem_cache_type mem_cache_type;
	cl_ulong global_mem_cacheline_size;
	cl_ulong global_mem_cache_size;

	cl_ulong max_constant_buffer_size;

	cl_device_local_mem_type local_mem_type;
	cl_ulong  local_mem_size;
*/
} icl_local_device;

typedef struct _icl_local_buffer {
	CUdeviceptr mem;
	size_t size;
	icl_local_device* local_dev; // derive the device directly from the buffer
} icl_local_buffer;

typedef struct _icl_local_kernel {
//	CUfunction cl_ker;	
	CUmodule module;
	CUfunction function;

	char** args; // is not important the type, is used only for the free
	int num_args; 
	icl_local_device* local_dev;
} icl_local_kernel;

typedef struct _icl_local_event {
	int num_cl_event;
///	cl_event cl_ev[1]; // hard to replace... maybe we need to use stream here
	CUstream cl_ev[1];
} icl_local_event;


typedef struct _icl_device {
	uint32_t node_id; 	// id of the node
	uint32_t device_id;	// id of the device in that machine
} icl_device;

typedef struct _icl_buffer {
	uint64_t buffer_add;
	icl_device* device;
} icl_buffer;

typedef struct _icl_kernel {
	uint64_t kernel_add;
	icl_device* device;
} icl_kernel;

typedef struct _icl_event {
	uint64_t event_add;
	icl_device* device;
} icl_event;

typedef struct _arg_info {
	size_t		 size;
	const void*  val;
} arg_info;


#ifdef __cplusplus
namespace {
#endif
// replacing uint32_t with int (CUDA uses this for device number)
#ifndef _WIN32
icl_local_device* local_devices __attribute__ ((aligned (16)));
int local_num_devices __attribute__ ((aligned (16)));
icl_device* devices __attribute__ ((aligned (16)));
uint32_t num_devices __attribute__ ((aligned (16)));
#else
icl_local_device* local_devices;
int	local_num_devices;
icl_device* devices;
uint32_t num_devices;
#endif
#ifdef __cplusplus
}
#endif



void icl_init_devices(icl_device_type device_type);
void icl_release_devices();
uint32_t icl_get_num_devices();
icl_device* icl_get_device(uint32_t id);

icl_buffer* icl_create_buffer(icl_device* device, icl_mem_flag flag, size_t size);
void _icl_create_buffer(icl_device* device, icl_mem_flag flag, size_t size, icl_buffer*);

void icl_read_buffer(const icl_buffer* buffer, icl_blocking_flag blocking, size_t size, void* source_ptr, icl_event* wait_event, icl_event* event);
void icl_write_buffer(icl_buffer* buffer, icl_blocking_flag blocking, size_t size, const void* source_ptr, icl_event* wait_event, icl_event* event);

void* icl_map_buffer(icl_buffer* buf, cl_bool blocking, cl_map_flags map_flags, size_t size, icl_event* wait_event, icl_event* event);
void icl_unmap_buffer(icl_buffer* buf, void* mapped_ptr, icl_event* wait_event, icl_event* event);

void icl_copy_buffer(icl_buffer* src_buf, icl_buffer* dest_buf, size_t size, icl_event* event_wait, icl_event* event);

void icl_release_buffer(icl_buffer* buffer);
void icl_release_buffers(uint32_t num, ...);

void icl_finish(icl_device* device);

icl_kernel* icl_create_kernel(icl_device* device, char* file_name, char* kernel_name, char* build_options, icl_create_kernel_flag flag);
void _icl_create_kernel(icl_device* device, char* file_name, char* kernel_name, char* build_options, icl_create_kernel_flag flag, icl_kernel* ret);

void icl_run_kernel(const icl_kernel* kernel, uint32_t work_dim, const size_t* global_work_size, const size_t* local_work_size, 
						icl_event* wait_event, icl_event* event, uint32_t num_args, ...);

void _icl_run_kernel(const icl_kernel* kernel, uint32_t work_dim, const size_t* global_work_size, const size_t* local_work_size, 
						icl_event* wait_event, icl_event* event, uint32_t num_args, const arg_info* args);

void icl_release_kernel(icl_kernel* kernel);
void icl_release_kernels(uint32_t num, ...);

void icl_print_device_short_info(icl_device* device);
void icl_print_device_infos(icl_device* device);

icl_event* icl_create_event();
icl_event* icl_merge_events(uint32_t num_event, ...);
void icl_release_event(icl_event* event);
void icl_release_events(uint32_t num, ...);
void icl_wait_for_events(uint32_t num, ...);


// CUDA support functions
const char* _cudaError_string(cudaError_t error);
const char* _CUresult_string(CUresult res);