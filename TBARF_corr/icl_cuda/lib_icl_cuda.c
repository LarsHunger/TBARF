#include "lib_icl_cuda.h"

#ifndef _WIN32	
	#include <alloca.h>
#else
	#include <malloc.h>
//	#define alloca _alloca
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// 32 or 64 bit check
// Check windows
#if _WIN32 
 #if _WIN64
  #define ENV64BIT
 #else
  #define ENV32BIT
 #endif
#endif

// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENV64BIT
#else
#define ENV32BIT
#endif
#endif


/*
 * =====================================================================================
 *  OpenCL Internal Functions Declarations
 * =====================================================================================
 */

const char* _cudaError_string(cudaError_t error) {
	return cudaGetErrorString (error);
}

const char* _CUresult_string(CUresult res) {
	switch(res){
	case CUDA_SUCCESS: return "CUDA DRIVER - no errors";
	case CUDA_ERROR_INVALID_VALUE: return "CUDA DRIVER - invalid value";
	case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA DRIVER - out of memory";
	case CUDA_ERROR_NOT_INITIALIZED:
		return "CUDA DRIVER - driver not initialized";
	case CUDA_ERROR_DEINITIALIZED: return "CUDA DRIVER - deinitialized";
	case CUDA_ERROR_NO_DEVICE: return "CUDA DRIVER - no device";
	case CUDA_ERROR_INVALID_DEVICE:
		return "CUDA DRIVER - invalid device";
	case CUDA_ERROR_INVALID_IMAGE:
		return "CUDA DRIVER - invalid kernel image";
	case CUDA_ERROR_INVALID_CONTEXT:
		return "CUDA DRIVER - invalid context";
	case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: 
		return "CUDA DRIVER - context already current";
	case CUDA_ERROR_MAP_FAILED: return "CUDA DRIVER - map failed";
	case CUDA_ERROR_UNMAP_FAILED: return "CUDA DRIVER - unmap failed";
	case CUDA_ERROR_ARRAY_IS_MAPPED:
		return "CUDA DRIVER - array is mapped";
	case CUDA_ERROR_ALREADY_MAPPED:
		return "CUDA DRIVER - already mapped";
	case CUDA_ERROR_NO_BINARY_FOR_GPU:
		return "CUDA DRIVER - no gpu binary";
	case CUDA_ERROR_ALREADY_ACQUIRED:
		return "CUDA DRIVER - already aquired";
	case CUDA_ERROR_NOT_MAPPED: return "CUDA DRIVER - not mapped";
	case CUDA_ERROR_INVALID_SOURCE:
		return "CUDA DRIVER - invalid source";
	case CUDA_ERROR_FILE_NOT_FOUND:
		return "CUDA DRIVER - file not found";
	case CUDA_ERROR_INVALID_HANDLE:
		return "CUDA DRIVER - invalid handle";
	case CUDA_ERROR_NOT_FOUND: return "CUDA DRIVER - not found";
	case CUDA_ERROR_NOT_READY: return "CUDA DRIVER - not ready";
	case CUDA_ERROR_LAUNCH_FAILED: return "CUDA DRIVER - launch failed";
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
		return "CUDA DRIVER - out of resources";
	case CUDA_ERROR_LAUNCH_TIMEOUT:
		return "CUDA DRIVER - launch timeout";
	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: 
		return "CUDA DRIVER - incompatible texturing";
	case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA DRIVER - not mapped as pointer";
	case CUDA_ERROR_UNKNOWN: return "CUDA DRIVER - unknown error";
	default: break;
	}
	return "invalid_error";	
}


static char* _icl_load_program_source (const char* filename, size_t* filesize);
static void _icl_save_program_binary (cl_program program, const char* binary_filename);
//static const char* _icl_error_string (cl_int err_code);
static const char* _icl_error_string (cudaError_t error);


/*
 * =====================================================================================
 *  OpenCL Device Functions
 * =====================================================================================
 */
void icl_finish(icl_device* device){
	uint32_t node = device->node_id;	
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");
/// XXX to fix by handling multiple contexts (devices) here
//	icl_local_device* ldev = &local_devices[device->device_id];	
	
	CUresult err_code = cuCtxSynchronize();	
	ICL_ASSERT(err_code == CL_SUCCESS, "Error on cudaDeviceSynchronize: \"%s\"",  _CUresult_string(err_code));
}




void icl_init_devices(icl_device_type device_type) {
	int flag = 0;  /// XXX check this flag
	CUresult err_code = cuInit(flag);	
	ICL_ASSERT(err_code == CL_SUCCESS, "Error in CUDA init: \"%s\"", _CUresult_string(err_code));

	devices = NULL;
	num_devices = 0;

	local_devices = NULL;
	local_num_devices = 0;	
	err_code = cuDeviceGetCount(&local_num_devices);
	
	ICL_ASSERT(err_code == CL_SUCCESS, "Error in CUDA init: \"%s\"", _CUresult_string(err_code));

	/* no platforms for CUDA	
	cl_platform_id *cl_platforms;
	cl_uint cl_num_platforms = _icl_get_num_platforms();
	if (cl_num_platforms != 0) {
	cl_platforms = (cl_platform_id *)alloca(cl_num_platforms * sizeof(cl_platform_id));
	_icl_get_platforms(cl_num_platforms, cl_platforms);

	#pragma omp parallel num_threads(cl_num_platforms)
	{
	#pragma omp for reduction(+ : local_num_devices)
	for (cl_uint i = 0; i < cl_num_platforms; i++) {
	cl_uint cl_num_devices =
	_icl_get_num_devices(&cl_platforms[i], device_type);
	local_num_devices += cl_num_devices;
	}
	}
	}
	*/

	// XXX
	printf("Found %d CUDA devices\n", local_num_devices);

	if (local_num_devices != 0) {
		cl_uint index = 0;
		cl_uint cl_num_devices = local_num_devices;
		cl_device_id *cl_devices = (cl_device_id *)alloca(cl_num_devices * sizeof(cl_device_id));
		unsigned j;

		local_devices = (icl_local_device *)malloc(local_num_devices * sizeof(icl_local_device));		

		 // for each CUDA device
		for (j = 0; j < cl_num_devices; ++j) 
		{
			// Note(Biagio): This flag can potentially affects performance.
			// Right now, I set up a default context scheduling, but it needs more investigations.
			// see http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html
			unsigned flag = CU_CTX_SCHED_AUTO ; 

			CUresult res;
			icl_local_device *ldev = &local_devices[index + j];
			cl_device_id *cl_dev = &cl_devices[j];
			
			res = cuDeviceGet(&ldev->device, j);
			ICL_ASSERT(res == CL_SUCCESS, "Error in CUDA init: \"%s\"", _CUresult_string(res));						
	
			res = cuCtxCreate(&ldev->context, flag, ldev->device); // clCreateContext(NULL, 1, cl_dev, NULL, NULL, &err_code);
			ICL_ASSERT(res == CL_SUCCESS, "Error in CUDA init: \"%s\"", _CUresult_string(res));

			res = cuStreamCreate(&ldev->queue, 0);
			ICL_ASSERT(res == CL_SUCCESS, "Error in CUDA init: \"%s\"", _CUresult_string(res));
			//ldev->queue = clCreateCommandQueue(ldev->context, *cl_dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err_code);						
			

			// Device Info 
			cuDeviceGetProperties(&ldev->property, ldev->device);
									
			cuDeviceGetName(ldev->name, 128, ldev->device); 
			ldev->type = ICL_GPU;
			sprintf(ldev->vendor, "NVIDIA Corporation");
			int major, minor;
			cuDeviceComputeCapability(&major, &minor, ldev->device)	;
			sprintf(ldev->version, "Compute Capability %d %d", major, minor);						
		}

		
		cl_uint cur_pos = 0;
//		cl_uint shift;
		//General devices list ordered: CPUs, GPUs, ACCs
		num_devices = local_num_devices;
		devices = (icl_device *)malloc(local_num_devices * sizeof(icl_device));	
//		for (shift = 1; shift < 4; ++shift) {
//			cl_uint type = (1 << shift);
			cl_uint i;
			for (i = 0; i < local_num_devices; ++i) {
				icl_local_device *ldev = &local_devices[i];
//				if (ldev->type == type) 
				{
					devices[cur_pos].node_id = 0;
					devices[cur_pos].device_id = i;
					cur_pos++;
				}
			}
//		}
			
	}
}

void icl_release_devices() {
	CUresult res;

//	#pragma omp parallel num_threads (local_num_devices)
	{
		cl_uint i;
//		#pragma omp for 
		for (i = 0; i < local_num_devices; ++i) {
			icl_local_device* ldev = &local_devices[i];

			// release the queue
			res = cuStreamDestroy(ldev->queue);
			ICL_ASSERT(res == CL_SUCCESS, "Error in CUDA release: \"%s\"", _CUresult_string(res));
						
			// release the context
			res = cuCtxDestroy(ldev->context);
			ICL_ASSERT(res == CL_SUCCESS, "Error in CUDA context destroy: \"%s\"", _CUresult_string(res));

		}
	}
	free(local_devices);
	free(devices);
	
}


uint32_t icl_get_num_devices() {
	return num_devices;
}

icl_device* icl_get_device(uint32_t id) {
	ICL_ASSERT(id < num_devices, "Error accessing device with wrong ID");
	return &devices[id];
}

/*
 * =====================================================================================
 *  OpenCL Buffer Functions
 * =====================================================================================
 */

void _icl_create_buffer(icl_device* device, icl_mem_flag flag, size_t size, icl_buffer* buffer) {
	uint32_t node = device->node_id;
	icl_local_device* ldev = &local_devices[device->device_id];
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");
//	ICL_ASSERT(size <= ldev->max_buffer_size, "Error creating buffer: \"Buffer size is too big\"");
//	ICL_ASSERT(size <= ldev->mem_available, "Error creating buffer: \"Out of memory\"");
	
	// create buffer	
	icl_local_buffer* lbuf = (icl_local_buffer*)malloc(sizeof(icl_local_buffer));
	CUresult res;
	res = cuMemAlloc(&lbuf->mem, size);
		//lbuf->mem = clCreateBuffer(ldev->context, flag, size, NULL, &err_code);
	ICL_ASSERT(res == CL_SUCCESS, "Error creating buffer: \"%s\"", _CUresult_string(res));
//	lbuf->size = size;
//	lbuf->local_dev = ldev;
//	ldev->mem_available -= size;

	buffer->device = device;
	buffer->buffer_add = (cl_ulong)lbuf;	
}

icl_buffer* icl_create_buffer(icl_device* device, icl_mem_flag flag, size_t size) {
	icl_buffer* buffer = (icl_buffer*)malloc(sizeof(icl_buffer));
	_icl_create_buffer(device, flag, size, buffer);
	return buffer;
}

void icl_release_buffer(icl_buffer* buffer) {
	if (buffer == NULL) { return; }
	uint32_t node = buffer->device->node_id;
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");

	icl_local_device* ldev = &local_devices[buffer->device->device_id];
	icl_local_buffer* lbuf = (icl_local_buffer*)(buffer->buffer_add);
	
	// free the buffer
	CUresult res = cuMemFree(lbuf->mem);
	//cl_int err_code = clReleaseMemObject(lbuf->mem);
	ICL_ASSERT(res == CL_SUCCESS, "Error creating buffer: \"%s\"", _CUresult_string(res));
//	ldev->mem_available += lbuf->size;
	free(lbuf);
	free(buffer);
	buffer = NULL;	
}

void icl_release_buffers(uint32_t num, ...) {
	va_list arg_list;
	int i;
	va_start(arg_list, num);
	for (i = 0; i < num; i++){
		icl_release_buffer(va_arg(arg_list, icl_buffer*));
	}
	va_end(arg_list);
}

/// XXX TODO(Biagio) not ported yet...a bit tricky
// pay attention they are pointer to pointer
void _icl_set_event(icl_device* dev, icl_event* wait_event, icl_event* event, cl_event** wait_ev, cl_event** ev, cl_uint* num) {
	
	/// XXX todo tofix 
	return;
/*
	if (event != NULL) {
		icl_local_event* levent = (icl_local_event*)(event->event_add);
		ICL_ASSERT(levent->num_cl_event == 1, "Error in events handler: Expected one single event");
		event->device = dev;
		clReleaseEvent(*(levent->cl_event)); // release before using it, in case of reuse
		*ev = levent->cl_ev;
	}

	if (wait_event != NULL) {
		icl_local_event* levent;
		ICL_ASSERT(wait_event->device == dev, "Error in events handler: Waiting for an event generated in a different device");
		levent = (icl_local_event*)(wait_event->event_add);
		*num = levent->num_cl_event;
		*wait_ev = levent->cl_ev;
	}
*/
}


void icl_write_buffer(icl_buffer* buffer, icl_blocking_flag blocking, size_t size, const void* source_ptr, icl_event* wait_event, icl_event* event) {
	uint32_t node = buffer->device->node_id;		
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only\n");
	icl_local_buffer* lbuf = (icl_local_buffer*)(buffer->buffer_add);

/// XXX events here
///	cl_event* ev = NULL; cl_event* wait_ev = NULL; cl_uint num = 0;
///	_icl_set_event(buffer->device, wait_event, event, &wait_ev, &ev, &num);
/// CUstream	
	CUresult res;
	if(blocking)
		res = cuMemcpyHtoD(lbuf->mem, source_ptr, size); 
	else
		res = cuMemcpyHtoDAsync(lbuf->mem, source_ptr, size, NULL); 
	//cl_int err_code = clEnqueueWriteBuffer(lbuf->local_dev->queue, lbuf->mem, blocking, 0, size, source_ptr, num, wait_ev, ev);
	ICL_ASSERT(res == CUDA_SUCCESS, 	"Error writing buffer - %s\n", _CUresult_string(res));
}


void icl_read_buffer(const icl_buffer* buffer, icl_blocking_flag blocking, size_t size, void* source_ptr, icl_event* wait_event, icl_event* event) {
	uint32_t node = buffer->device->node_id;		
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only\n");
	icl_local_buffer* lbuf = (icl_local_buffer*)(buffer->buffer_add);
	
///	cl_event* ev = NULL; cl_event* wait_ev = NULL; cl_uint num = 0;
///	_icl_set_event(buffer->device, wait_event, event, &wait_ev, &ev, &num);
// XXX	CUstream stream;
	
	CUresult res;
	if(blocking)
		res = cuMemcpyDtoH(source_ptr, lbuf->mem, size);
	else
		res = cuMemcpyDtoHAsync(source_ptr, lbuf->mem, size, NULL);
	ICL_ASSERT(res == CUDA_SUCCESS,	"Error reading buffer %s\n", _CUresult_string(res));	
}

void icl_copy_buffer(icl_buffer* src_buf, icl_buffer* dest_buf, size_t size, icl_event* wait_event, icl_event* event) {
	cl_event* ev = NULL; cl_event* wait_ev = NULL; cl_uint num = 0;
//	_icl_set_event(wait_event, event, &wait_ev, &ev, &num);	
/// XXX
//	ICL_ASSERT(src_buf->dev->queue  == dest_buf->dev->queue, "Error: source and destination buffer have a different queue");
	
	icl_local_buffer* src_lbuf = (icl_local_buffer*)(src_buf->buffer_add);
	icl_local_buffer* dest_lbuf = (icl_local_buffer*)(dest_buf->buffer_add);
			
	CUresult res = cuMemcpyDtoDAsync(dest_lbuf->mem, src_lbuf->mem, size, 0	)	;
	ICL_ASSERT(res == CUDA_SUCCESS,	"Error copying buffers %s\n", _CUresult_string(res));	

}

/// XXX TODO
void* icl_map_buffer(icl_buffer* buf, cl_bool blocking, cl_map_flags map_flags, size_t size, icl_event* wait_event, icl_event* event) {
///	cl_event* ev = NULL; cl_event* wait_ev = NULL; cl_uint num = 0;
///	_icl_set_event(buf->device, wait_event, event, &wait_ev, &ev, &num);
	
	icl_local_buffer* lbuf = (icl_local_buffer*)(buf->buffer_add);
///	cl_int err_code;
///	void* ptr = clEnqueueMapBuffer(lbuf->local_dev->queue, lbuf->mem, blocking, map_flags, 0, size, num, wait_ev, ev, &err_code); 
///	ICL_ASSERT(err_code == CL_SUCCESS, "Error mapping buffer: \"%s\"",  _icl_error_string(err_code));
///	return ptr;
	return NULL;
}

/// XXX TODO
void icl_unmap_buffer(icl_buffer* buf, void* mapped_ptr, icl_event* wait_event, icl_event* event) {
///	cl_event* ev = NULL; cl_event* wait_ev = NULL; cl_uint num = 0;
///	_icl_set_event(buf->device, wait_event, event, &wait_ev, &ev, &num);

	icl_local_buffer* lbuf = (icl_local_buffer*)(buf->buffer_add);
///	cl_int err_code = clEnqueueUnmapMemObject(lbuf->local_dev->queue, lbuf->mem, mapped_ptr, num, wait_ev, ev);
///	ICL_ASSERT(err_code == CL_SUCCESS, "Error unmapping buffer: \"%s\"",  _icl_error_string(err_code));
}


/*
 * =====================================================================================
 *  OpenCL Kernel Functions
 * =====================================================================================
 */
static int str_ends_with(const char * str, const char * suffix) {
	if( str == NULL || suffix == NULL )	return 0;

	size_t str_len = strlen(str);
	size_t suffix_len = strlen(suffix);
	if(suffix_len > str_len) return 0;
	return 0 == strncmp( str + str_len - suffix_len, suffix, suffix_len );
}


/// XXX only ICL_SOURCE is currently supported
void _icl_create_kernel(icl_device* device, char* file_name, char* kernel_name, char* build_options, icl_create_kernel_flag flag, icl_kernel* kernel) {
	uint32_t node = device->node_id;
	icl_local_device* ldev;
	size_t filesize = 0;	
	
	char ptx_name[128];	
	char cuda_file_name[128];
	FILE *ptx_file;	

	ICL_ASSERT(flag == ICL_SOURCE, "Create kernel error: only ICL_SOURCE is currently supported.\n");	

	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");
	ICL_ASSERT(flag == ICL_SOURCE, "Error while creating a kernel: only ICL_SOURCE is currently supported\n");
	ldev = &local_devices[device->device_id];
	
	// 1. check if a ptx is available XXX so far it always recreates a new ptx	
		sprintf(ptx_name, "%s.ptx", file_name);
	ptx_file = NULL;//fopen(ptx_name, "r");

	// 2. if not, compile from source and generate ptx
	if(ptx_file == NULL)
	{
		/// XXX query a env var to seek for NVCC
		char *cuda_path;
		cuda_path = getenv("CUDA_PATH");
		if(cuda_path  == NULL) {
			printf("Error: CUDA_PATH not defined\n");
			exit(1);
		}
		char nvcc_path[128];
		sprintf(nvcc_path, "%s/bin/nvcc", cuda_path);

		// NVCC compiler requires .cu extension
		// if not available, we copy the kernel file in a temporary .cu file
		if(!str_ends_with(file_name,"cu"))	
		{						
			sprintf(cuda_file_name,"%s.cu", file_name);
			
			//creating a temporary cu file
			FILE *source_file = fopen(file_name, "r");
			FILE *cuda_file = fopen(cuda_file_name, "w");
			
			if(cuda_file == NULL){
				fclose(cuda_file);				
				printf("Error: cannot create a temporary CUDA file\n");
				exit(1);
			}

			// start the CUDA file with a prefix macro include
			char prefix[] = "#ifdef __CUDACC__ \n#include \"icl_macro.kernel\"\n  #endif \n\n";
			size_t len = strlen(prefix);
			fwrite(prefix, sizeof(char), len, cuda_file);
			char ch;
			while((ch = fgetc(source_file) ) != EOF ) fputc(ch, cuda_file);
			
			fclose(source_file);
			fclose(cuda_file);
			printf("Created a temporary .cu file (%s)\n", cuda_file_name);

			// now we use a different input file_name
		}
		else {
			strcpy(cuda_file_name, file_name);
		}

		// creation of the ptx file		
		char cmd[256];
		
		// detect if 32 or 64 bit 
		int rCode; 
		#ifdef ENV32BIT
		char mode[] = "-m 32 ";
		#else
		char mode[] = "-m 64 ";
		#endif
		sprintf(cmd, "\"%s\" %s -ptx -o %s %s %s -arch sm_20", nvcc_path, mode, ptx_name, build_options, cuda_file_name);
		printf("Compilation command: %s\n", cmd);
		rCode = system(cmd);
		ICL_INFO("Compilation of %s exited with %d\n", file_name, rCode);
		ICL_ASSERT(rCode == 0, "Compilation failed\n");

		// check if the ptx has been successfully created
		ptx_file = fopen(ptx_name, "r");
		ICL_ASSERT(ptx_file != NULL, "Error while compiling %s in a ptx format\n", file_name);					
		fclose(ptx_file);
	}

	// 3. create a CUmodule and CUfunction from the ptx	
	CUfunction function;
	if ((kernel_name != NULL) && (*kernel_name != '\0')) {
		CUresult res;
		icl_local_kernel* lkernel = (icl_local_kernel*)malloc(sizeof(icl_local_kernel));		
		//lkernel->cl_ker = clCreateKernel(program, kernel_name, &err_code);

		printf("loading module %s\n", ptx_name);
		res = cuModuleLoad(&lkernel->module, ptx_name);
		ICL_ASSERT(res == CL_SUCCESS, "Error loading module: \"%s\"\n", _CUresult_string(res));					

		res = cuModuleGetFunction(&lkernel->function, lkernel->module, kernel_name);		
		ICL_ASSERT(res == CL_SUCCESS, "Error while extracting the function from the module (is the function name \"%s\" correct?): \"%s\"", kernel_name, _CUresult_string(res));
		function = lkernel->function;

		lkernel->local_dev = ldev;
		lkernel->args = NULL;
		lkernel->num_args = 0;
		kernel->device = device;
		kernel->kernel_add = (cl_ulong)lkernel;
	} // 3

	// debug print of CUfunction attributes
	printf("Kernel (CUfunction) attributes:\n");
	int val;
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
	printf("  MAX_THREADS_PER_BLOCK %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);
	printf("  SHARED_SIZE_BYTES %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function);
	printf("  CONST_SIZE_BYTES %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
	printf("  LOCAL_SIZE_BYTES %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
	printf("  NUM_REGS %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PTX_VERSION, function);
	printf("  PTX_VERSION %d\n", val); 
	cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function);
	printf("  BINARY_VERSION %d\n", val); 
}

icl_kernel* icl_create_kernel(icl_device* device, char* file_name, char* kernel_name, char* build_options, icl_create_kernel_flag flag) {
	icl_kernel* kernel = (icl_kernel*)malloc(sizeof(icl_kernel));
	_icl_create_kernel(device, file_name, kernel_name, build_options, flag, kernel);
	return kernel;
}

void icl_release_kernel(icl_kernel* kernel) {
	uint32_t node;
	icl_local_kernel* lker;
	int err;
	unsigned i;
	if (kernel == NULL) { return; }
	node = kernel->device->node_id;

	ICL_ASSERT(node == 0, "Function must be invoked on local devices only\n");

	// icl_local_device* ldev = &local_devices[kernel->device->device_id];
	lker = (icl_local_kernel*)(kernel->kernel_add);
	
	//cl_int err_code = clReleaseKernel(lker->cl_ker);
	err = cuModuleUnload(lker->module);

	ICL_ASSERT(err == CL_SUCCESS, "Error releasing kernel\n");
	for(i = 0; i < lker->num_args; ++i)
		free(lker->args[i]);
	free(lker->args);
	free(lker);
	free(kernel);
}

void icl_release_kernels(uint32_t num, ...){
	unsigned i; 
	va_list arg_list;
	va_start(arg_list, num);
	for (i = 0; i < num; i++){
		icl_release_kernel(va_arg(arg_list, icl_kernel*));
	}
	va_end(arg_list);
}


void _icl_run_kernel(const icl_kernel* kernel, uint32_t work_dim, const size_t* global_work_size, const size_t* local_work_size, 
						icl_event* wait_event, icl_event* event, uint32_t num_args, const arg_info* args) {
	cl_event* ev = NULL; cl_event* wait_ev = NULL; 
	cl_uint num = 0;
	icl_local_kernel* lkernel;
	unsigned i;	
		
	unsigned int gridDimX, gridDimY, gridDimZ;
	unsigned int blockDimX, blockDimY, blockDimZ;
	
	unsigned int sharedMemBytes = 0;	
	static void *arg_buffer[64]; 	

	// set sizes
	blockDimX = local_work_size[0];	
	blockDimY = (work_dim > 1) ? local_work_size[1] : 1;
	blockDimZ = (work_dim > 2) ? local_work_size[2] : 1;
	
	gridDimX =	(global_work_size[0] + blockDimX - 1) / blockDimX;
	gridDimY  =	(work_dim > 1) ? (global_work_size[1] + blockDimY - 1) / blockDimY : 1;
	gridDimZ =	(work_dim > 2) ? (global_work_size[1] + blockDimZ - 1) / blockDimZ : 1;
	
//	printf("cuLaunchKernel args %d %d %d, %d %d %d, %d\n", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes);

/// TOFIX with events/streams
///	_icl_set_event(kernel->device, wait_event, event, &wait_ev, &ev, &num);

	lkernel = (icl_local_kernel*)(kernel->kernel_add);
	
	// Argument packing using CUDA 4.0 API for kernel parameter passing and kernel launch (advanced method),
	int offset = 0;
	for (i = 0; i < num_args; i++) {		
		size_t arg_size = (args+i)->size;
		void *arg_val = (void*) ((args+i)->val);

		// buffer argument
		if (arg_size == 0) {
			icl_buffer *buffer = (icl_buffer *) arg_val;
			icl_local_device* ldev = &local_devices[buffer->device->device_id];			
			icl_local_buffer* lbuf = (icl_local_buffer*)(buffer->buffer_add);			
			arg_buffer[i] = & (lbuf->mem);			
		}
		// data argument
		else {
			arg_buffer[i] = arg_val;
		}
	
	}
		
	CUresult res = cuLaunchKernel(lkernel->function,
			gridDimX, gridDimY, gridDimZ, 
			blockDimX, blockDimY, blockDimZ, 
			0, 
			0, arg_buffer, NULL);

	ICL_ASSERT(res == CUDA_SUCCESS, "Error in cuLaunchKernel: \"%s\"\n", _CUresult_string(res));	
}

void icl_run_kernel(const icl_kernel* kernel, uint32_t work_dim, const size_t* global_work_size, 
					const size_t* local_work_size, icl_event* wait_event, icl_event* event, uint32_t num_args, ...) {
	size_t arg_size;
	const void *arg_val;
	va_list arg_list;
	arg_info* args;
	unsigned i;

	uint32_t node = kernel->device->node_id; 
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only\n");

	//loop through the arguments and call clSetKernelArg for each argument	
	args = (arg_info*) malloc(sizeof(arg_info)*num_args);

	va_start (arg_list, num_args);
	for (i = 0; i < num_args; i++) {		
		arg_info ai;
		arg_size = va_arg (arg_list, size_t);
		arg_val = va_arg (arg_list, void*);		
		ai.size = arg_size;
		ai.val  = arg_val;
		*(args+i) = ai;
	}
	
	_icl_run_kernel(kernel, work_dim, global_work_size, local_work_size, wait_event, event, num_args, args);

	va_end (arg_list);
	free(args);
}

/*
 * =====================================================================================
 *  OpenCL Print & Profile Functions
 * =====================================================================================
 */

void icl_print_device_infos(icl_device* device) {
	uint32_t node = device->node_id;
	icl_local_device* ldev;
	 
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");
	ldev = &local_devices[device->device_id];
	CUdevprop p = ldev->property;	
	ICL_INFO("maxThreadsPerBlock:    %d\n",  p.maxThreadsPerBlock);
	ICL_INFO("maxThreadsDim:         %d %d %d\n",  p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
	ICL_INFO("maxGridSize:           %d %d %d\n",  p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
	ICL_INFO("sharedMemPerBlock:     %d\n",  p.sharedMemPerBlock);
	ICL_INFO("totalConstantMemory:   %d\n",  p.totalConstantMemory);
	ICL_INFO("SIMDWidth:             %d\n",  p.SIMDWidth);
	ICL_INFO("memPitch:              %d\n",  p.memPitch);
	ICL_INFO("regsPerBlock:          %d\n",  p.regsPerBlock);
	ICL_INFO("clockRate:             %d\n",  p.clockRate);
	ICL_INFO("textureAlign:          %d\n",  p.textureAlign);
}


void icl_print_device_short_info(icl_device* device) {
	uint32_t node = device->node_id;
	icl_local_device* ldev;
	ICL_ASSERT(node == 0, "Function must be invoked on local devices only");
	ldev = &local_devices[device->device_id];
	ICL_INFO("CUDA GPU: %s | %s | %s | CU: %u\n", ldev->name, ldev->vendor, ldev->version, ldev->max_compute_units);
}

icl_event* icl_create_event() {
	icl_event* event;
	icl_local_event* levent = (icl_local_event*)malloc(sizeof(icl_local_event));
	levent->num_cl_event = 1;

	event = (icl_event*)malloc(sizeof(icl_event));
	event->event_add = (cl_ulong)levent;
	return event;
}


icl_event* icl_merge_events(uint32_t num_event, ...){
	icl_device* curr_device = NULL;
	icl_local_event* icl_ev;
	icl_event* event;
	size_t num_elems = 0;
	va_list arg_list;
	cl_uint i, j, c;

	if (num_event == 0) { return NULL; }
	
	va_start(arg_list, num_event);
	for (i = 0; i < num_event; i++){
		icl_event* evt = va_arg(arg_list, icl_event*);
		num_elems += ((icl_local_event*) evt->event_add)->num_cl_event;

		// Checks device compatibility 
		if (curr_device == NULL) { 	curr_device = evt->device; }
		ICL_ASSERT(evt->device == curr_device, "Merging of events allowed for the same device"); 
	}
	va_end(arg_list);
	
	icl_ev = (icl_local_event*)malloc(sizeof(icl_local_event) + (num_elems-1)*sizeof(cl_event));
	va_start(arg_list, num_event);
	for (i=0, c=0; c<num_event; ++c) { 
		icl_event* evt = va_arg(arg_list, icl_event*);
		for(j=0; j<((icl_local_event*) evt->event_add)->num_cl_event; ++j, ++i) {
			icl_ev->cl_ev[i] = ((icl_local_event*) evt->event_add)->cl_ev[j];
		}
	}
	va_end(arg_list);
	
	icl_ev->num_cl_event = num_elems;
	event = (icl_event*)malloc(sizeof(icl_event));
	event->event_add = (cl_ulong)icl_ev;
	ICL_ASSERT(curr_device != NULL, "Current device not set");
	event->device = curr_device;
	return event;
}

void icl_wait_for_events(uint32_t num, ...) {
	va_list arg_list;
	uint32_t i;
	if (num==0) { return; }
	
	va_start(arg_list, num);
	for (i = 0; i < num; i++) {		
		icl_local_event* evt = (icl_local_event*)va_arg(arg_list, icl_event*)->event_add;
///	xxx	clWaitForEvents(evt->num_cl_event, evt->cl_ev);
	}
	va_end(arg_list);
}

void icl_release_event(icl_event* event){
	if (event != NULL) {
		icl_local_event* levent = (icl_local_event*)(event->event_add);
		if (levent->num_cl_event == 1)
/// xxx			clReleaseEvent(*(levent->cl_ev));
	//	free(levent->cl_ev);
		free(levent);
		free(event);
		return;
	}
}

void icl_release_events(uint32_t num, ...){
	va_list arg_list;
	uint32_t i ;
	va_start(arg_list, num);
	for (i = 0; i < num; i++){
		icl_release_event(va_arg(arg_list, icl_event*));
	}
	va_end(arg_list);
}

// ----------------------------------------------------------- OpenCL Internal Functions --------------------------------------------------

/* 
* =====================================================================================
*  OpenCL Internal Load, Save, Error Functions
* =====================================================================================
*/

static char* _icl_load_program_source (const char* filename, size_t* filesize) { // remember to free the returned source
	FILE* fp; 
	long unsigned int size;
	char* source;
	ICL_ASSERT(filename != NULL && filesize != NULL, "Error input parameters");
	fp = fopen(filename, "rb");
	ICL_ASSERT(fp != NULL, "Error opening kernel file");
	ICL_ASSERT(fseek(fp, 0, SEEK_END) == 0, "Error seeking to end of file");
	size = ftell(fp);
	ICL_ASSERT(fseek(fp, 0, SEEK_SET) == 0, "Error seeking to begin of file");
	source = (char*)malloc(size+1);
	ICL_ASSERT(source != NULL, "Error allocating space for program source");
	ICL_ASSERT(fread(source, 1, size, fp) == size, "Error reading file");
	source[size] = '\0';
	*filesize = size; // this is the size useful for create program from binary
	ICL_ASSERT(fclose (fp) == 0, "Error closing the file");
	return source;
}

/// XXX to fix
static void _icl_save_program_binary (cl_program program, const char* binary_filename) {
//	size_t size_ret;
	ICL_ASSERT(binary_filename != NULL && program != NULL, "Error input parameters");	
/*
	cl_int err_code;
	err_code = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES, 0, NULL, &size_ret);
	ICL_ASSERT(err_code == CL_SUCCESS, "Error getting program info: \"%s\"", _icl_error_string(err_code));
	size_t* binary_size = (size_t *) alloca (size_ret);
	ICL_ASSERT(binary_size != NULL, "Error allocating binary_size");
	err_code = clGetProgramInfo (program, CL_PROGRAM_BINARY_SIZES, size_ret, binary_size, NULL);
	ICL_ASSERT(err_code == CL_SUCCESS,  "Error getting program info: \"%s\"", _icl_error_string(err_code));
		unsigned char* binary = (unsigned char *) alloca (sizeof (unsigned char) * (*binary_size));
	ICL_ASSERT(binary != NULL, "Error allocating binary");

	// get the binary
	err_code = clGetProgramInfo (program, CL_PROGRAM_BINARIES, sizeof (unsigned char *), &binary, NULL);
	ICL_ASSERT(err_code == CL_SUCCESS,  "Error getting program info: \"%s\"", _icl_error_string(err_code));

	FILE *fp = fopen (binary_filename, "w");
	ICL_ASSERT(fp != NULL, "Error opening binary file");
	ICL_ASSERT(fwrite (binary, 1, *binary_size, fp) ==  (size_t) *binary_size, "Error writing file");
	ICL_ASSERT(fclose (fp) == 0, "Error closing the file");
*/
}


