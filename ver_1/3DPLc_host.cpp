#include "3DPLc_host.h"

#define MIN(a,b) ((a) < (b) ? a : b)

// macro for 3d version (deprecated)
//#define LOCAL_SIZE 4 // this is injected also in the kernel code
//#define BLOCK_SIZE (LOCAL_SIZE*LOCAL_SIZE*LOCAL_SIZE)

#define OUTOFCORE_SIZE (128*128*128)


// macro for 1d linear version
#define LOCAL_GROUP_SIZE 64
#define LOOP_UNROLL 4


icl_kernel *reg_field1_kernel, *reg_field1b_kernel, *reg_field1u_kernel, *reg_field1bu_kernel;
icl_kernel *reg_field2_kernel, *reg_field3_kernel;
icl_kernel *reg_field_outofcore_kernel;
icl_kernel *irr_field_kernel, *irr_fieldb_kernel, *irr_fieldu_kernel, *irr_fieldbu_kernel;
icl_device* device;

icl_buffer* vecs_buf;
icl_buffer* y_buf;
icl_buffer* RF_buf;

cl_ulong local_memory_size;


void ocl_init(int deviceId, int nr_lines, size_t field_nr_points,int linelength, 
			  int fielddim_x,int fielddim_y,int fielddim_z, bool regular, bool outofcore)
{
	#ifdef CUDA
	printf("CUDA version\n");
	#else
	printf("OpenCL version\n");
	#endif

	icl_init_devices(ICL_ALL);
	size_t deviceNum = icl_get_num_devices();

	if (deviceNum != 0) {
		for(int i=0; i<deviceNum; i++){
			icl_device *temp = icl_get_device(i);
			printf("%d - ", i);
			icl_print_device_short_info(temp);
		}

		device = icl_get_device(deviceId);
		printf("\nSelected device: ");
		icl_print_device_short_info(device);
		char comp_flags[256];
		sprintf(comp_flags, "-D LOCAL_GROUP_SIZE=%d -D LOOP_UNROLL=%d", LOCAL_GROUP_SIZE, LOOP_UNROLL);
		reg_field1_kernel  = icl_create_kernel(device, "../turning_band.cl", "make_reg_field1", comp_flags, ICL_SOURCE);
		reg_field1b_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field1_blocking", comp_flags, ICL_SOURCE);
		reg_field1u_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field1_unroll", comp_flags, ICL_SOURCE);
		reg_field1bu_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field1_blocking_unroll", comp_flags, ICL_SOURCE);
		
		reg_field2_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field2", comp_flags, ICL_SOURCE);
		reg_field3_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field3", comp_flags, ICL_SOURCE);
		
		reg_field_outofcore_kernel = icl_create_kernel(device, "../turning_band.cl", "make_reg_field_outofcore_blocking", comp_flags, ICL_SOURCE);

		irr_field_kernel  = icl_create_kernel(device, "../turning_band.cl", "make_irr_field", comp_flags, ICL_SOURCE);
		irr_fieldb_kernel = icl_create_kernel(device, "../turning_band.cl", "make_irr_field_blocking", comp_flags, ICL_SOURCE);
		irr_fieldu_kernel = icl_create_kernel(device, "../turning_band.cl", "make_irr_field_unroll", comp_flags, ICL_SOURCE);
		irr_fieldbu_kernel = icl_create_kernel(device, "../turning_band.cl", "make_irr_field_blocking_unroll", comp_flags, ICL_SOURCE);

		vecs_buf   = icl_create_buffer(device, CL_MEM_READ_ONLY, sizeof(cl_double4) * nr_lines);
		y_buf      = icl_create_buffer(device, CL_MEM_READ_ONLY, sizeof(cl_double) * linelength * nr_lines + 1);
		if(regular) {
			size_t _size = outofcore ? (OUTOFCORE_SIZE) : (fielddim_x * fielddim_y * fielddim_z);
			RF_buf = icl_create_buffer(device, CL_MEM_READ_WRITE, sizeof(cl_double) * _size);

		}
		else        
			RF_buf = icl_create_buffer(device, CL_MEM_READ_WRITE, sizeof(cl_double4) * field_nr_points);
		
		 icl_print_device_infos(device);
	}
	else {
		fprintf(stderr, "Error: OpenCL device not found");
		exit(1); // failure exit
	}	
}

void ocl_finalize()
{	
	icl_release_buffer(vecs_buf);
	icl_release_buffer(y_buf);
	icl_release_buffer(RF_buf);
	icl_release_kernel(reg_field1_kernel);
	icl_release_kernel(reg_field1b_kernel);
	icl_release_kernel(reg_field1u_kernel);
	icl_release_kernel(reg_field1bu_kernel);
	icl_release_kernel(reg_field2_kernel);
	icl_release_kernel(reg_field3_kernel);
	icl_release_kernel(irr_field_kernel);
	icl_release_kernel(irr_fieldb_kernel);
	icl_release_kernel(irr_fieldu_kernel);
	icl_release_kernel(irr_fieldbu_kernel);
	icl_release_devices();
}


/* parallelization on grid */
void ocl_make_field1(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	

	size_t szGlobalWorkSize[1] = { fielddim_x*fielddim_y*fielddim_z };
	size_t szLocalWorkSize[1] = { LOCAL_GROUP_SIZE }; 	 	
		
	printf("\n\nocl_make_field1, global size (%d), local size (%d)\n", szGlobalWorkSize[0], szLocalWorkSize[0]);
	icl_run_kernel(reg_field1_kernel, 1, szGlobalWorkSize, szLocalWorkSize, NULL, NULL, 9,	
		sizeof(cl_int), (void*) &nr_lines,
		sizeof(cl_int), (void*) &fielddim_x,
		sizeof(cl_int), (void*) &fielddim_y,
		sizeof(cl_int), (void*) &fielddim_z,
		sizeof(cl_int), (void*) &linelength,
		(size_t)0, (void*) vecs_buf,
		(size_t)0, (void*) y_buf,
		(size_t)0, (void*) RF_buf,
		sizeof(cl_double), (void*) &resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}

void ocl_make_field1b(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	

	size_t szGlobalWorkSize[1] = { fielddim_x*fielddim_y*fielddim_z };
	size_t szLocalWorkSize[1] = { LOCAL_GROUP_SIZE }; 		
		
	printf("\n\nocl_make_field1 with blocking, global size (%d), local size (%d)\n", szGlobalWorkSize[0], szLocalWorkSize[0]);
	icl_run_kernel(reg_field1b_kernel, 1, szGlobalWorkSize, szLocalWorkSize, NULL, NULL, 9,
		sizeof(cl_int), (void*) &nr_lines,
		sizeof(cl_int), (void*) &fielddim_x,
		sizeof(cl_int), (void*) &fielddim_y,
		sizeof(cl_int), (void*) &fielddim_z,
		sizeof(cl_int), (void*) &linelength,
		(size_t)0, (void*) vecs_buf,
		(size_t)0, (void*) y_buf,
		(size_t)0, (void*) RF_buf,
		sizeof(cl_double), (void*) &resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}

void ocl_make_field1u(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	

	size_t szGlobalWorkSize[1] = { fielddim_x*fielddim_y*fielddim_z };
	size_t szLocalWorkSize[1] = { LOCAL_GROUP_SIZE }; 	
		
	printf("\n\nocl_make_field1 with unrolling factor %d, global size (%d), local size (%d)\n", LOOP_UNROLL, szGlobalWorkSize[0], szLocalWorkSize[0]);
	icl_run_kernel(reg_field1u_kernel, 1, szGlobalWorkSize, szLocalWorkSize, NULL, NULL, 9,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&fielddim_x,
		sizeof(cl_int), (void *)&fielddim_y,
		sizeof(cl_int), (void *)&fielddim_z,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}

void ocl_make_field1bu(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	

	size_t szGlobalWorkSize[1] = { fielddim_x*fielddim_y*fielddim_z };
	size_t szLocalWorkSize[1] = { LOCAL_GROUP_SIZE }; 	
		
	printf("\n\nocl_make_field1 with blocking and unrolling factor %d, global size (%d), local size (%d)\n", LOOP_UNROLL, szGlobalWorkSize[0], szLocalWorkSize[0]);
	icl_run_kernel(reg_field1bu_kernel, 1, szGlobalWorkSize, szLocalWorkSize, NULL, NULL, 9,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&fielddim_x,
		sizeof(cl_int), (void *)&fielddim_y,
		sizeof(cl_int), (void *)&fielddim_z,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}


/* parallelization on line loop */
void ocl_make_field2(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	
		
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 	
	float multiplier = nr_lines/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier) multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;
	
	printf("\n\nocl_make_field2, global size (%d), local size (%d)\n", szGlobalWorkSize, szLocalWorkSize);
	icl_run_kernel(reg_field2_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 9,
		sizeof(cl_int), (void*) &nr_lines,
		sizeof(cl_int), (void*) &fielddim_x,
		sizeof(cl_int), (void*) &fielddim_y,
		sizeof(cl_int), (void*) &fielddim_z,
		sizeof(cl_int), (void*) &linelength,
		(size_t)0, (void*) vecs_buf,
		(size_t)0, (void*) y_buf,
		(size_t)0, (void*) RF_buf,
		sizeof(cl_double), (void*) &resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}

/* unrool of all loops */
void ocl_make_field3(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	
			
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 	
	size_t size = nr_lines * fielddim_x * fielddim_y * fielddim_z;
	float multiplier = size/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier) multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;	
	
	printf("\n\nocl_make_field3, global size (%d), local size (%d)\n", szGlobalWorkSize, szLocalWorkSize);
	icl_run_kernel(reg_field3_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 9,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&fielddim_x,
		sizeof(cl_int), (void *)&fielddim_y,
		sizeof(cl_int), (void *)&fielddim_z,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double) * fielddim_x * fielddim_y * fielddim_z, &RF[0], NULL, NULL);
	icl_finish(device);
}


// non uniform grid methods
void ocl_make_field_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines,        &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf,    CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);
	icl_write_buffer(RF_buf,   CL_FALSE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);

	size_t size = field_nr_points;
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 
	float multiplier = size/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier)
		multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;
	
	printf("\n\nocl_make_field_irregular, global size %d, local size %d\n", szGlobalWorkSize, szLocalWorkSize);	

	icl_run_kernel(irr_field_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 6,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);
	icl_finish(device);
}

void ocl_make_fieldb_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines,        &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf,    CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);
	icl_write_buffer(RF_buf,   CL_FALSE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);

	size_t size = field_nr_points;
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 
	float multiplier = size/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier)
		multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;
	
	printf("\n\nocl_make_field_irregular with blocking, global size %d, local size %d\n", szGlobalWorkSize, szLocalWorkSize);	
	icl_run_kernel(irr_fieldb_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 6,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);
	icl_finish(device);
}

void ocl_make_fieldu_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines,        &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf,    CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);
	icl_write_buffer(RF_buf,   CL_FALSE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);

	size_t size = field_nr_points;
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 
	float multiplier = size/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier) multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;
	
	printf("\n\nocl_make_field_irregular with unrolling factor %d, global size %d, local size %d\n", LOOP_UNROLL, szGlobalWorkSize, szLocalWorkSize);	
	icl_run_kernel(irr_fieldu_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 6,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);
	icl_finish(device);
}


void ocl_make_fieldbu_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor)
{
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines,        &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf,    CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);
	icl_write_buffer(RF_buf,   CL_FALSE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);

	size_t size = field_nr_points;
	size_t szLocalWorkSize = LOCAL_GROUP_SIZE; 
	float multiplier = size/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier) multiplier += 1;
	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;
	
	printf("\n\nocl_make_field_irregular with blocking and unrolling (%d), global size %d, local size %d\n", LOOP_UNROLL, szGlobalWorkSize, szLocalWorkSize);	
	icl_run_kernel(irr_fieldbu_kernel, 1, &szGlobalWorkSize, &szLocalWorkSize, NULL, NULL, 6,
		sizeof(cl_int), (void *)&nr_lines,
		sizeof(cl_int), (void *)&linelength,
		(size_t)0, (void *)vecs_buf,
		(size_t)0, (void *)y_buf,
		(size_t)0, (void *)RF_buf,
		sizeof(cl_double), (void *)&resolutionfactor
	);

	icl_read_buffer(RF_buf, CL_TRUE, sizeof(cl_double4) * field_nr_points, &RF[0], NULL, NULL);
	icl_finish(device);
}


///////////////////////////////////////////////

void ocl_make_field_outcore(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
					const cl_double4* vecs, const double* y, double* RF, 
					double resolutionfactor)
{
	size_t xy = fielddim_x*fielddim_y;
	size_t z = fielddim_z;
	size_t overall_size = xy * z;
	size_t chunk_size = OUTOFCORE_SIZE;
	icl_write_buffer(vecs_buf, CL_FALSE, sizeof(cl_double4) * nr_lines, &vecs[0], NULL, NULL);
	icl_write_buffer(y_buf, CL_FALSE, sizeof(cl_double) * linelength*nr_lines+1, &y[0], NULL, NULL);	

	printf("ocl_make_field out-of-core with overall size of %d, chunked by %d\n", overall_size, chunk_size);

	// for each chunk, we send and receive back a new part of the field
	for(size_t offset = 0, bid = 0; offset < overall_size; offset += chunk_size, bid++)
	{	
		printf("block %d offset %d,", bid, offset);
		size_t current_chunk_size = MIN(overall_size-offset, chunk_size);
		size_t szGlobalWorkSize[1] = { current_chunk_size };
		size_t szLocalWorkSize[1] = { LOCAL_GROUP_SIZE }; 	 	
		unsigned long long in_offset = offset;

//		printf("\nocl_make_field out-of-core, global size (%d), local size (%d)\n", szGlobalWorkSize[0], szLocalWorkSize[0]);
		icl_run_kernel(reg_field_outofcore_kernel, 1, szGlobalWorkSize, szLocalWorkSize, NULL, NULL, 10,	
			sizeof(cl_int), (void*) &nr_lines,
			sizeof(cl_int), (void*) &fielddim_x,
			sizeof(cl_int), (void*) &fielddim_y,
			sizeof(cl_int), (void*) &fielddim_z,
			sizeof(cl_int), (void*) &linelength,
			(size_t)0, (void*) vecs_buf,
			(size_t)0, (void*) y_buf,
			(size_t)0, (void*) RF_buf,
			sizeof(cl_double), (void*) &resolutionfactor,
			sizeof(cl_ulong), (void*) &in_offset
		);
		icl_read_buffer(&RF_buf[0], CL_FALSE, sizeof(cl_double) * current_chunk_size, &RF[offset], NULL, NULL);
	}	
	icl_finish(device);
	printf("\n");
}
