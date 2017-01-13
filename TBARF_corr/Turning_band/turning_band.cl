// Note: some macros are defined during clBuildProgram
#ifndef __CUDACC__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable						// double support
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable	// 32 bit atomic op
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable			// 64 bit atomic op
#endif
///////////////////////////////////////////////////////////////////////// atomic support

#ifndef __CUDACC__
void atomic_add_double(volatile __global double *source, const double operand){
	*source += operand;
}

/*
void atomic_add_double(volatile __global double *source, const double operand) {
    union {
        volatile ulong ulongVal;
        double doubleVal;
    } prevVal, newVal;

	__global ulong* source_ulp = (volatile __global ulong*)source;
    do {
        prevVal.doubleVal = *source;
        newVal.doubleVal = prevVal.doubleVal + operand;
    } while (atom_cmpxchg(source_ulp, prevVal.ulongVal, newVal.ulongVal) != prevVal.ulongVal);
}
*/
#else
__device__ void atomic_add_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, (unsigned long long int)(val + assumed));
    } while (assumed != old);
//    return __longlong_as_double(old);
}
#endif


///////////////////////////////////////////////////////////////////////// regular field computation

/* Regular field creation. This version is not using local memory and tiling, and it merges 4 loops togheter. */
__kernel void make_reg_field1(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, 
	__global double* RF, 
	double resolutionfactor)
{
/*
	int i = get_global_id(0);	
	int j = get_global_id(1);
	int k = get_global_id(2);
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	//const int fielddim_xyz = fielddim_x*fielddim_yz;		
	const size_t rf_index = i*fielddim_yz + j*fielddim_z + k;
	double4 id4 = {k, j, i, 0};		
*/
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	int gid = get_global_id(0);			

	// linearize gid
	int k = gid / fielddim_yz; 
	int j = (gid % (fielddim_yz  )) / fielddim_y;
	int i = gid - j * fielddim_y - k * fielddim_yz;	
	double4 id4 = {k, j, i, 0};	

	double rf_value = 0;
	for(int l=0; l<nr_lines; l++)
	{			
		double linecoord = - dot(id4, vecs[l]);
		//double linecoord = - (i*vecs[l].x) - (j*vecs[l].y) - (k*vecs[l].z);
		size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
		double y_val = y[l*linelength+linepoint];		
		rf_value += y_val;				
	}
	RF[gid] = rf_value;
}


//#define BLOCK_SIZE (LOCAL_SIZE*LOCAL_SIZE*LOCAL_SIZE)

__kernel void make_reg_field1_blocking(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor
	)
{
	__local double4 loc_vecs[LOCAL_GROUP_SIZE];
	const int block_size = LOCAL_GROUP_SIZE;	
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	int gid = get_global_id(0);		
	int tid = get_local_id(0);		

	// linearize gid
	int k = gid / fielddim_yz; 
	int j = (gid % (fielddim_yz  )) / fielddim_y;
	int i = gid - j * fielddim_y - k * fielddim_yz;
	
	double4 id4 = {k, j, i, 0};		
	//int rfid = k*fielddim_yz+j*fielddim_z+i;
	
	__private double rf_value = 0;
	for(int b=0; b<nr_lines; b+=block_size) { 
		int idx = b+tid;
		// prefetching vecs values to local memory
		loc_vecs[tid] = vecs[idx]; 
		barrier(CLK_LOCAL_MEM_FENCE);	

		for(int l=b, t=0; l<min(nr_lines,b+block_size); l++, t++)	
		{				
			//double linecoord = - dot(id4, vecs[l]);
			double linecoord = - dot(id4, loc_vecs[t]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
			rf_value += y[l*linelength+linepoint];					
		}
		barrier(CLK_LOCAL_MEM_FENCE); // need a synch before the next block
	}
	RF[gid] = rf_value;
}



__kernel void make_reg_field1_unroll(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor)
{
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	int gid = get_global_id(0);			

	// linearize gid
	int k = gid / fielddim_yz; 
	int j = (gid % (fielddim_yz  )) / fielddim_y;
	int i = gid - j * fielddim_y - k * fielddim_yz;	
	double4 id4 = {k, j, i, 0};	

	__private double rf_value = 0;
	for(int l=0; l<nr_lines; l+= LOOP_UNROLL) {
		double linecoord = - dot(id4, vecs[l]);
		size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
		rf_value += y[l*linelength+linepoint];				
		
#if LOOP_UNROLL > 1
		linecoord = - dot(id4, vecs[l+1]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+1)*linelength+linepoint];				
#endif
#if LOOP_UNROLL > 2
		linecoord = - dot(id4, vecs[l+2]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+2)*linelength+linepoint];		
		
		linecoord = - dot(id4, vecs[l+3]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+3)*linelength+linepoint];				
#endif
#if LOOP_UNROLL > 4
		linecoord = - dot(id4, vecs[l+4]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+4)*linelength+linepoint];		
		
		linecoord = - dot(id4, vecs[l+5]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+5)*linelength+linepoint];		

		linecoord = - dot(id4, vecs[l+6]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+6)*linelength+linepoint];				

		linecoord = - dot(id4, vecs[l+7]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+7)*linelength+linepoint];				
#endif		
	}
	RF[gid] = rf_value;
}


__kernel void make_reg_field1_blocking_unroll(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor)
{
	__local double4 loc_vecs[LOCAL_GROUP_SIZE];
	const int block_size = LOCAL_GROUP_SIZE;	
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	int gid = get_global_id(0);		
	int tid = get_local_id(0);		

	// linearize gid
	int k = gid / fielddim_yz; 
	int j = (gid % (fielddim_yz  )) / fielddim_y;
	int i = gid - j * fielddim_y - k * fielddim_yz;
	
	double4 id4 = {k, j, i, 0};		
	//int rfid = k*fielddim_yz+j*fielddim_z+i;
	
	__private double rf_value = 0;
	for(int b=0; b<nr_lines; b+=block_size) { 
		int idx = b+tid;

		// prefetching vecs values to local memory
		loc_vecs[tid] = vecs[idx]; 
		barrier(CLK_LOCAL_MEM_FENCE);	
		
		for(int l=b, t=0; l<min(nr_lines,b+block_size); l+=LOOP_UNROLL, t+=LOOP_UNROLL)	
		{				
			double linecoord = - dot(id4, loc_vecs[t]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
			double y_val = y[l*linelength+linepoint];		
			rf_value += y_val;		
			
#if LOOP_UNROLL > 1
			linecoord = - dot(id4, loc_vecs[t+1]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+1)*linelength+linepoint];					
#endif
#if LOOP_UNROLL > 2
			linecoord = - dot(id4, loc_vecs[t+2]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+2)*linelength+linepoint];		

			linecoord = - dot(id4, loc_vecs[t+3]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+3)*linelength+linepoint];
#endif
#if LOOP_UNROLL > 4
			linecoord = - dot(id4, loc_vecs[t+4]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+4)*linelength+linepoint];					

			linecoord = - dot(id4, loc_vecs[t+5]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+5)*linelength+linepoint];			

			linecoord = - dot(id4, loc_vecs[t+6]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+6)*linelength+linepoint];

			linecoord = - dot(id4, loc_vecs[t+7]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+7)*linelength+linepoint];					
#endif
			barrier(CLK_LOCAL_MEM_FENCE); // need a synch before the next block
		}
	}
	RF[gid] = rf_value;
}

////////////////////////// out of core

__kernel void make_reg_field_outofcore_blocking(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor, unsigned long offset
	)
{
	__local double4 loc_vecs[LOCAL_GROUP_SIZE];
	const int block_size = LOCAL_GROUP_SIZE;	
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	
	int gid = get_global_id(0);		
	int tid = get_local_id(0);		

#ifdef __CUDACC__
	unsigned long long f_yz = fielddim_yz;
	unsigned long long f_y  = fielddim_y;
	unsigned long long rid = get_global_id(0) + offset;		
	unsigned long long k = rid / f_yz; 
	unsigned long long j = (rid % f_yz) / f_y;
	unsigned long long i = rid - j * f_y - k * f_yz;
#else
	size_t rid = get_global_id(0) + offset;		
	int k = rid / fielddim_yz; 
	int j = (rid % (fielddim_yz  )) / fielddim_y;
	int i = rid - j * fielddim_y - k * fielddim_yz;
#endif
	
	double4 id4 = {k, j, i, 0};		
	//int rfid = k*fielddim_yz+j*fielddim_z+i;
	
	__private double rf_value = 0;
	for(int b=0; b<nr_lines; b+=block_size) { 
		int idx = b+tid;
		// prefetching vecs values to local memory
		loc_vecs[tid] = vecs[idx]; 
		barrier(CLK_LOCAL_MEM_FENCE);	

		for(int l=b, t=0; l<min(nr_lines,b+block_size); l++, t++)	
		{				
			//double linecoord = - dot(id4, vecs[l]);
			double linecoord = - dot(id4, loc_vecs[t]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
			rf_value += y[l*linelength+linepoint];					
		}
		barrier(CLK_LOCAL_MEM_FENCE); // need a synch before the next block
	}
	RF[gid] = rf_value;
}


///////////////////////////////////////////////////////////////////////// regular field computation, alternative parallelization

/* for each line kernel */
__kernel void make_reg_field2(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor)
{
	int l = get_global_id(0);		
	const int fielddim_yz  = fielddim_y*fielddim_z;
	//const int fielddim_xyz = fielddim_x*fielddim_yz;			
	
	for(int i=0; i<fielddim_x; i++){	
		for(int j=0;j<fielddim_y;j++){
			for(int k=0;k<fielddim_z;k++){
				double linecoord = - (i*vecs[l].x)-(j*vecs[l].y)-(k*vecs[l].z);
				long int linepoint = round(linecoord*resolutionfactor) + linelength*0.5+1;									
				double y_val = y[l*linelength+linepoint];
				long int  rf_index = i*fielddim_yz+j*fielddim_z+k;
				atomic_add_double( & RF[rf_index], y_val); 				
			}
		}
	}	
}

// for each (cell, line) -> this parallelization does not make sense at all!
__kernel void make_reg_field3(
	int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
	__global double4* vecs, __global double* y, __global double* RF, 
	double resolutionfactor)
{	
	const size_t fielddim_yz  = fielddim_y*fielddim_z;
	int hid = get_global_id(0);			
	int gid = hid / nr_lines;			
	int l =  hid % nr_lines;			

	// linearize gid
	int k = gid / fielddim_yz; 
	int j = (gid % (fielddim_yz  )) / fielddim_y;
	int i = gid - j * fielddim_y - k * fielddim_yz;
	
	double4 id4 = {i, j, k, 0};	

	double rf_value = 0;

		
	//double linecoord = - (i*vecs[l].x) - (j*vecs[l].y) - (k*vecs[l].z);
	double linecoord = - dot(id4, vecs[l]);
	size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
	double y_val = y[l*linelength+linepoint];		
	atomic_add_double( & RF[gid], y_val);				
	
	RF[gid] = rf_value;
}


///////////////////////////////////////////////////////////////////////// irregular field computation

/* standard version */
__kernel void make_irr_field
	(int nr_lines, int linelength,
	__global double4* vecs, __global double* y, __global double4* RF, double resolutionfactor)
{		
	int gid = get_global_id(0);		
	double4 id4 = RF[gid];
	id4.w = 0;

	__private double rf_value = 0;
	
	for(int l=0; l<nr_lines; l++) { 
		double linecoord = - dot(id4, vecs[l]);			
		size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
		double y_val = y[l*linelength+linepoint];		
		rf_value += y_val;				
	}
	RF[gid].w = rf_value;
}

/* optimized with blocking */
__kernel void make_irr_field_blocking
	(int nr_lines, int linelength,
	__global double4* vecs, __global double* y, __global double4* RF, double resolutionfactor)
{		
	__local double4 loc_vecs[LOCAL_GROUP_SIZE];
	const int block_size = LOCAL_GROUP_SIZE;	

	int gid = get_global_id(0);		
	int tid = get_local_id(0);		
	double4 id4 = RF[gid];
	id4.w = 0;

	__private double rf_value = 0;
	
	for(int b=0; b<nr_lines; b+=block_size) { 
		int idx = b+tid;
		// prefetching vecs values to local memory
		loc_vecs[tid] = vecs[idx]; 
		barrier(CLK_LOCAL_MEM_FENCE);	

		for(int l=b, t=0; l<min(nr_lines,b+block_size); l++, t++){	
			double linecoord = - dot(id4, loc_vecs[t]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
			double y_val = y[l*linelength+linepoint];		
			rf_value += y_val;				
		}

		barrier(CLK_LOCAL_MEM_FENCE); // need a synch before the next block
	}
	RF[gid].w = rf_value;
}

/* optimized with loop unroll */
__kernel void make_irr_field_unroll
	(int nr_lines, int linelength,
	__global double4* vecs, __global double* y, __global double4* RF, double resolutionfactor)
{		
	int gid = get_global_id(0);			
	double4 id4 = RF[gid];
	id4.w = 0;

	__private double rf_value = 0;	
	for(int l=0; l<nr_lines; l+=LOOP_UNROLL)	
	{
		double linecoord = - dot(id4, vecs[l]);
		size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
		rf_value += y[l*linelength+linepoint];				
		
#if LOOP_UNROLL > 1
		linecoord = - dot(id4, vecs[l+1]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+1)*linelength+linepoint];				
#endif
#if LOOP_UNROLL > 2
		linecoord = - dot(id4, vecs[l+2]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+2)*linelength+linepoint];				

		linecoord = - dot(id4, vecs[l+3]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+3)*linelength+linepoint];				
#endif
#if LOOP_UNROLL > 4
		linecoord = - dot(id4, vecs[l+4]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+4)*linelength+linepoint];		

		linecoord = - dot(id4, vecs[l+5]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+5)*linelength+linepoint];		

		linecoord = - dot(id4, vecs[l+6]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+6)*linelength+linepoint];		

		linecoord = - dot(id4, vecs[l+7]);
		linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
		rf_value += y[(l+7)*linelength+linepoint];				
#endif		
	}
	RF[gid].w = rf_value;
}

/* optimized with blocking and loop unroll */
__kernel void make_irr_field_blocking_unroll (int nr_lines, int linelength,
	__global double4* vecs, __global double* y, __global double4* RF, double resolutionfactor)
{		
	__local double4 loc_vecs[LOCAL_GROUP_SIZE];
	const int block_size = LOCAL_GROUP_SIZE;	

	int gid = get_global_id(0);		
	int tid = get_local_id(0);		
	double4 id4 = RF[gid];
	id4.w = 0;

	__private double rf_value = 0;	
	for(int b=0; b<nr_lines; b+=block_size) { 
		int idx = b+tid;
		// prefetching vecs values to local memory
		loc_vecs[tid] = vecs[idx]; 
		barrier(CLK_LOCAL_MEM_FENCE);	

		for(int l=b, t=0; l<min(nr_lines,b+block_size); l+=LOOP_UNROLL, t+=LOOP_UNROLL)	
		{
			double linecoord = - dot(id4, loc_vecs[t]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;									
			rf_value += y[l*linelength+linepoint];					 		
			
#if LOOP_UNROLL > 1
			linecoord = - dot(id4, loc_vecs[t+1]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+1)*linelength+linepoint];					
#endif
#if LOOP_UNROLL > 2
			linecoord = - dot(id4, loc_vecs[t+2]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+2)*linelength+linepoint];		

			linecoord = - dot(id4, loc_vecs[t+3]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+3)*linelength+linepoint];			
#endif
#if LOOP_UNROLL > 4
			linecoord = - dot(id4, loc_vecs[t+4]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+4)*linelength+linepoint];				

			linecoord = - dot(id4, loc_vecs[t+5]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+5)*linelength+linepoint];

			linecoord = - dot(id4, loc_vecs[t+6]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+6)*linelength+linepoint];

			linecoord = - dot(id4, loc_vecs[t+7]);
			linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;												
			rf_value += y[(l+7)*linelength+linepoint];	
#endif

			barrier(CLK_LOCAL_MEM_FENCE); // need a synch before the next block
		}
	}
	RF[gid].w = rf_value;
}


///////////////////////////////////////////////////////////////////////// irregular field computation

/*
#define PI M_PI
//#define M_PI 3.14159265358979323846

void Rotx(double alpha, cl_double4* vecs,int nr_lines) {	
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);
//	for(int i=0;i<nr_lines;i++)
	{
		double temp=cos_alpha*vecs[i].s[1]+vecs[i].s[2]*sin_alpha;
		vecs[i].s[2]=vecs[i].s[2]*cos_alpha-sin_alpha*vecs[i].s[1];
		vecs[i].s[1]=temp;
	}
}

void Roty(double alpha, cl_double4* vecs,int nr_lines) {
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);
//	for(int i=0;i<nr_lines;i++)
	{
		double temp = cos_alpha*vecs[i].s[0]-vecs[i].s[2]*sin_alpha;
		vecs[i].s[2] = vecs[i].s[2]*cos_alpha+sin_alpha*vecs[i].s[0];
		vecs[i].s[0] = temp;
	}

}

void Rotz(double alpha, cl_double4* vecs,int nr_lines) {
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);
//	for(int i=0;i<nr_lines;i++)
	{
		double temp=cos_alpha*vecs[i].s[0]+vecs[i].s[1]*sin_alpha;
		vecs[i].s[1] = vecs[i].s[1]*cos_alpha-sin_alpha*vecs[i].s[0];
		vecs[i].s[0] = temp;
	}
}


// Create a direction vector. Not really so parallel (we have with lines), but requried to run everything on the device
__kernel void vec_gen(const double *h_seq, double4* vecs)
{
	int i = get_global_id(0);
	int nr_lines = get_global_size(0);
	
	int perm;
	double alpha;
		
	double t = 2 * h_seq[nr_lines+i]-1;
	double phi = 2.0 * PI * h_seq[i];
	vecs[i].x = sqrt(1.0-t*t) * cos(phi);	//x-coordinate
	vecs[i].y = sqrt(1.0-t*t) * sin(phi);	//y-coordinate
	vecs[i].z = t;							//z_coordinate
	vecs[i].w = 0;	
//------------------------------------------------------
	double temp = rand_01();		// one of 5 permutations
	temp = temp*6.0;	
	perm = (int) ceil(temp);	
	switch ( perm ) {	
	case 1:
		alpha = rand_01();
		alpha = alpha*2*PI;
		Rotx(alpha,vecs,nr_lines);
		alpha = rand_01();
		alpha = alpha*2*PI;
		Roty(alpha,vecs,nr_lines);
		alpha = rand_01();
		alpha = alpha*2*PI;
		Rotz(alpha,vecs,nr_lines);
		break;

	case 2:
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		break;

	case 3:
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		break;

	case 4:
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		break;

	case 5:
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		break;

	case 6:
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		break;

	default:
		alpha = rand_01();
		Rotx(alpha,vecs,nr_lines);
		alpha = rand_01();
		Roty(alpha,vecs,nr_lines);
		alpha = rand_01();
		Rotz(alpha,vecs,nr_lines);
		break;
	}

}

*/