#include "fft_field_line.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "3DPlc.h"




const char *test_fftw[] = {"X", "-3", "128", "1"};
#define argv test_fftw

#ifdef CUDA
#include <cufft.h>
#include <cuda_runtime.h>
///#include "helper_cuda.h"
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if(result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d \n", file, line, static_cast<unsigned int>(result));		
		fprintf(stderr, "%s\n", cudaGetErrorString(result));
		//DEVICE_RESET
		// Make sure we call CUDA Device Reset before exiting
		getchar();
		exit(EXIT_FAILURE);		
	}
}
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#endif


//#define fftw_complex cufftDoubleComplex
void warmup(){
	make_field_3dfft(64);
}

void make_field_3dfft_batch()
{
//	for(int size = 64; size<=512; size+=64)		
//	for(int size = 64; size<=1024; size+=64)
	for(int size = 384; size<=1024; size+=64)
		make_field_3dfft(size);
/*
	make_field_3dfft(64);
	make_field_3dfft(96);
	make_field_3dfft(128);
	make_field_3dfft(160);
	make_field_3dfft(192);
	make_field_3dfft(224);
	make_field_3dfft(256);
*/
}

/* 3D FFT for field generation */
void make_field_3dfft(int _fielddim)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> time1, time2, time3;
	time1 = std::chrono::system_clock::now();

	double PLindex;	
	//	int nr_3Dfields, fielddim, index, mod_fielddim;	
	int neg_k,neg_i,neg_j;

	cufftDoubleComplex *host_array;
	cufftDoubleComplex *device_array;
	cufftHandle back_plan;	

	PLindex = atof(test_fftw[1]);	
	int fielddim = _fielddim;//atoi(test_fftw[2]);
	int nr_3Dfields = atoi(test_fftw[3]);
	int mod_fielddim = fielddim+1;

	size_t nr_points = (fielddim+1)*(fielddim+1)*(fielddim+1);
	size_t num_bytes = nr_points * sizeof(cufftDoubleComplex);

	// mem allocation
	host_array = (cufftDoubleComplex*) malloc(num_bytes);
	//array_in = (fftw_complex*) fftw_malloc(nr_points * sizeof(fftw_complex));
	checkCudaErrors(cudaMalloc((void **)&device_array, num_bytes));

	//back_plan = fftw_plan_dft_3d(mod_fielddim, mod_fielddim, mod_fielddim,array_in, array_in, FFTW_BACKWARD , FFTW_MEASURE);
	cufftPlan3d(&back_plan, mod_fielddim, mod_fielddim, mod_fielddim, CUFFT_Z2Z);
	//array_in, array_in, FFTW_BACKWARD , FFTW_MEASURE);

	cout << endl <<"fiel_dim " << mod_fielddim << endl;
	
	for(int m=0; m<nr_3Dfields; m++){	
		cout << endl <<"field #" << m << endl;		

		for(int i=-fielddim/2;i<=fielddim/2;i++){	//fill the data that goes into the FFT 
			for(int j=-fielddim/2;j<=fielddim/2;j++){
				for(int k=0;k<=fielddim/2;k++){
					//positive k parts 
					double abs_k = sqrt((double)(i*i+j*j+k*k));
					double A = 1000.0*pow(abs_k,(PLindex*0.5));
					double phase=(2.0*rand_01()-1.0)*PI;
					int index = (i+fielddim/2)*mod_fielddim*mod_fielddim+(j+fielddim/2)*mod_fielddim+k+fielddim/2;

					//array_in[index]=A*cos(phase)+I*A*sin(phase);
					host_array[index].x = A*cos(phase);
					host_array[index].y = A*sin(phase);

					//negative k parts
					//phase=-1.0*phase;
					neg_i =- 1*i;
					neg_j =- 1*j;
					neg_k =- 1*k;
					index = (neg_i+fielddim/2)*mod_fielddim*mod_fielddim+(neg_j+fielddim/2)*mod_fielddim+neg_k+fielddim/2;

					//array_in[index]=A*cos(phase)+I*A*sin(-1.0*phase);
					host_array[index].x = A*cos(phase);
					host_array[index].y = A*sin(-1.0*phase);
				}
			}
		}

		host_array[(fielddim/2)*mod_fielddim*mod_fielddim+(fielddim/2)*mod_fielddim+fielddim/2].x = 1.0;
		host_array[(fielddim/2)*mod_fielddim*mod_fielddim+(fielddim/2)*mod_fielddim+fielddim/2].y = 0.0;

		//Rearange array for use in FFTW library 
		for(int j=0;j<=nr_points/2;j++){
			cufftDoubleComplex temp = host_array[j];
			host_array[j] = host_array[j+nr_points/2];
			host_array[j+nr_points/2] = temp;
		}

		time2 = std::chrono::system_clock::now();

		// move data to the GPU
		cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyDeviceToHost);

		///	fftw_execute(back_plan);
		cufftExecZ2Z(back_plan, device_array, device_array, FFTW_BACKWARD);

		// move back
		cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyHostToDevice);

		time3 = std::chrono::system_clock::now();

		auto s1 = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
		auto s2 = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
		auto time_total = s1 + s2;	

		cout << endl << 
			"--- 3D FFT make_field ---" << endl
			<< "1.        " << s1 << " ms" << endl
			<< "2.        " << s2 << " ms" << endl				
			<< "---" << endl
			<< "total time             " << time_total << " ms" << endl
			<< "---" << endl;


		//for(i=0;i<nr_points;i++) printf("%g + i*%g\n",creal(array_in[i]),cimag(array_in[i]));	

		/*		for(int i=0;i<mod_fielddim;i++){
		for(int j=0;j<mod_fielddim;j++){
		for(int k=0;k<mod_fielddim;k++){
		printf("%.15lf\n", host_array[i*mod_fielddim*mod_fielddim+j*mod_fielddim+k].x);
		}
		}
		}
		*/

	}

	// print some elemntes for debug
	double *print_array = (double*) host_array;
	print_double(print_array, 100); // first 100 elements
	print_double(print_array+(2*nr_points-100), 100); // last 100 elements


	/*
	fftw_free(array_in);	
	fftw_destroy_plan(back_plan);	
	*/
	cufftDestroy(back_plan);
	cudaFree(device_array);
	free(host_array);
}




/* 1D FFT for line generation */

void generate_PL_lines_fft(double PLindex, int nr_lines,int linelength, 
						   icl_buffer *y_buffer, //double *y, // out
						   int fielddim_z) 
{

	cufftHandle fftw_plan_transfer_fw, fftw_plan_noise_fw, fftw_plan_transfer_bw;

	// FFT preparation 
	const size_t out_size		 = sizeof(cufftDoubleComplex)*(linelength/2+1);
	const size_t in_size		 = sizeof(double)*(linelength+1);
	
	// memory allocation
	double *noise_in_host		 = (double*) malloc(in_size);  // all these allocs can technically also be done outside of the function, if performance is bad this might be a consideration	
	double *noise_in_device;
	checkCudaErrors(cudaMalloc((void **)&noise_in_device, in_size));

	cufftDoubleComplex *noise_out_host = (cufftDoubleComplex*) fftw_malloc(out_size); // check if i need 2 times +1
	cufftDoubleComplex *noise_out_device; 
	checkCudaErrors(cudaMalloc((void **)&noise_out_device, out_size));

	double *transfer_in_host 	 = (double*) malloc(in_size);  
	double *transfer_in_device;
	checkCudaErrors(cudaMalloc((void **)&transfer_in_device, in_size));
	
	cufftDoubleComplex *transfer_out_host = (cufftDoubleComplex*) fftw_malloc(out_size);	
	cufftDoubleComplex *transfer_out_device;
	checkCudaErrors(cudaMalloc((void **)&transfer_out_device, out_size));

	// plans
	cufftPlan1d(&fftw_plan_transfer_fw,	linelength, CUFFT_D2Z, 1);
	cufftPlan1d(&fftw_plan_noise_fw,	linelength, CUFFT_D2Z, 1);
	cufftPlan1d(&fftw_plan_transfer_bw,	linelength, CUFFT_Z2D, 1);	

	// XXX todo here we can potentially cufftPlanMany, to performa a batch of many plan1d fft
/*
	fftw_plan_transfer_fw = fftw_plan_dft_r2c_1d(linelength, transfer_in, transfer_out,FFTW_MEASURE); 
	fftw_plan_noise_fw    = fftw_plan_dft_r2c_1d(linelength, noise_in, noise_out,FFTW_MEASURE);     
	fftw_plan_transfer_bw = fftw_plan_dft_c2r_1d(linelength, transfer_out, transfer_in,FFTW_MEASURE);	
*/

	for(int v=0; v<nr_lines; v++){

		for(int i=0; i<linelength; i++){
			noise_in_host[i] = rand_01();
			transfer_in_host[i] = rand_01();
		}	

		Box_Mueller(linelength, noise_in_host, transfer_in_host);

		for(int i=0; i<linelength; i++){	
			noise_in_host[i] = 2*noise_in_host[i]-1; // around 0
			noise_in_host[i] = 5*noise_in_host[i];  //changes the deviation, which values need to be put will be investigated			
		}

		transfer_in_host[0]=1.0;
		for(int i=1; i<linelength; i++){
			transfer_in_host[i] = (transfer_in_host[i-1]/(i))*(i-1-(PLindex/2.0));
		}		

		/// (a) moving transfer_in and noise_in to the device
		cudaMemcpy(noise_in_device, noise_in_host, in_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(transfer_in_device, transfer_in_host, in_size, cudaMemcpyDeviceToHost);
		
		//fftw_execute(fftw_plan_noise_fw);	
		//fftw_execute(fftw_plan_transfer_fw);
		cufftExecD2Z(fftw_plan_noise_fw, transfer_in_device, transfer_out_device);
		cufftExecD2Z(fftw_plan_noise_fw, noise_in_device, noise_out_device);

		/// (b) moving back transfer_out and noise out
		cudaMemcpy(transfer_out_host, transfer_out_device, out_size, cudaMemcpyHostToDevice);
		cudaMemcpy(noise_out_host, noise_out_device, out_size, cudaMemcpyHostToDevice);

		for(int i=0; i<0.5*linelength+1; i++){
			double temp = (transfer_out_host[i].x*noise_out_host[i].x+transfer_out_host[i].y*noise_out_host[i].y) / linelength;
			transfer_out_host[i].y = (transfer_out_host[i].x*noise_out_host[i].y-transfer_out_host[i].y*noise_out_host[i].x) / linelength;
			transfer_out_host[i].x = temp;
		}		

		/// (c) moving to the device transfer_out
		cudaMemcpy(transfer_out_device, transfer_out_host, out_size, cudaMemcpyDeviceToHost);

		// fftw_execute(fftw_plan_transfer_bw);	
		cufftExecZ2D(fftw_plan_transfer_bw, transfer_out_device, transfer_in_device);

		//// (d) moving back transfer_in 
		cudaMemcpy(transfer_in_host, transfer_in_device, in_size, cudaMemcpyHostToDevice);

		for(int i=0; i<linelength; i++){
			transfer_in_host[i] = transfer_in_host[i]/sqrt((double) linelength);
		}

		// xxx reduce max/avg
		double average=0;
		for(int i=0; i<linelength; i++){
			average = average + transfer_in_host[i];
		}
		average = average/linelength;


		double *y_host = (double*)malloc(sizeof(cl_double) * linelength * nr_lines + 1);
		size_t index = 0;
		for(int i=0; i<linelength; i++){
			y_host[index]=(transfer_in_host[i]-average);
			index++;
		}

		/// (e) write y_buf
		//icl_local_device* ldev = &local_devices[y_buffer->device->device_id];
		icl_local_buffer* lbuf = (icl_local_buffer*)(y_buffer->buffer_add);	
		cudaMemcpy((void*)lbuf->mem, transfer_out_host, out_size, cudaMemcpyDeviceToHost);
	}
	
	free(transfer_in_host);
	free(transfer_out_host);
	free(noise_out_host);
	free(noise_in_host);

	cudaFree(transfer_in_device);
	cudaFree(transfer_out_device);
	cudaFree(noise_out_device);
	cudaFree(noise_in_device);

	cufftDestroy(fftw_plan_transfer_fw);	
	cufftDestroy(fftw_plan_transfer_bw);
	cufftDestroy(fftw_plan_noise_fw);
//	fftw_cleanup();
}

void generate_PL_lines_fft_many(double PLindex, int nr_lines,int linelength, icl_buffer *y_buffer, int fielddim_z)
{}