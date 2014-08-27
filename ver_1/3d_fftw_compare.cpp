#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <chrono>


#define _USE_MATH_DEFINES 
#include <cmath>
#include <math.h>
#define PI M_PI
#define SEED 42

/* Random number generation */
#if 0
// C approach
inline void init_rand(){ srand(SEED); }
inline double rand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
#else
#include <random>
// C++ TR1 approach
static std::default_random_engine re;
inline void init_rand(){srand(SEED);}
inline double rand(double min, double max){
	std::uniform_real_distribution<double> unif(min, max);
	return unif(re);
}
#endif
inline double rand_01(){
	return rand(0.0, 1.0);
}


using namespace std;

inline void print_double(const double *A, size_t size){ 
	cout << "(";
	for(int i=0; i<size; i++) {
		cout << A[i];
		if(i<size-1)
			cout << ",";
	}
	cout << ")" << endl;	
}



//const char *test_neb[] = {"X", "0", "1024", "2600",		"256", "256", "256",			"1", "1"};
const char *test_fftw[] = {"X", "-3", "128", "1"};
#define argv test_fftw

void test_3d_fftw(int size);

int main(int argc, char *_argv[]){
//	for(int size = 64; size <= 320; size+=64){
//	for(int size = 384; size <= 512; size+=64){
	for(int size = 64; size <= 704; size+=64){	
		test_3d_fftw(size);
	}
	printf("press a key to continue\n");
	getchar();
}


void test_3d_fftw(int _size){
	std::chrono::time_point<std::chrono::high_resolution_clock> time1, time2;
	time1 = std::chrono::system_clock::now();


	double PLindex,abs_k,A,phase;
	
	int nr_3Dfields, fielddim, nr_points, index, mod_fielddim;	
	int neg_k, neg_i, neg_j;
	fftw_complex *array_in;	
	fftw_plan back_plan;	

	PLindex = atof(argv[1]);	// read input
	fielddim = _size; //atoi(argv[2]);
	nr_3Dfields = atoi(argv[3]);
	mod_fielddim = fielddim+1;

	printf("\n\nmod_field dim %i\n", mod_fielddim);
	printf("filed #%i\n", nr_3Dfields);

	for(int m=0;m<nr_3Dfields;m++)
	{
		nr_points = (fielddim+1)*(fielddim+1)*(fielddim+1);
		array_in = (fftw_complex*) fftw_malloc(nr_points * sizeof(fftw_complex));

		back_plan = fftw_plan_dft_3d(mod_fielddim, mod_fielddim, mod_fielddim,array_in, array_in, FFTW_BACKWARD , FFTW_MEASURE);


		for(int i=-fielddim/2;i<=fielddim/2;i++){	//fill the data that goes into the FFT 
			for(int j=-fielddim/2;j<=fielddim/2;j++){
				for(int k=0;k<=fielddim/2;k++){
					//positive k parts 
					abs_k=sqrt(i*i+j*j+k*k);
					A=1000.0*pow(abs_k,(PLindex*0.5));
					phase=(2.0*rand_01()-1.0)*PI;
					index=(i+fielddim/2)*mod_fielddim*mod_fielddim+(j+fielddim/2)*mod_fielddim+k+fielddim/2;
					
					// array_in[index]=A*cos(phase)+I*A*sin(phase);
					array_in[index][0] = A*cos(phase); 
					array_in[index][1] = A*sin(phase);

					//negative k parts
					//phase=-1.0*phase;
					neg_i=-1*i;
					neg_j=-1*j;
					neg_k=-1*k;
					index=(neg_i+fielddim/2)*mod_fielddim*mod_fielddim+(neg_j+fielddim/2)*mod_fielddim+neg_k+fielddim/2;

					array_in[index][0] = A*cos(phase);
					array_in[index][1] = A*sin(-1.0*phase);
				}
			}
		}

		array_in[(fielddim/2)*mod_fielddim*mod_fielddim+(fielddim/2)*mod_fielddim+fielddim/2][0] = 1.0;
		array_in[(fielddim/2)*mod_fielddim*mod_fielddim+(fielddim/2)*mod_fielddim+fielddim/2][1] = 0;

		//Rearange array for use in FFTW library 
		for(int j=0; j<=nr_points/2; j++)	{
			fftw_complex temp = { array_in[j][0], array_in[j][1]};			
			array_in[j][0] = array_in[j+nr_points/2][0];
			array_in[j][1] = array_in[j+nr_points/2][1];
			array_in[j+nr_points/2][0] = temp[0];
			array_in[j+nr_points/2][1] = temp[1];
		}

		fftw_execute(back_plan);

		//for(i=0;i<nr_points;i++) printf("%g + i*%g\n",creal(array_in[i]),cimag(array_in[i]));	
		/*
		for(int i=0; i<mod_fielddim; i++){
			for(int j=0; j<mod_fielddim; j++){
				for(int k=0; k<mod_fielddim; k++){
					printf("%.15lf\n", array_in[i*mod_fielddim*mod_fielddim+j*mod_fielddim+k][0] );
				}
			}
		}
		*/

		time2 = std::chrono::system_clock::now();

		auto total = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();

		cout << endl << 
			"--- 3D FFTW make_field ---" << endl
			<< "total time             " << total << " ms" << endl
			<< "---" << endl;		

		// print some elemntes for debug
		double *print_array = (double*) array_in;
		print_double(print_array, 100); // first 100 elements
		print_double(print_array+(2*nr_points-100), 100); // last 100 elements

		fftw_free(array_in);	
		fftw_destroy_plan(back_plan);	
	}	

	
}
