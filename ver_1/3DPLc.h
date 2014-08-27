#pragma once

// for cl_double4
#ifdef CUDA
	#include "lib_icl_cuda.h"
#else
	#include "lib_icl.h"
#endif

#include <fftw3.h> // for complex

#define _USE_MATH_DEFINES 
#include <cmath>
#include <math.h>
#define PI M_PI

#include <stdlib.h> // RAND_MAX
#include <iostream>
using namespace std;


// to assure deterministic behaviour, we set the seed to a constant value 
#define SEED 42

/* Transforms equally distributed random numbers into normally distributed ones*/
void Box_Mueller(int linelength,double *a,double *b);

/* Generate a Halton sequence */
void Halton_seq(int nr_lines, double* h_seq);

/* Line generation with FFT */
void generate_PL_lines(double PLindex,int nr_lines,int linelength, double *y,int fielddim_z);

/* Creates direction vectors for the lines and rotates the whole sphere by the same random angles */
void vec_gen(int nr_lines, const double *h_seq, double* vecs);	
void vec_gen(int nr_lines, const double *h_seq, cl_double4* vecs);

void make_field_scalar(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const double* vecs, const double* y, double* RF, double resolutionfactor);
void make_field(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);
void make_field_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4 *vecs, const double *y,  cl_double4 *RF, double resolutionfactor);
void make_field_irregular_scalar(int nr_lines, int field_nr_points, int linelength, const double *vecs, const double *y, double **RF,  double resolutionfactor);

void ocl_make_field_outcore(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);


void Rotx(double alpha,double* vecs,int nr_lines);
void Roty(double alpha,double* vecs,int nr_lines);
void Rotz(double alpha,double* vecs,int nr_lines);
// follow overloaded vectorial versions
void Rotx(double alpha,cl_double4* vecs,int nr_lines);
void Roty(double alpha,cl_double4* vecs,int nr_lines);
void Rotz(double alpha,cl_double4* vecs,int nr_lines);



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


template <typename T> T round_even(T x, T precision) {
  T result, integral, fractional;
  // parity_flag - false : true :: odd : even
  // sign_flag - false : true :: negative : positive
  // round_flag - false : true :: down : up
  bool parity_flag, sign_flag, round_flag = false, roundeven_flag = false;  
  result = x / precision;
  fractional = modf(result, &integral); 
  integral = fabs(integral);
  parity_flag = fmod(integral, 2) == 1 ? false : true;
  sign_flag = x < 0 ? false : true; 
  fractional = fabs(fractional);
  fractional *= 10;
  fractional -= 5;
  
  if (fractional >= 1)		round_flag = true;
  else if (fractional < 0)  round_flag = false;
  else						roundeven_flag = true;
    
  if (roundeven_flag)
	  if (!parity_flag)     round_flag = true;
  
  if (!sign_flag)		    round_flag ^= true;
    
  if (round_flag)		    result = ceil(result);
  else						result = floor(result);
    
  result *= precision;  
  return result;
}

inline double round(double n){
	return round_even<double>(n, 1);
}

//#define app_round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

//#define round(dbl) dbl >= 0.0 ? (int)(dbl + 0.5) : ((dbl - (double)(int)dbl) <= -0.5 ? (int)dbl : (int)(dbl - 0.5))
//#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))

/*
inline double round(double number)
{ return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5); }
inline float round(float number)
{ return number < 0.0 ? ceilf(number - 0.5f) : floorf(number + 0.5f); }
*/
/*
inline double round(double n)//round up a float type and show one decimal place
{
	float t;
    t=n-floor(n);
    if (t>=0.5) {
		n*=10;//where n is the multi-decimal float
        ceil(n);
        n/=10;
	} else {
		n*=10;//where n is the multi-decimal float
		floor(n);
		n/=10;
	}
    return n;
}          
*/


inline double vdc(int n, int base)
{
	int num,dem;
	int p = 0, q = 1;

	while (n) {
		p = p * base + (n % base);
		q *= base;
		n /= base;
	}

	num = p;  
	dem = q;

	while (p) { n = p; p = q % p; q = n; }
	num /= q;
	dem /= q;
	return (double)num/dem;
}

inline
void print_regular(const double *RF, int fielddim_x, int fielddim_y, int fielddim_z){
	for(int i=0; i<fielddim_x; i++)
		for(int j=0; j<fielddim_y; j++)
			for(int k=0; k<fielddim_z; k++)
				cout << RF[i*fielddim_y*fielddim_z+j*fielddim_z+k] << "  ";
}

inline
void print_irregular(const double **RF_irregular, int fielddim_x, int fielddim_y, int fielddim_z){
	for(int i=0; i<fielddim_x; i++)
		for(int j=0; j<fielddim_y; j++)
			for(int k=0; k<fielddim_z; k++)
				//	printf("%.15lf	%.15f	%.15f	%.15f\n",RF_irregular[0][i*fielddim_y*fielddim_z+j*fielddim_z+k],RF_irregular[1][i*fielddim_y*fielddim_z+j*fielddim_z+k],RF_irregular[2][i*fielddim_y*fielddim_z+j*fielddim_z+k],RF_irregular[3][i*fielddim_y*fielddim_z+j*fielddim_z+k]);
				cout << RF_irregular[3][i*fielddim_y*fielddim_z+j*fielddim_z+k] << "  ";
}

inline void print_double(const double *A, size_t size){ 
	cout << "(";
	for(int i=0; i<size; i++) {
		cout << A[i];
		if(i<size-1)
			cout << ",";
	}
	cout << ")" << endl;	
}

inline void print_double(const cl_double4 *A, size_t size){ 
	cout << "(";
	for(int i=0; i<size; i++) {
		cout << A[i].s[3]; 
		if(i<size-1)
			cout << ",";
	}
	cout << ")" << endl;	
}

inline void print_float(const float *A, size_t size){ 
	cout << "(";
	for(int i=0; i<size; i++) {
		cout << A[i];
		if(i<size-1)
			cout << ",";
	}
	cout << ")" << endl;
}

inline void check_nan(const double *A, size_t size){
	for(int i=0; i<size; i++) {
		if(_isnan(A[i]))
			cout << "NAN at position " << i << endl;
	}
}

inline void check_nan(const fftw_complex *A, size_t size){
	for(int i=0; i<size; i++) {
		if(_isnan(A[i][0]))
			cout << "NAN at position " << i << endl;
		if(_isnan(A[i][1]))
			cout << "NAN at position " << i << endl;
	}

}