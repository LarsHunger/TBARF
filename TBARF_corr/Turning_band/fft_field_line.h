#pragma once

#ifdef CUDA
	#include "lib_icl_cuda.h"
#else
	#include "lib_icl.h"
#endif

void warmup();

void make_field_3dfft_batch();
void make_field_3dfft(int);
//void generate_PL_lines_fft(double PLindex, int nr_lines, int linelength, double *y, int fielddim_z);
void generate_PL_lines_fft(double PLindex, int nr_lines,int linelength, icl_buffer *y_buffer, int fielddim_z);
void generate_PL_lines_fft_many(double PLindex, int nr_lines,int linelength, icl_buffer *y_buffer, int fielddim_z);