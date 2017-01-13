#pragma once
#include <stdio.h>
#ifdef CUDA
	#include "lib_icl_cuda.h"
#else
	#include "lib_icl.h"
#endif


/* OpenCL implementations of the make field code for both regular and irregular fields */
void ocl_init(int device, int nr_lines, size_t field_nr_points, int linelength, int fielddim_x, int fielddim_y, int fielddim_z, bool regular, bool outofcore);
void ocl_finalize();

void ocl_make_field1 (int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);
void ocl_make_field1b(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);
void ocl_make_field1u(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);
void ocl_make_field1bu(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);

void ocl_make_field2 (int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);
void ocl_make_field3 (int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength, const cl_double4* vecs, const double* y, double* RF, double resolutionfactor);

void ocl_make_field_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor);
void ocl_make_fieldb_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor);
void ocl_make_fieldu_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor);
void ocl_make_fieldbu_irregular(int nr_lines, int field_nr_points, int linelength, const cl_double4* vecs, const double* y, cl_double4* RF, double resolutionfactor);