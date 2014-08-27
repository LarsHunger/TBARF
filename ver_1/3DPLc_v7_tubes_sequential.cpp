#include <stdlib.h>
#include <iostream>
#include <chrono>

#include <float.h> // isnan

#include "3DPLc.h"
using namespace std;

//nr_lines around 1000	
//fieldim_x,y,z should be between 128 and 1024 with more memory also 2048 or 4096
//linepoint=NINT(resolutionfactor*linecoord)+n*0.5+1
const char *test_1[] = {"X", "0", "1000", "200", "128", "128", "128", "1", "1"};
const char *test_2[] = {"X", "0", "1024", "256", "64", "64", "64", "1", "1"};
const char *test_3[] = {"X", "0", "1000", "2000", "512", "512", "512", "1", "1"};
const char *test_4[] = {"X", "0", "10", "20", "4", "4", "4", "1", "1"};
#define argv test_2



int main(int argc, char *_argv_[])
{
/*	if(argc == 0)
		char &*argv[] = _argv_;
	else	
		char &*argv[] = test_1;
*/
	double PLindex, resolutionfactor;
	int	nr_lines, linelength, fielddim_x, fielddim_y, fielddim_z, nr_3Dfields;
	double *RF, **RF_irregular;	 
	bool grid_regular = true;

	// read input 
	PLindex     = atof(argv[1]);	
	nr_lines    = atoi(argv[2]);
	linelength  = atoi(argv[3]);
	fielddim_x  = atoi(argv[4]);
	fielddim_y  = atoi(argv[5]);
	fielddim_z  = atoi(argv[6]);
	nr_3Dfields  = atoi(argv[7]);
	resolutionfactor = atof(argv[8]);	

	linelength = resolutionfactor*2*linelength; //scale line to the right size and double in length for invers directions  
	long int field_nr_points = fielddim_x*fielddim_y*fielddim_z;

	// memory allocation XXX to use std::vector instead
	double *y     = (double*) malloc (sizeof(double)*(linelength*nr_lines+1));	
	double *h_seq = (double*) malloc(sizeof(double)*2*nr_lines); 
	double *vecs  = (double*) malloc(sizeof(double)*3*nr_lines);


	if (grid_regular) {
		// Allocates RF as 1 block array with the values but not the coordinates saved if grid==regular
		RF = (double*)malloc(sizeof(double)*(field_nr_points)); 
	}
	else {
		
		// Allocates RF as 4 block array where index 0,1,2 are x,y,z and 3 is the field value usefull for extention to irregular grids
		RF_irregular = (double**)malloc(sizeof(double*)*4); 
		for (int i=0; i<4; ++i){
			RF_irregular[i]=(double*)malloc(sizeof(double)*(field_nr_points));
		}

		int index = 0;
		for (int i=0; i<fielddim_x; i++){	
			for (int j=0; j<fielddim_y; j++){	
				for (int k=0; k<fielddim_z; k++){
					RF_irregular[0][index] = i;
					RF_irregular[1][index] = j;
					RF_irregular[2][index] = k;
					index++;
				}
			}	
		}
	} 	

	
	init_rand();
	cout << "field dimension " << fielddim_x << "," << fielddim_y << "," << fielddim_z << " - 3d fields " << nr_3Dfields << endl;
	
	
	std::chrono::time_point<std::chrono::high_resolution_clock> time1, time2, time3, time4, time5, time6;
	time1 = chrono::high_resolution_clock::now();

	// halton sequence in h_seq, where the first nr_lines elements are the base 2 VDc sequence 
	// and the last nr_lines elements are the base 3 Vdc sequence, Halton sequence is the same 
	// for all iterations so it is done outside of the loop
	Halton_seq(nr_lines, h_seq); 
//	cout << endl << "Halton " << nr_lines; print_double(h_seq, std::min(nr_lines, 20));
	time2 = chrono::high_resolution_clock::now();

	// for each field
	for(int v=0; v<nr_3Dfields; v++) 
	{
		// fft here - writes all the RF lines in y
		time3 = chrono::high_resolution_clock::now();	
		generate_PL_lines(PLindex, nr_lines, linelength, y, fielddim_z); 

		time4 = chrono::high_resolution_clock::now();
		vec_gen(nr_lines, h_seq, vecs);

		time5 = chrono::high_resolution_clock::now();
		// make filed
		if (grid_regular) 
			make_field_scalar(nr_lines, fielddim_x, fielddim_y, fielddim_z, linelength, vecs, y, RF, resolutionfactor);	
		else  
			make_field_irregular_scalar(nr_lines, field_nr_points, linelength, vecs, y, RF_irregular, resolutionfactor);
		time6 = chrono::high_resolution_clock::now();

		// print		
//		if (grid_regular) print_regular(RF, fielddim_x, fielddim_y, fielddim_z);
//		else print_irregular(RF_irregular, fielddim_x, fielddim_y, fielddim_z);
	}

	// end timing
	auto time_total     = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-time1).count();
	auto time_halton    = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
	auto time_lines     = std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
	auto time_vec_gen   = std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
	auto time_makefield = std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();
	
	cout << endl
		<< "--- make_field " << (grid_regular?"regular":"irregular") << " sequential time ---" << endl
		<< "halton sequence        " << time_halton << " ms" << endl
		<< "lines generation (fft) " << time_lines << " ms" << endl
		<< "vector generation      " << time_vec_gen << " ms" << endl
		<< "make field             " << time_makefield << " ms" << endl
		<< "---" << endl
		<< "total time             " << time_total << " ms" << endl
		<< "---" << endl;

	// debug
	if (grid_regular) {
		print_double(RF, 100); // first 100 elements
		print_double(RF+(field_nr_points-100-1), 100); // last 100 elements
	}

	// cleanup 
	if (grid_regular) free(RF);	
	else {
		free(RF_irregular[0]);
		free(RF_irregular[1]);
		free(RF_irregular[2]);
		free(RF_irregular[3]);
		free(RF_irregular);
	}	
	free(vecs);
	free(h_seq);
	free(y);
	
	#ifdef _WIN32
	getchar();
	#endif

	return 0;
}
