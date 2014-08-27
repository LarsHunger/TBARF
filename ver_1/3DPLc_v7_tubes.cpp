#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "3DPLC_host.h"
#include "3DPLc.h"
#include  "fft_field_line.h"
using namespace std;

//                           pl   nrlines linelen size(x,y,z)			nr_fields factor
const char *test_1[] = {"X", "0", "1024", "135",	"64", "64", "64",		"1", "1"};
const char *test_2[] = {"X", "0", "1024", "200",	"96", "96", "96",		"1", "1"};
const char *test_3[] = {"X", "0", "1024", "270",	"128", "128", "128",	"1", "1"};
const char *test_4[] = {"X", "0", "1024", "335",	"160", "160", "160",	"1", "1"};
const char *test_5[] = {"X", "0", "1024", "400",	"192", "192", "192",	"1", "1"};
const char *test_6[] = {"X", "0", "1024", "470",	"224", "224", "224",	"1", "1"};
const char *test_7[] = {"X", "0", "1024", "535",	"256", "256", "256",	"1", "1"};

const char *test_neb[] = {"X", "0", "1024", "2600",		"256", "256", "256",			"1", "1"};
const char *test_cos[] = {"X", "0", "1065", "2048",		"512", "512", "512",			"1", "1"};
const char *test_mil[] = {"X", "0", "8550", "2048",		"4096", "4096", "4096",			"1", "1"};

#define argv test_3

struct tbarf_param {
	double PLindex;
	int nr_lines;
	int linelength;
	int x, y, z;
	int nr_3Dfields;
	double resolution_factor;
};



// complete test set with different sizes
vector<tbarf_param> get_test_set_A() {
	vector<tbarf_param> tests;

//	for(int size = 64; size <= 192; size+=64)
//	for(int size = 64; size <= 704; size+=64)
    for(int size = 64; size <= 320; size+=64)
//	for(int size = 384; size <= 704; size+=64)
//	int size = 704;
	{
		int l = (int) ceil(1.2 *sqrt(3) * size);		
		tbarf_param t = { 0, 1024, l, size, size, size, 1, 1};
		tests.push_back(t);
	}
	return tests;
}

/* A base grid of 128^3 with 1024 lines, used for optimization copmarison */
vector<tbarf_param> get_test_set_B() {
	vector<tbarf_param> tests;
	int size = 128;	
	tbarf_param t = { 0, 1024, 2600, size, size, size, 1, 1};
	tests.push_back(t);
	return tests;
}

/* Massive dataset */
vector<tbarf_param> get_test_set_C() {
	vector<tbarf_param> tests;	
	tbarf_param t1 = { 0, 1024, 1065, 512, 512, 512, 1, 1}; // nebulae
	tbarf_param t2 = { 0, 1024, 4350, 256, 256, 2500, 1, 1}; // cosmological	
	tbarf_param t3 = { 0, 2048, 8550, 4096, 4096, 4096, 1, 1}; // millennium
	tests.push_back(t1);
	tests.push_back(t2);
//	tests.push_back(t3);	
	return tests;
}



void tbarf(tbarf_param p) ;

/* parallel version of the turning band method */
int main(int argc, char *_argv_[]) 
{
	// warmup();
	// make_field_3dfft_batch();		

	vector<tbarf_param> tests = get_test_set_C();
	for(tbarf_param p : tests){
		tbarf(p);
	}

	cout << "press a key to continue" << endl;
	getchar();
}

void tbarf(tbarf_param p) 
{
	bool grid_regular = true;
	int deviceId = 1;
	bool outofcore = true;
		
	cout << (grid_regular ? "Regular" : "Irregular") << " mesh" << endl;
	cout << (outofcore ? "Out-of-core" : "In-core") << " kernel" << endl;
	
	double *RF = nullptr;
	cl_double4 *RF_irregular = nullptr;		

	// read input	
	double PLindex     = p.PLindex; //atof(argv[1]);	
	int nr_lines    = p.nr_lines; //atoi(argv[2]);
	int linelength  = p.linelength; //atoi(argv[3]);
	int fielddim_x  = p.x; //atoi(argv[4]);
	int fielddim_y  = p.y; //atoi(argv[5]);
	int fielddim_z  = p.z; //atoi(argv[6]);
	int nr_3Dfields = p.nr_3Dfields; //atoi(argv[7]);
	double resolutionfactor = p.resolution_factor; //atof(argv[8]);	

	linelength = (int) resolutionfactor*2*linelength; //scale line to the right size and double in length for invers directions  
	size_t field_nr_points = fielddim_x*fielddim_y*fielddim_z;

	// memory allocation
	double *y     = (double*) malloc (sizeof(double)*(linelength*nr_lines+1));	
	double *h_seq = (double*) malloc(sizeof(double)*2*nr_lines); 
	cl_double4 *vecs = (cl_double4*) malloc(sizeof(cl_double4)*nr_lines);

	
	if (grid_regular) {
		// Allocates RF as 1 block array with the values but not the coordinates saved if grid==regular
		RF = (double*)malloc(sizeof(double)*(field_nr_points)); 
		if(RF == NULL) cout << "Error while allocating a filed of " << field_nr_points << " points" << endl;		
	}
	else {		
		RF_irregular = (cl_double4*) malloc(sizeof(cl_double4)*(field_nr_points));
		if(RF_irregular == NULL) cout << "Error while allocating a filed of " << field_nr_points << " points" << endl;

		/*
		// Allocates RF as 4 block array where index 0,1,2 are x,y,z and 3 is the field value usefull for extention to irregular grids				
		RF_irregular = (double**)malloc(sizeof(double*)*4); 
		for (int i=0; i<4; ++i){
			RF_irregular[i]=(double*)malloc(sizeof(double)*(field_nr_points));
		}
		*/		

		// regular distribution		
		int index = 0;
		for (int i=0; i<fielddim_x; i++)	
			for (int j=0; j<fielddim_y; j++)	
				for (int k=0; k<fielddim_z; k++){
					RF_irregular[index].s[0] = i;
					RF_irregular[index].s[1] = j;
					RF_irregular[index].s[2] = k;
					index++;
				}
/*
		// random distribution
		init_rand();
		for (size_t i=0; i<field_nr_points; i++){
			RF_irregular[i].s[0] = rand_01() * (fielddim_x-1);
			RF_irregular[i].s[1] = rand_01() * (fielddim_y-1);
			RF_irregular[i].s[2] = rand_01() * (fielddim_z-1);
		}

		// jitter pattern
		init_rand();		
		for (int i=0; i<fielddim_x; i++)	
			for (int j=0; j<fielddim_y; j++)	
				for (int k=0; k<fielddim_z; k++){
					RF_irregular[i].s[0] = i + rand_01();
					RF_irregular[i].s[1] = j + rand_01();
					RF_irregular[i].s[2] = k + rand_01();
				}	
*/
	}
	
	init_rand();
	std::cout << "field dimension " << fielddim_x << "," << fielddim_y << "," << fielddim_z << " - 3d fields " << nr_3Dfields << std::endl;
	
	ocl_init(deviceId, nr_lines, field_nr_points, linelength, fielddim_x, fielddim_y, fielddim_z, grid_regular, outofcore);

	/*
		Testing different parallel implementations (is using function pointers)		
	*/
	typedef void (*make_field_ptr)(int, int, int, int, int, const cl_double4*, const double*, double*, double);		
	std::vector<make_field_ptr> make_field_impls;
//	make_field_impls.push_back(make_field);		// this is slow (runs on the host)
//	make_field_impls.push_back(ocl_make_field1);
//	make_field_impls.push_back(ocl_make_field1b);	
//	make_field_impls.push_back(ocl_make_field1u);
//	make_field_impls.push_back(ocl_make_field1bu); // best
//	make_field_impls.push_back(ocl_make_field2); // this uses a slower parallelization approach
//	make_field_impls.push_back(ocl_make_field3); // this uses an awful parallelization approach (not working!)
	make_field_impls.push_back(ocl_make_field_outcore); // this rocks

	// prototypes for irregular
	typedef void (*make_irrfield_ptr)(int,int,int,const cl_double4*, const double*, cl_double4*,double);		
	std::vector<make_irrfield_ptr> make_irrfield_impls;
//	make_irrfield_impls.push_back(make_field_irregular); // this is slow (runs on the host)
//	make_irrfield_impls.push_back(ocl_make_field_irregular);		
//	make_irrfield_impls.push_back(ocl_make_fieldb_irregular);
//	make_irrfield_impls.push_back(ocl_make_fieldu_irregular); // best
	make_irrfield_impls.push_back(ocl_make_fieldbu_irregular);

	size_t impl_size = (grid_regular)? make_field_impls.size() : make_irrfield_impls.size();	

	for(int v=0; v<nr_3Dfields; v++) {		
		cout << endl << "--- Field #" << v << endl << endl;
		
		// for each implementation
		for(int impl = 0; impl<impl_size; impl++) {
			std::chrono::time_point<std::chrono::high_resolution_clock> time1, time2, time3, time4, time5, time6, time7;

			// only the first time (this may be a problem otherwsise, as Halton sequence gives random results)
			if(impl == 0){
				// halton sequence in h_seq, where the first nr_lines elements are the base 2 VDc sequence 
				// and the last nr_lines elements are the base 3 Vdc sequence, Halton sequence is the same 
				// for all iterations so it is done outside of the loop	
				time1 = std::chrono::system_clock::now();
				Halton_seq(nr_lines, h_seq); 
				time2 = chrono::high_resolution_clock::now();

				// fft here - writes all the RF lines in y
				time3 = chrono::high_resolution_clock::now();
				generate_PL_lines(PLindex, nr_lines, linelength, y, fielddim_z);
				time4 = chrono::high_resolution_clock::now();

				vec_gen(nr_lines, h_seq, vecs);			
				time5 = chrono::high_resolution_clock::now();
			}

			time6 = chrono::high_resolution_clock::now();
			if (grid_regular) 
				make_field_impls[impl](nr_lines, fielddim_x, fielddim_y, fielddim_z, linelength, vecs, y, RF, resolutionfactor);	
			else 
				make_irrfield_impls[impl](nr_lines, field_nr_points, linelength, vecs, y, RF_irregular, resolutionfactor);
			time7 = chrono::high_resolution_clock::now();		
	
			// end timing			
			auto time_halton    = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
			auto time_lines     = std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
			auto time_vec_gen   = std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
			auto time_makefield = std::chrono::duration_cast<std::chrono::milliseconds>(time7-time6).count();
			//auto time_total     = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-time1).count();
			auto time_total = time_halton + time_lines + time_vec_gen + time_makefield;
	
			cout << endl << "--- make_field " << (grid_regular?"regular":"irregular") << " ---" << endl
				<< "halton sequence        " << time_halton << " ms" << endl
				<< "lines generation (fft) " << time_lines << " ms" << endl
				<< "vector generation      " << time_vec_gen << " ms" << endl
				<< "make field             " << time_makefield << " ms" << endl
				<< "---" << endl
				<< "total time             " << time_total << " ms" << endl
				<< "---" << endl;

			// debug print
			if (grid_regular) 
			{
				print_double(RF, 100); // first 100 elements
				print_double(RF+(field_nr_points-100), 100); // last 100 elements
			}
			else {
				print_double(RF_irregular, 100); // first 100 elements (only forth value) 
				print_double(RF_irregular+(field_nr_points-100), 100); // last 100 elements			
			}			

		} // for each implementation
	} // for each grid


	// cleanup 
	if (grid_regular) free(RF);	
	else free(RF_irregular);
		
	free(vecs);
	free(h_seq);
	free(y);
	
	ocl_finalize();
}





	


