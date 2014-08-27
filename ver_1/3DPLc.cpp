#include "3DPLc.h"

#include <cmath>
#include <fftw3.h>
#include <assert.h>

// scalar version
void Rotx(double alpha,double* vecs,int nr_lines) {
	int i;
	double cos_alpha,sin_alpha,temp;
	cos_alpha=cos(alpha);
	sin_alpha=sin(alpha);
	for(i=0;i<nr_lines;i++){
		temp=cos_alpha*vecs[3*i+1]+vecs[3*i+2]*sin_alpha;
		vecs[3*i+2]=vecs[3*i+2]*cos_alpha-sin_alpha*vecs[3*i+1];
		vecs[3*i+1]=temp;
	}
}


void Roty(double alpha,double* vecs,int nr_lines){
	int i;
	double cos_alpha,sin_alpha,temp;
	cos_alpha=cos(alpha);
	sin_alpha=sin(alpha);
	
	for(i=0;i<nr_lines;i++){
		temp=cos_alpha*vecs[3*i]-vecs[3*i+2]*sin_alpha;
		vecs[3*i+2]=vecs[3*i+2]*cos_alpha+sin_alpha*vecs[3*i];
		vecs[3*i]=temp;
	}
}


void Rotz(double alpha,double* vecs,int nr_lines) {
	int i;
	double cos_alpha,sin_alpha,temp;
	cos_alpha=cos(alpha);
	sin_alpha=sin(alpha);

	for(i=0;i<nr_lines;i++){
		temp=cos_alpha*vecs[3*i]+vecs[3*i+1]*sin_alpha;
		vecs[3*i+1]=vecs[3*i+1]*cos_alpha-sin_alpha*vecs[3*i];
		vecs[3*i]=temp;
	}
}


// vector version
void Rotx(double alpha, cl_double4* vecs,int nr_lines) {	
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);
	for(int i=0;i<nr_lines;i++){
		double temp=cos_alpha*vecs[i].s[1]+vecs[i].s[2]*sin_alpha;
		vecs[i].s[2]=vecs[i].s[2]*cos_alpha-sin_alpha*vecs[i].s[1];
		vecs[i].s[1]=temp;
	}
}

void Roty(double alpha, cl_double4* vecs,int nr_lines) {
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);
	for(int i=0;i<nr_lines;i++){
		double temp = cos_alpha*vecs[i].s[0]-vecs[i].s[2]*sin_alpha;
		vecs[i].s[2] = vecs[i].s[2]*cos_alpha+sin_alpha*vecs[i].s[0];
		vecs[i].s[0] = temp;
	}

}

void Rotz(double alpha, cl_double4* vecs,int nr_lines) {
	double cos_alpha=cos(alpha);
	double sin_alpha=sin(alpha);

	for(int i=0;i<nr_lines;i++){
		double temp=cos_alpha*vecs[i].s[0]+vecs[i].s[1]*sin_alpha;
		vecs[i].s[1] = vecs[i].s[1]*cos_alpha-sin_alpha*vecs[i].s[0];
		vecs[i].s[0] = temp;
	}
}



/* Transforms equally distributed random numbers into normally distributed ones*/
/// XXX  C++11 adds std::normal_distribution
void Box_Mueller(int linelength,double *a,double *b) { 
	int k;
	double temp1,temp2;

	for(k=0;k<linelength;k++){
		temp1 = a[k];
		temp2 = b[k];
		a[k]= sqrt((-2)*log(temp1))*cos(2*PI*temp2);
		assert(!_isnan(a[k]));
		b[k]= sqrt((-2)*log(temp1))*sin(2*PI*temp2);
		assert(!_isnan(a[k]));

	}
}


void generate_PL_lines(double PLindex,int nr_lines,int linelength, 
					   double *y, // out
					   int fielddim_z) 
{
	double *noise_in,*transfer_in,temp,average;	
	fftw_plan fftw_plan_transfer_fw,fftw_plan_noise_fw,fftw_plan_transfer_bw;
	fftw_complex *transfer_out,*noise_out; 	
	
	size_t index = 0;

	// FFT preparation 
	const size_t out_size = sizeof(fftw_complex)*(linelength/2+1);
	const size_t in_size = sizeof(double)*(linelength+1);
	noise_in     = (double*)       fftw_malloc(in_size);	/* all these allocs can technically also be done outside of the function, if performance is bad this might be a consideration*/	
	noise_out    = (fftw_complex*) fftw_malloc(out_size); // check if i need 2 times +1
	transfer_in  = (double*)       fftw_malloc(in_size);  
	transfer_out = (fftw_complex*) fftw_malloc(out_size);	

	// if memory is allocated outside of the function plan generation etc can also be done outside
	fftw_plan_transfer_fw = fftw_plan_dft_r2c_1d(linelength, transfer_in, transfer_out,FFTW_MEASURE); 
	fftw_plan_noise_fw    = fftw_plan_dft_r2c_1d(linelength, noise_in, noise_out,FFTW_MEASURE);     
	fftw_plan_transfer_bw = fftw_plan_dft_c2r_1d(linelength, transfer_out, transfer_in,FFTW_MEASURE);	
	 
		
	for(int v=0; v<nr_lines; v++){

		for(int i=0; i<linelength; i++){
			noise_in[i] = rand_01();
			assert(noise_in[i] < 1.0 && noise_in[i] >= 0.0);
			assert(!_isnan(noise_in[i]));
			transfer_in[i] = rand_01();
			assert(transfer_in[i] < 1.0 && transfer_in[i] >= 0.0);
			assert(!_isnan(transfer_in[i]));
		}	
	
		Box_Mueller(linelength, noise_in, transfer_in);

		for(int i=0; i<linelength; i++){	
			noise_in[i] = 2*noise_in[i]-1; // around 0
			noise_in[i] = 5*noise_in[i];  //changes the deviation, which values need to be put will be investigated			
		}

		transfer_in[0]=1.0;
		for(int i=1; i<linelength; i++){
			transfer_in[i] = (transfer_in[i-1]/(i))*(i-1-(PLindex/2.0));
		}		

		fftw_execute(fftw_plan_noise_fw);		
		fftw_execute(fftw_plan_transfer_fw);

		for(int i=0; i<0.5*linelength+1; i++){
			temp = (transfer_out[i][0]*noise_out[i][0]+transfer_out[i][1]*noise_out[i][1]) / linelength;
			transfer_out[i][1] = (transfer_out[i][0]*noise_out[i][1]-transfer_out[i][1]*noise_out[i][0]) / linelength;
			transfer_out[i][0] = temp;
		}		
		
		fftw_execute(fftw_plan_transfer_bw);	

		for(int i=0; i<linelength; i++){
			transfer_in[i]=transfer_in[i]/sqrt((double) linelength);
		}

		average=0;
		for(int i=0; i<linelength; i++){
			average=average+transfer_in[i];
		}
		average=average/linelength;

		for(int i=0; i<linelength; i++){
			y[index]=(transfer_in[i]-average);
			index++;
		}

	}


	/* Cleanup */
	fftw_free(transfer_in);
	fftw_free(transfer_out);
	fftw_free(noise_out);
	fftw_free(noise_in);

	fftw_destroy_plan(fftw_plan_transfer_fw);	
	fftw_destroy_plan(fftw_plan_transfer_bw);
	fftw_destroy_plan(fftw_plan_noise_fw);
	fftw_cleanup();
}

void vec_gen(int nr_lines, const double *h_seq, cl_double4* vecs) {	
	int i,perm;
	double t,phi,alpha,temp;

	for(i=0;i<nr_lines;i++){
		t = 2*h_seq[nr_lines+i]-1;
		phi = 2.0*PI*h_seq[i];
		vecs[i].s[0] = sqrt(1.0-t*t)*cos(phi);		//x-coordinate
		vecs[i].s[1] = sqrt(1.0-t*t)*sin(phi);		//y-coordinate
		vecs[i].s[2] = t;							//z_coordinate
		vecs[i].s[3] = 0;
	}

	temp = rand_01();		// one of 5 permutations
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



/* creates direction vectors for the lines and rotates the whole sphere by the same random angles */
void vec_gen(int nr_lines, const double *h_seq, double* vecs) {	
	int i,perm;
	double t,phi,alpha,temp;

	for(i=0;i<nr_lines;i++){
		t = 2*h_seq[nr_lines+i]-1;
		phi = 2.0*PI*h_seq[i];
		vecs[3*i]   = sqrt(1.0-t*t)*cos(phi);		//x-coordinate
		vecs[3*i+1] = sqrt(1.0-t*t)*sin(phi);		//y-coordinate
		vecs[3*i+2] = t;							//z_coordinate
	}

	temp = rand_01();		// one of 5 permutations
	temp = temp*6.0;	
	perm = (int) ceil(temp);	/// xxx check this cast

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

void Halton_seq(int nr_lines, double* h_seq) 
{
	// generate 1st part of Halton_seq... Van der Corput sequence with base 2
	for(int i=0; i<nr_lines; i++) 
		h_seq[i] = vdc(i+1,2);

	// generate 2nd part of Halton_seq... Van der Corput sequence with base 3
	for(int i=nr_lines; i<2*nr_lines; i++) 
		h_seq[i] = vdc(i-nr_lines+1,3);	
}


/*
 * Make a regular field.
 */
void make_field_scalar(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
				const double* vecs, const double* y, 
				double* RF, // out
				double resolutionfactor) 
{		
	const size_t fielddim_prod = fielddim_x * fielddim_y * fielddim_z;
	const size_t fielddim_yz = fielddim_y * fielddim_z;

	for(int i=0; i<fielddim_prod; i++) 
		RF[i]=0;

	int test = 0;
	for(int l=0; l<nr_lines; l++){	
		for(int i=0; i<fielddim_x; i++){	
			for(int j=0;j<fielddim_y;j++){
				for(int k=0;k<fielddim_z;k++){
					double linecoord = - (i*vecs[3*l])-(j*vecs[3*l+1])-(k*vecs[3*l+2]);
					size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;

					double y_val = y[l*linelength+linepoint];				
					size_t rf_index = i*fielddim_yz+j*fielddim_z+k;

					RF[rf_index] += y_val;
				}
			}
		}
	}
}

// vectorial version
void make_field(int nr_lines, int fielddim_x, int fielddim_y, int fielddim_z, int linelength,
				const cl_double4* vecs, const double* y, 
				double* RF, // out
				double resolutionfactor) 
{		
	const size_t fielddim_prod = fielddim_x * fielddim_y * fielddim_z;
	const size_t fielddim_yz = fielddim_y * fielddim_z;

	for(int i=0; i<fielddim_prod; i++) 
		RF[i]=0;

	int test = 0;
	for(int l=0; l<nr_lines; l++){	
		for(int i=0; i<fielddim_x; i++){	
			for(int j=0;j<fielddim_y;j++){
				for(int k=0;k<fielddim_z;k++){		
					double linecoord = - (i*vecs[l].s[0]) - (j*vecs[l].s[1]) - (k*vecs[l].s[2]);
					size_t linepoint = round(linecoord*resolutionfactor)+ linelength*0.5 + 1;
					double y_val = y[l*linelength+linepoint];				
					size_t rf_index = i*fielddim_yz+j*fielddim_z+k;
					RF[rf_index] += y_val;					
				}
			}
		}
	}
}

/*
 * Make an irregular field.
 */
void make_field_irregular_scalar(int nr_lines, int field_nr_points, int linelength,
						  const double *vecs, const double *y, 
						  double **RF, // out
						  double resolutionfactor) 
{	
	for(int i=0;i<field_nr_points;i++) 
		RF[3][i]=0;

	for(int l=0;l<nr_lines;l++){	
		for(int i=0;i<field_nr_points;i++){
			double linecoord = -(RF[0][i]*vecs[3*l])-(RF[1][i]*vecs[3*l+1])-(RF[2][i]*vecs[3*l+2]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;
			double y_val = y[l*linelength+linepoint];
			RF[3][i] += y_val;					//linepoint=NINT(resolutionfactor*linecoord)+n*0.5+1

		}
	}
}

void make_field_irregular(int nr_lines, int field_nr_points, int linelength,
						  const cl_double4 *vecs, const double *y, 
						  cl_double4 *RF, 
						  double resolutionfactor) 
{		
	for(int i=0;i<field_nr_points;i++) 
		RF[i].s[3] = 0;

	for(int l=0;l<nr_lines;l++){	
		for(int i=0;i<field_nr_points;i++){
			//double linecoord = -(RF[0][i]*vecs[3*l])-(RF[1][i]*vecs[3*l+1])-(RF[2][i]*vecs[3*l+2]);
			double linecoord = -(RF[i].s[0]*vecs[l].s[0])-(RF[i].s[1]*vecs[l].s[1])-(RF[i].s[2]*vecs[l].s[2]);
			size_t linepoint = round(linecoord*resolutionfactor)+linelength*0.5+1;
			double y_val = y[l*linelength+linepoint];
			RF[i].s[3] += y_val;					//linepoint=NINT(resolutionfactor*linecoord)+n*0.5+1
		}
	}
}
