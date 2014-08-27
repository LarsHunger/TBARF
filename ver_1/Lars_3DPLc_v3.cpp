/*
	Turning bands. Original Fortan code from Lars Hunger, ported to C/OpenCL by Biagio Cosenza.
*/

#include "lib_icl.h"

#include "fft/fftw3.h"

//#include <string.h>
//#include <cstdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


//#define M_PI 3.1415926535897932384626433832795028841971f;

struct matrix {
	cl_float3 val;	
//	float val[9];
//	float operator(int x, int y){ return val[x+y*3]; }
};


void BOX_MULLER(float *a, float *b, int linelength);

float rand_01(){
	return (float)(rand() / (float) RAND_MAX);
}

void rand_01_v(float *in, int size){
	for(int i=0; i<size; i++)
		in[i] = rand_01();
}

int main(int argc, char **argv)
{	
	//real, allocatable, dimension(:,:,:) :: CrossCovMatrix
	cl_float3 *CrossCovMatrix;

	int n, m, nr_fields,nr_matrizes,nr_variables,f,nr_cov_functions,nr_3Dfields,v;
	int info,fielddim_xy,fielddim_z,whichout,z,linelength;
	float C,a,temp,PLindex,t1,t2;

	//real, allocatable, dimension(:) :: imag_vec,real_vec,tempvec,transfer_in,noise_in //y has the one dimensional Random field
	float *imag_vec, *real_vec, *tempvec; //y has the one dimensional Random field

	
	cl_float3 *y;
	cl_float2 *h_seq, *fieldswap;
	cl_float2 *vecs, *Cov_matrix;	
	cl_float3 *RF, *RFfinal;
	/*	real, allocatable, dimension(:,:,:) :: y
	real,allocatable, dimension(:,:) :: h_seq,fieldswap
	real,allocatable,dimension(:,:) :: vecs,Cov_matrix
	real, allocatable,dimension(:,:,:,:) :: RF,RFfinal
	*/	
		
	//integer*8 :: fftw_plan_transfer_fw,fftw_plan_transfer_bw,fftw_plan_noise_fw,fftw_plan_noise_bw
	int fftw_plan_transfer_fw, fftw_plan_transfer_bw,fftw_plan_noise_fw,fftw_plan_noise_bw; // XXX init to 0?
	char filename[50], s[50];


	// parsing arguments from command line
	PLindex = atof(argv[0]);
	nr_fields = atoi(argv[1]);
	linelength = atoi(argv[2]);
	fielddim_xy = atoi(argv[3]);
	fielddim_z = atoi(argv[4]);
	nr_3Dfields = atoi(argv[5]);

	/*
	CALL GETARG(1, s)	//PLindex of the 1D field
	read (s,*) PLindex
	CALL GETARG(2, s)	//how many lines
	read (s,*) nr_fields
	CALL GETARG(3, s)	//how long lines
	read (s,*) linelength
	CALL GETARG(4, s)	//length of 3D cube xy side
	read (s,*) fielddim_xy
	CALL GETARG(5, s)	//length of 3D cube z side should be made 3 variables x,y,z
	read (s,*) fielddim_z
	CALL GETARG(6, s)	//  how many 3D fields
	read (s,*) nr_3Dfields
	*/
	nr_variables=1;	

	linelength=2*linelength;

	// mem allocation
	y = new cl_float3[linelength*nr_fields];

	real_vec = new float[linelength];
	imag_vec = new float[linelength];
	RF = new cl_float3[fielddim_xy*fielddim_xy*fielddim_z];
	h_seq = new cl_float2[nr_fields];
	vecs = new cl_float2[nr_fields];
	/*
	ALLOCATE(y(1,linelength,nr_fields))
	ALLOCATE(noise_in(linelength))
	ALLOCATE(noise_out(INT(0.5*linelength)+1))	
	ALLOCATE(transfer_in(linelength))
	ALLOCATE(transfer_out(INT(0.5*linelength)+1))	
	ALLOCATE(real_vec(linelength))
	ALLOCATE(imag_vec(linelength))
	ALLOCATE(RF(1,fielddim_xy,fielddim_xy,fielddim_z))
	ALLOCATE(h_seq(2,nr_fields))
	ALLOCATE(vecs(3,nr_fields))
	*/

	
	// init random seed
	srand(42/*time(NULL)*/);

	//complex, allocatable, dimension(:) :: transfer_out,noise_out
	double *noise_in    = (double*) fftw_malloc(sizeof(double) * linelength);
	double *transfer_in = (double*) fftw_malloc(sizeof(double) * linelength);
	fftw_complex *noise_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * int(0.5*linelength)+1);	
	fftw_complex *transfer_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * int(0.5*linelength)+1);

/* using fftw from MIT
	dfftw_plan_dft_r2c_1d(fftw_plan_transfer_fw,linelength,transfer_in, transfer_out,"FFTW_MEASURE");
	dfftw_plan_dft_r2c_1d(fftw_plan_noise_fw,linelength,noise_in, noise_out,"FFTW_MEASURE");
	dfftw_plan_dft_c2r_1d(fftw_plan_transfer_bw,linelength,transfer_out, transfer_in,"FFTW_MEASURE");
	dfftw_plan_dft_1d(fftw_plan_noise_bw,2*linelength, noise_in, noise_out, "FFTW_BACKWARD", "FFTW_MEASURE");	
	
	fftw_complex *in, *out;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
*/  
	fftw_plan p1, p2, p3, p4;	
	p1 = fftw_plan_dft_r2c_1d(linelength, transfer_in, transfer_out, FFTW_FORWARD | FFTW_MEASURE); // FFTW_FORWARD, FFTW_MEASURE
	p2 = fftw_plan_dft_r2c_1d(linelength, noise_in, noise_out, FFTW_FORWARD | FFTW_MEASURE);
	p3 = fftw_plan_dft_c2r_1d(linelength, transfer_out, transfer_in, FFTW_FORWARD | FFTW_MEASURE);
	p4 = fftw_plan_dft_1d(linelength*2, (fftw_complex*) noise_in, noise_out, FFTW_BACKWARD, FFTW_MEASURE);
  
/*
	fftw_execute(p);    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
*/
	printf("%d, %d, %d\n", fielddim_xy, fielddim_z, nr_3Dfields);

	

	
	for(v=0; v<nr_3Dfields; v++)
	{
		for(int j=0; j<nr_fields; j++)
		{
			// LineGeneration Goes Here
			for(int i=0; i<linelength; i++)
				real_vec[i] = 0;

			rand_01_v(real_vec,linelength);
			rand_01_v(imag_vec,linelength);
			
			BOX_MULLER(real_vec,imag_vec,linelength);

			//real_vec=2*real_vec-1 //  normalverteilt zwischen -1 und 1
			imag_vec=0;
			//noise_in = ((float)fielddim_z)*real_vec; //  deviation of the noise changes here
			for(int i=0; i<linelength; i++)
				noise_in[i] = fielddim_z * real_vec[i];


			real_vec[0]=1.0;
			for(int i=1; i<linelength; i++){
				real_vec[i]=(real_vec[i-1]/double(i-1))*(i-2-(PLindex/2.0));
				transfer_in[i] = real_vec[i];
			}			

						
			//dfftw_execute(fftw_plan_noise_fw);
			//dfftw_execute(fftw_plan_transfer_fw);
			fftw_execute(p2);
			fftw_execute(p1);

			transfer_out=CONJG(transfer_out) ;	//  I think my definition of complex multiplication is different
			transfer_out=(transfer_out*noise_out)/(linelength); //  normalization usually sqrt(N) but since I multiply 2 FFT's it's /N

			//dfftw_execute(fftw_plan_transfer_bw);
			fftw_execute(plan3);

		
			transfer_in=transfer_in/sqrt(double(linelength));

			y[0,j] = transfer_in;
		}		


		Halton_seq(nr_fields,h_seq);

		/*			
		!do i=1,nr_fields
		!	Write(*,*) h_seq(1,i), h_seq(2,i)	
		!enddo 
		*/	
		vec_gen(nr_fields,h_seq,vecs);

		/*
		!do i=1,nr_fields
		!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i)
		!enddo 
		!call cpu_time(t1)
		*/

		make_field(nr_fields ,fielddim_xy,fielddim_z ,linelength ,vecs ,y(1,:,:) , RF(1,:,:,:))

		for(int i=0; i<fielddim_xy; i++)
			for(int j=0; j<fielddim_xy; j++)
				for(int k=0; k<fielddim_z; k++){
					printf("%f ", RF(1,i,j,k));
				}

	} //enddo

	/*
	dfftw_destroy_plan(fftw_plan_noise_bw);
	dfftw_destroy_plan(fftw_plan_transfer_bw);
	dfftw_destroy_plan(fftw_plan_noise_fw);
	dfftw_destroy_plan(fftw_plan_transfer_fw);
	*/
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_destroy_plan(p3);
	fftw_destroy_plan(p4);


	fftw_free(transfer_in);
	fftw_free(transfer_out);	
	fftw_free(noise_in);
	fftw_free(noise_out);

	delete real_vec;
	delete imag_vec;
	delete h_seq;
	delete vecs;
	delete RF;
	delete y;

}




void BOX_MULLER(float *a, float *b, int linelength){
	//integer :: linelength
	//real, dimension(linelength) :: a,b
	
	
	for(int k=0; k<linelength; k++){
		float temp1=a(k);
		float temp2=b(k);
		a(k)=sqrt((-2)*log(temp1))*cos(2*M_PI*temp2);
		b(k)=sqrt((-2)*log(temp1))*sin(2*M_PI*temp2);
	}	
	return;
};


void VdC_seq (int seed, int base, int n, int r){
	//integer n,i,base,digit(n),seed,seed2(n)
	//real base_inv,r(n)  

	for(i = 0; i<n; i++){	//fill vector with the 
		seed2(i) = i;
	}

	seed2(1:n) = seed2(1:n) + seed - 1;

	base_inv = 1.0D+00 / real ( base, kind = 8 );

	r(1:n) = 0.0D+00;

	while ( any ( seed2(1:n) /= 0 ) )
	{
		digit(1:n) = mod ( seed2(1:n), base );
		r(1:n) = r(1:n) + real ( digit(1:n), kind = 8 ) * base_inv;
		base_inv = base_inv / real ( base, kind = 8 );
		seed2(1:n) = seed2(1:n) / base;
	}
	return;
}

void Halton_seq(int nr_fields,float h_seq*, int nr_fields){
	//integer, intent(in) :: nr_fields
	//real, intent(out),dimension(2,nr_fields) :: h_seq

	VdC_seq(1,2,nr_fields,h_seq(1,:));
	VdC_seq(1,3,nr_fields,h_seq(2,:));
}


void vec_gen(int nr_fields, float *h_seq, float *vecs){

	/*
	integer, intent(in) :: nr_fields
	integer :: i,order,j
	integer,dimension(6)	:: rn_check
	real, dimension(3,3)	:: Rx,Ry,Rz
	real :: t,phi,perm,alpha
	real, intent(in),dimension(2,nr_fields) :: h_seq
	real,dimension(3,nr_fields),intent(out)  :: vecs 
	*/

	for(i=0;i<nr_fields; i++){ // iters 1024
		t=2*h_seq(2,i)-1.0;
		phi=2.0*M_PI*h_seq(1,i);
		vecs(1,i)=sqrt(1.0-t**2)*cos(phi);
		vecs(2,i)=sqrt(1.0-t**2)*sin(phi);
		vecs(3,i)=t; 
	}

	/*
	!do i=1,nr_fields
	!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i) !, (vecs(1,i)**2+vecs(2,i)**2+vecs(3,i)**2)
	!enddo 
	*/

	rn_check=0;


	order=8;
	while (order > 5){
		RANDOM_NUMBER(perm);
		order=perm/0.1666666666667;
	}

	RANDOM_NUMBER(alpha);	//  create Rotation matrizes
	alpha=alpha*2*M_PI;
	Rot_mat(alpha,1,Rx);

	RANDOM_NUMBER(alpha);
	alpha=alpha*2*M_PI;
	Rot_mat(alpha,2,Ry);

	RANDOM_NUMBER(alpha);
	alpha=alpha*2*M_PI;
	Rot_mat(alpha,3,Rz);

	//!write(*,*) Rx
	//!write(*,*) Ry
	//!write(*,*) Rz

	switch(order){ //  Turn in one of the permutation orders
	case 0:
		for(i=0; i<nr_fields; i++){
			vecs(:,i)=MATMUL(Rx,vecs(:,i));
			vecs(:,i)=MATMUL(Ry,vecs(:,i));
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
		}
	case 1:
		for(i=0; i<nr_fields; i++){
			vecs(:,i)=MATMUL(Rx,vecs(:,i));
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
			vecs(:,i)=MATMUL(Ry,vecs(:,i));
		}
	case 2:
		for(i=0; i<nr_fields; i++){
			vecs(:,i)=MATMUL(Ry,vecs(:,i));
			vecs(:,i)=MATMUL(Rx,vecs(:,i));
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
		}
	case 3:
		for(i=0; i<nr_fields; i++){
			vecs(:,i)=MATMUL(Ry,vecs(:,i));
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
			vecs(:,i)=MATMUL(Rx,vecs(:,i));
		}
	case 4:
		for(i=0; i<nr_fields; i++){
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
			vecs(:,i)=MATMUL(Rx,vecs(:,i));
			vecs(:,i)=MATMUL(Ry,vecs(:,i));
		}
	case 5:
		for(i=0;i<nr_fields;i++)
			vecs(:,i)=MATMUL(Rz,vecs(:,i));
		vecs(:,i)=MATMUL(Ry,vecs(:,i));
		vecs(:,i)=MATMUL(Rx,vecs(:,i));
		enddo
	default:
		printf("You fucked up!\n");
	}

	/*
	!Write(*,*)
	!do i=1,nr_fields
	!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i) , (vecs(1,i)**2+vecs(2,i)**2+vecs(3,i)**2)
	!enddo 
	*/

	return;
}




void Rot_Mat(float alpha, int axis, matrix Rotmat){
	//int  intent(in) :: axis //  the axis to rotate around 1,2,3 => x,y,z
	//float intent(in) :: alpha //  the angle to rotate
	//float dimension(3,3) :: Rotmat //  Rotation matrix in 3 dimensions

	switch (axis) {
	case 1:	//  x axis
		Rotmat=0;
		Rotmat(1,1)=1;
		Rotmat(2,2)=cos(alpha);
		Rotmat(3,3)=cos(alpha);
		Rotmat(3,2)=-1*sin(alpha);
		Rotmat(2,3)=sin(alpha);

	case 2: //  y axis
		Rotmat=0;
		Rotmat(1,1)=cos(alpha);
		Rotmat(2,2)=1;
		Rotmat(3,3)=cos(alpha);
		Rotmat(3,1)=-1*sin(alpha);
		Rotmat(1,3)=sin(alpha);

	case 3: //  z axis
		Rotmat=0;
		Rotmat(1,1)=cos(alpha);
		Rotmat(2,2)=cos(alpha);
		Rotmat(3,3)=1;
		Rotmat(1,2)=-1*sin(alpha);
		Rotmat(2,1)=sin(alpha);
	}	
	return;
} // end subroutine Rot_Mat

void make_field(nr_fields,fielddim_xy,fielddim_z ,n ,vecs ,y , RF){
	/*
	integer :: nr_fields,fielddim_xy,fielddim_z,i,j,k,l,n,linepoint
	real,dimension(3,nr_fields),intent(in):: vecs
	real,dimension(fielddim_xy,fielddim_xy,fielddim_z) :: RF
	real,dimension(3) :: testvec
	real :: linecoord,maximum,factor,minimum,SPmin,temp
	real,dimension(n,nr_fields) :: y	
	*/
	int nr_fields,fielddim_xy,fielddim_z,i,j,k,l,n,linepoint;
	float3 vecs[nr_fields];
	float *RF; //fielddim_xy,fielddim_xy,fielddim_z) :: RF
	float3 testvec;
	float linecoord,maximum,factor,minimum,SPmin,temp;
	float *y; //(n,nr_fields) 



	RF=0;
	factor=(n/2-1)/(sqrt(2.0*(fielddim_xy*fielddim_xy)+fielddim_z*fielddim_z));

	/// XXX TODO in OpenCL
	for(int l=0; l<nr_fields; l++) // about 1000 iters
		for(int i=0; i<fielddim_xy; i++) // from 64 to 1024
			for(int j=0; j<fielddim_xy; j++) // 
				for(int k=0; k<fielddim_z; k++)  // from 64 to 1024
				{				
					linecoord=-(i*vecs(1,l))-(j*vecs(2,l))-(k*vecs(3,l));
					//!	linecoord=-i*testvec(1)-j*testvec(2)-k*testvec(3)

					linepoint = NINT(linecoord)+1+n*0.5;



					RF(i,j,k)=RF(i,j,k)+y(linepoint,l);
				}
	}
		}
	}

	temp=nr_fields;
	RF=RF/(SQRT(temp));

} // subroutine make_field
