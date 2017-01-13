// Calc_Line_matrix_v1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
using namespace std;
#include <mkl_lapacke.h>



int Fill_corr_matrix(double a, double * matrix, int linelength)
{
	/* This fills the correlation matrix with the correlation value from each point to each point so i and j  are the points on the line and fabs(i-j) is there distance to each other
	This is a matrix of size linelength^2 because it needs the correlation from each point to each. This matrix is then decomposed with the lapack routine dysev into its eigenvalues and 
	eigenvectors These are later used to generate the Random field.
	
	The correlation that is plugged into this function has to be the appropriate 1D correlation function for generating the correct 3D field (the formula for creating the 1D correlation 
	for a given 3D correlation is C_1D=d/dr[r*C_3D(r)]   )*/

	int i, j;
	
	for (i = 0; i < linelength; i++) {
		for (j = i; j < linelength; j++) {	//this goes from i since the matrix is symetric and lapack only requires the upper diagonal triangle to be filled
			matrix[i*linelength+j] = ((a*a-2*fabs(i-j)*fabs(i-j))*exp(-((fabs(i - j) / a)*(fabs(i-j)/a))))/(a*a);		//this example is for C3D=exp(-(r/a)^2)
		}

	}
	return 0;




}



int main(int argc, char *argv[])
{
	int linelength,info,i,j; //the length off the line that should be generated with this correlation function, This length should be at least 2*sqrt(2)*diagonal_length in the case of a cubical domain
	string filename; //the filename in which the result is supposed to be saved
	double a; //the length parameter of the correlation function, can be extended for other correlation functions
	double *Corr_matrix,*eigen;
	double temp,sum;
	double max_eigen; //the maximum eigenvalue
	std::fstream f,g;
	


	linelength = atoi(argv[1]);
	filename = argv[2];
	a = atof(argv[3]);

	Corr_matrix = (double*)malloc(sizeof(double)*(linelength*linelength));
	eigen = (double*)malloc(sizeof(double)*(linelength));

	Fill_corr_matrix(a, Corr_matrix,linelength);	//fills the correlation matrix

	info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', linelength, Corr_matrix, linelength, eigen);
	//cout << info << endl;
	max_eigen = 0;
	//finds the eigenvalue that is largest 
	for (i = 0; i < linelength; i++) {
		if (max_eigen < fabs(eigen[i])) max_eigen = fabs(eigen[i]);
	}




	//Here we write the eigenvectors, appropriately scaled by the eigenvalues, into a file that can be used later by our RF generation code
	f.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
	f.write((char*) &linelength, sizeof(int)); //write the linelength to file 
	

	for (j = 0; j < linelength; j++) {
		for (i = 0; i < linelength; i++) {

			if (fabs(eigen[i]) < 1e-8*max_eigen) temp = 0;
			else temp = (Corr_matrix[j*linelength + i] * sqrt(eigen[i]));
			f.write((char*) &temp, sizeof(double));
		}

	}
	f.close();

	//cout << info << endl;
	sum = 0;
	for (i = 0; i < linelength; i++) {
		sum += eigen[i];
		//cout << eigen[i] << endl;
	}
	cout << "The sum of the eigenvalues, which should be the variance of a line generated is: " << sum << endl;

	//filename = "Finished.dat";
	//f.open(filename, std::ios::out | std::ios::trunc);
	//for (i = 0; i < linelength; i++) {
	//	for (j = 0; j < linelength; j++) {
	//			if (fabs(eigen[i]) < 1e-7*max_eigen) temp = 0;
	//			else temp = (Corr_matrix[j*linelength + i] * sqrt(eigen[i]));
	//			f << std::fixed << std::setprecision(8) <<temp << endl;
	//		}
	//	}

	//f.close();



	free(eigen);
	free(Corr_matrix);
}

