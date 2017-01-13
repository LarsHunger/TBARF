// Check_corr_gauss.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>

using namespace std;

int main(int argc, char *argv[])
{
	int nr_fields;	//the number of fields supplied
	int fielddim;  // the field dimension supplied
	double* Fields;
	int i,size_one_field,j,k,l;
	int x_ref, y_ref, z_ref, ref_point;
	double temp,sum,r;
	double *Expectation,*Variance;
	double *Expectation_field, *Variance_field;	//the expectation and variance for one point in each field
	double *Correlation_field;
	fstream file;

	string basename;
	string filename;

	nr_fields = atoi(argv[1]);
	fielddim = atoi(argv[2]);
	basename = argv[3];
	
	size_one_field = fielddim*fielddim*fielddim;





	//read fields
	Fields = (double *)malloc(sizeof(double)*nr_fields*fielddim*fielddim*fielddim);	//holds all the fields
	for (i = 0; i < nr_fields; i++) {
		filename = basename + "_" + std::to_string(i) + ".dat";
		//std::cout << filename << std::endl;
		file.open(filename, std::ios::in | std::ios::binary);
		if (file.is_open()) {
			for (j = 0; j < fielddim; j++) {
				for (k = 0; k < fielddim; k++) {
					for (l = 0; l < fielddim; l++) {
						file.read((char*)&temp, sizeof(double));
						//std::cout << temp << std::endl;
						Fields[i*size_one_field + j*fielddim*fielddim + k*fielddim + l] = temp;
					}
				}

			}
		}
		else cout << "something stupid happend" << std::endl;

		file.close();

	
	}

	Expectation = (double *)malloc(sizeof(double)*nr_fields);	//holds expectation for all the fields
	//calculate the expectation value for each field 
	for (i = 0; i < nr_fields; i++) {
		sum = 0;
		for (j = 0; j < size_one_field; j++) {
			sum += Fields[i*size_one_field + j];
		}
		sum /= size_one_field;
		Expectation[i] = sum;
	
	}

	Variance = (double *)malloc(sizeof(double)*nr_fields);	//holds expectation for all the fields
																//calculate the expectation value for each field 
	for (i = 0; i < nr_fields; i++) {
		sum = 0;
		for (j = 0; j < size_one_field; j++) {
			sum += (Fields[i*size_one_field + j]-Expectation[i])*(Fields[i*size_one_field + j] - Expectation[i]);
		}
		sum /= size_one_field;
		Variance[i] = sum;

	}

	//print expectation and variance
	for (i = 0; i < nr_fields; i++) {
		std::cout << Expectation[i] << "     " << Variance[i] << endl;
	}

	/* Next part */
	Expectation_field = (double *)calloc(size_one_field, sizeof(double));
	Variance_field = (double *)calloc(size_one_field, sizeof(double) );
	Correlation_field = (double *)calloc(size_one_field, sizeof(double));

	/*Calculate expectation for each field coordinate*/

	
	for (j = 0; j < size_one_field; j++) {
		for (i = 0; i < nr_fields; i++) {
			Expectation_field[j] += Fields[i*size_one_field + j];
		}
		Expectation_field[j] /= nr_fields;
	}
		
	/*Calculate variance for each field coordinate*/
	for (j = 0; j < size_one_field; j++) {
		for (i = 0; i < nr_fields; i++) {
			Variance_field[j] += (Fields[i*size_one_field + j]-Expectation_field[j])*(Fields[i*size_one_field + j] - Expectation_field[j]);
		}
		Variance_field[j] /= nr_fields;
	}
	filename = "Expectation_variance_field.dat";
	file.open(filename, std::ios::out | std::ios::trunc);
	for (j = 0; j < fielddim; j++) {
		for (k = 0; k < fielddim; k++) {
			for (l = 0; l < fielddim; l++) {
				//r = sqrt((j - x_ref)*(j - x_ref) + (k - y_ref)*(k - y_ref) + (l - z_ref)*(l - z_ref));
				file << std::fixed << std::setprecision(8) << Expectation_field[j*fielddim*fielddim+k*fielddim+l]  << "    " << Variance_field[j*fielddim*fielddim+k*fielddim+l] << endl;
			}
		}

	}
	
	
	file.close();
	
	
	
	
	//reference point is 64,64,64 ... calculate correlation to reference point and calculate distance for each
	x_ref = 64;
	y_ref = 64;
	z_ref = 64;
	ref_point = fielddim*fielddim*x_ref + fielddim*y_ref + z_ref;

		
	for (j = 0; j < size_one_field; j++) {
		for (i = 0; i < nr_fields; i++) {
			Correlation_field[j] += Fields[i*size_one_field + j] * Fields[i*size_one_field + ref_point];
		}
		Correlation_field[j] =Correlation_field[j]/nr_fields;	//expectation of X_s*X_t
		Correlation_field[j] =Correlation_field[j]-(Expectation_field[j] * Expectation_field[ref_point]);	//- mu(x)*mu(t)
		Correlation_field[j]=Correlation_field[j]/(sqrt(Variance_field[j]) * sqrt(Variance_field[ref_point]));
	}
	filename = "Correlation.dat";
	file.open(filename, std::ios::out | std::ios::trunc);
	for (j = 0; j < fielddim; j++) {
		for (k = 0; k < fielddim; k++) {
			for (l = 0; l < fielddim; l++) {
				r = sqrt((j - x_ref)*(j - x_ref) + (k - y_ref)*(k - y_ref) + (l - z_ref)*(l - z_ref));
				file << std::fixed << std::setprecision(8) << Correlation_field[j*fielddim*fielddim+k*fielddim+l]  << "    " << r  << endl;
			}
		}

	}
	file.close();



	free(Correlation_field);
	free(Expectation_field);
	free(Variance_field);
	free(Variance);
	free(Expectation);
	free(Fields);
	return 0;
}

