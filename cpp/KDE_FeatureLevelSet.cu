#include <iostream>
#include <stdio.h>
#include <cassert>
#include <math.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <assert.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <algorithm>
#include <time.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkStreamTracer.h>
#include <vtkPolyDataReader.h>
#include <vtkIdTypeArray.h>
#include <vtkIdList.h>
#include <vtkGradientFilter.h>
#include <vtkAssignAttribute.h>
#include <vtkStructuredGridWriter.h>
#include <vtkXMLPolyDataWriter.h>


#define PI       3.14159265358979323846 

using namespace std;


/* CUDA kernel functions */
__global__ void averageKernel(float *input_values, float *output_values);
__global__ void kdeKernel(float *input_values, float *output_values);
__global__ void normalDensityKernel(float *input_values, float *output_values);
__global__ void kdeKernelRotation(float *input_values, float *output_values);
__global__ void kdeSigmodKernel(float *input_values, float *output_values);
__global__ void kdeGaussianKernel(float *input_values, float *output_values);
__global__ void convolutionKernel(float *input_values, float *output_values);
__global__ void distanceKernel_Min(float *firstAttr, float* secondAttr, float* thresholds, float *distanceField);
__global__ void velocityAverageKernel(float *input_values, float *output_values);

/* Helper function for using CUDA to perform the computation in parallel */
cudaError_t computeKDE_Cuda(float *input_values, float * output_values, int size);
cudaError_t computeConvolutionKernel_Cuda(float *input_values, float * output_values, int size);
cudaError_t computeDistanceField_Cuda(float *firstAttr, float *secondAttr, float threshold[4], float *distanceField, int size);
cudaError_t computeAverageVelocity_Cuda(float *input_values, float * output_values, int size);


/* Ultility functions */
float computeStandardDeviation();
void outputVelocityToVTK(string outputFileName, float* velocityArray);
void outputDataToVTK(string outputFileName, float* values);
void outputDataToVTK(string outputFileName, vtkSmartPointer<vtkDataArray> velocity, vtkSmartPointer<vtkDataArray> attributes);
void outputDataToVTK(string outputFileName, float* values1, vtkSmartPointer<vtkDataArray> values2);
vtkSmartPointer<vtkStructuredPoints> readVTKInputData(string inputFilename);

/* Global variables */
int dims[3]; // Dimension of the flow data
double spacing[3]; // Grid spacing
double origin[3]; // The origin of the data
float xMin, xMax, yMin, yMax, zMin, zMax;  

// Output a velocity field to a VTK file in the STRUCTURED_POINTS format
void outputVelocityToVTK(string outputFileName, float* velocityArray)
{
	FILE * vtkFileWriter = fopen(outputFileName.c_str(), "w");
	fprintf(vtkFileWriter, "# vtk DataFile Version 3.0\n");
	fprintf(vtkFileWriter, "Volume example\n");
	fprintf(vtkFileWriter, "ASCII\n");
	fprintf(vtkFileWriter, "DATASET STRUCTURED_POINTS\n");
	fprintf(vtkFileWriter, "DIMENSIONS %d %d %d  \n", dims[0], dims[1], dims[2]);
	fprintf(vtkFileWriter, "ASPECT_RATIO %f %f %f  \n", spacing[0], spacing[1], spacing[2]);
	fprintf(vtkFileWriter, "ORIGIN %f %f %f  \n", origin[0], origin[1], origin[2]);
	fprintf(vtkFileWriter, "POINT_DATA %d  \n", dims[0] * dims[1] * dims[2]);
	fprintf(vtkFileWriter, "VECTORS velocity float\n");
	int tupleIdx;


	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				fprintf(vtkFileWriter, "%f %f %f\n", velocityArray[tupleIdx * 3], velocityArray[tupleIdx * 3 + 1], velocityArray[tupleIdx * 3 + 2]);
			}
	fclose(vtkFileWriter);
}

// Output an attribute field to a VTK file in the STRUCTURED_POINTS format
void outputDataToVTK(string outputFileName, float* values) {
	FILE * vtkFileWriter = fopen(outputFileName.c_str(), "w");
	fprintf(vtkFileWriter, "# vtk DataFile Version 3.0\n");
	fprintf(vtkFileWriter, "Volume example\n");
	fprintf(vtkFileWriter, "ASCII\n");
	fprintf(vtkFileWriter, "DATASET STRUCTURED_POINTS\n");
	fprintf(vtkFileWriter, "DIMENSIONS %d %d %d  \n", dims[0], dims[1], dims[2]);
	fprintf(vtkFileWriter, "ASPECT_RATIO %f %f %f  \n", spacing[0], spacing[1], spacing[2]);
	fprintf(vtkFileWriter, "ORIGIN %f %f %f  \n", origin[0], origin[1], origin[2]);
	fprintf(vtkFileWriter, "POINT_DATA %d  \n", dims[0] * dims[1] * dims[2]);
	fprintf(vtkFileWriter, "SCALARS KDE float 1\n");
	fprintf(vtkFileWriter, "LOOKUP_TABLE attribute_table\n");
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				//if (k == 0 || j == 0 || i == 0 || k == dims[2]-1 || j == dims[1] - 1 || i == dims[0] - 1)
				//	fprintf(vtkFileWriter, "-1\n");
				//else
				fprintf(vtkFileWriter, "%f\n", values[k * dims[0] * dims[1] + j * dims[0] + i]);
			}

	fclose(vtkFileWriter);
}

// Output both velocity and attribute fields to a VTK file
void outputDataToVTK(string outputFileName, vtkSmartPointer<vtkDataArray> velocity, vtkSmartPointer<vtkDataArray> attributes) {
	FILE * vtkFileWriter = fopen(outputFileName.c_str(), "w");
	fprintf(vtkFileWriter, "# vtk DataFile Version 3.0\n");
	fprintf(vtkFileWriter, "Volume example\n");
	fprintf(vtkFileWriter, "ASCII\n");
	fprintf(vtkFileWriter, "DATASET STRUCTURED_POINTS\n");
	fprintf(vtkFileWriter, "DIMENSIONS %d %d %d  \n", dims[0], dims[1], dims[2]);
	fprintf(vtkFileWriter, "ASPECT_RATIO %f %f %f  \n", spacing[0], spacing[1], spacing[2]);
	fprintf(vtkFileWriter, "ORIGIN %f %f %f  \n", origin[0], origin[1], origin[2]);
	fprintf(vtkFileWriter, "POINT_DATA %d  \n", dims[0] * dims[1] * dims[2]);
	fprintf(vtkFileWriter, "VECTORS velocity float\n");
	int tupleIdx;


	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				fprintf(vtkFileWriter, "%f %f %f\n", velocity->GetTuple(tupleIdx)[0], velocity->GetTuple(tupleIdx)[1], velocity->GetTuple(tupleIdx)[2]);
			}

	fprintf(vtkFileWriter, "SCALARS something float 1\n");
	fprintf(vtkFileWriter, "LOOKUP_TABLE something\n");

	int tmpCount = 0;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;

				fprintf(vtkFileWriter, "%f\n", attributes->GetTuple(tupleIdx)[0]);
				tmpCount++;
			}
	std::cout << "tmpCount " << tmpCount << std::endl;


	fclose(vtkFileWriter);
}

// Output multiple attribute fields to a VTK file
void outputDataToVTK(string outputFileName, float* values1, vtkSmartPointer<vtkDataArray> values2) {
	FILE * vtkFileWriter = fopen(outputFileName.c_str(), "w");
	fprintf(vtkFileWriter, "# vtk DataFile Version 3.0\n");
	fprintf(vtkFileWriter, "Volume example\n");
	fprintf(vtkFileWriter, "ASCII\n");
	fprintf(vtkFileWriter, "DATASET STRUCTURED_POINTS\n");
	fprintf(vtkFileWriter, "DIMENSIONS %d %d %d  \n", dims[0], dims[1], dims[2]);
	fprintf(vtkFileWriter, "ASPECT_RATIO %f %f %f  \n", spacing[0], spacing[1], spacing[2]);
	fprintf(vtkFileWriter, "ORIGIN %f %f %f  \n", origin[0], origin[1], origin[2]);
	fprintf(vtkFileWriter, "POINT_DATA %d  \n", dims[0] * dims[1] * dims[2]);

	fprintf(vtkFileWriter, "SCALARS Shearing float 1\n");
	fprintf(vtkFileWriter, "LOOKUP_TABLE something\n");
	int tupleIdx;

	int tmpCount = 0;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;

				fprintf(vtkFileWriter, "%f\n", values2->GetTuple(tupleIdx)[0]);
				tmpCount++;
			}
	std::cout << "tmpCount " << tmpCount << std::endl;

	fprintf(vtkFileWriter, "SCALARS KDE float 1\n");
	fprintf(vtkFileWriter, "LOOKUP_TABLE attribute_table\n");

	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				fprintf(vtkFileWriter, "%f\n", values1[k * dims[0] * dims[1] + j * dims[0] + i]);

			}

	fclose(vtkFileWriter);
}


// Read the VTK file which has velocity and Q attribtues
vtkSmartPointer<vtkStructuredPoints> readVTKInputData(string inputFilename)
{
	vtkSmartPointer<vtkStructuredPointsReader> reader =
		vtkSmartPointer<vtkStructuredPointsReader>::New();
	reader->SetFileName(inputFilename.c_str());
	reader->Update();
	vtkSmartPointer<vtkStructuredPoints> structurePoint = reader->GetOutput();

	// Get dimension
	structurePoint->GetDimensions(dims);

	// Get spacing
	structurePoint->GetSpacing(spacing);

	// Get the origin
	structurePoint->GetOrigin(origin);

	// Initialization
	xMin = origin[0]; xMax = xMin + (dims[0] - 1) * spacing[0];
	yMin = origin[1]; yMax = yMin + (dims[1] - 1) * spacing[1];
	zMin = origin[2]; zMax = zMin + (dims[2] - 1) * spacing[2];

	return structurePoint;
}

// Helper function for using CUDA to compute the feature level set of the two attributes in parallel 
int computeFeatureLevelSet(string inputVTKFile, string fieldName_1, string fieldName_2, string outputDistanceFieldFileName) {
	vtkSmartPointer<vtkStructuredPoints> structurePoints = readVTKInputData(inputVTKFile);
	vtkSmartPointer<vtkDataArray> firstAttributeVTK = structurePoints->GetPointData()->GetArray(fieldName_1.c_str());
	vtkSmartPointer<vtkDataArray> secondAttributeVTK = structurePoints->GetPointData()->GetArray(fieldName_2.c_str());

	// Covert vtkDataArray to the regular array so that we can process with cuda
	int tupleIdx;
	int size = dims[0] * dims[1] * dims[2];
	float* firstAttribute = (float*)malloc(size * sizeof(float));
	float* secondAttribute = (float*)malloc(size * sizeof(float));
	float* distance_values = (float*)malloc(size * sizeof(float));

	float firstMin = FLT_MAX, secondMin = FLT_MAX;
	float firstMax = FLT_MIN, secondMax = FLT_MIN;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				firstAttribute[tupleIdx] = firstAttributeVTK->GetTuple(tupleIdx)[0];

				if (firstAttribute[tupleIdx] > firstMax)
					firstMax = firstAttribute[tupleIdx];
				if (firstAttribute[tupleIdx] < firstMin)
					firstMin = firstAttribute[tupleIdx];

				secondAttribute[tupleIdx] = secondAttributeVTK->GetTuple(tupleIdx)[0];

				if (secondAttribute[tupleIdx] > secondMax)
					secondMax = secondAttribute[tupleIdx];
				if (firstAttribute[tupleIdx] < secondMin)
					secondMin = secondAttribute[tupleIdx];
			}
	// Normalize
	float firstDiff = firstMax - firstMin;
	float secondDiff = secondMax - secondMin;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				firstAttribute[tupleIdx] = 100 * (firstAttribute[tupleIdx] - firstMin) / firstDiff;
				secondAttribute[tupleIdx] = 100 * (secondAttribute[tupleIdx] - secondMin) / secondDiff;
			}

	float thresholds[4];

	// set threshold range for the first attribute
	thresholds[0] = 100 * (0 - firstMin) / firstDiff; //e.g 0->0.05 for Q
	thresholds[1] = 100 * (0.1 - firstMin) / firstDiff;

	// set threshold range for the second attribute
	thresholds[2] = 100 * (0 - secondMin) / secondDiff;
	thresholds[3] = 100 * (0.4 - secondMin) / secondDiff;

	cudaError_t cudaStatus = computeDistanceField_Cuda(firstAttribute, secondAttribute, thresholds, distance_values, size);

	outputDataToVTK(outputDistanceFieldFileName, distance_values);
	free(firstAttribute);
	free(secondAttribute);
	free(distance_values);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


// Helper function for using CUDA to compute the KDE in parallel 
int computeSingleAttribute_KDE(string velocityVTKInputFile, string kdeAttributeFieldName, string attributeFieldName, string outputFileName) {

	vtkSmartPointer<vtkStructuredPoints> structurePoints = readVTKInputData(velocityVTKInputFile);
	vtkSmartPointer<vtkDataArray> attrValues = structurePoints->GetPointData()->GetArray(attributeFieldName.c_str());

	vtkSmartPointer<vtkDataArray> kdeAttrValues = structurePoints->GetPointData()->GetArray(kdeAttributeFieldName.c_str()) ;
	

	// Covert vtkDataArray to the regular array so that we can process with cuda
	int tupleIdx;
	int size = dims[0] * dims[1] * dims[2];
	float* input_values = (float*)malloc(size * sizeof(float));
	float* output_values = (float*)malloc(size * sizeof(float));
	double* q;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				input_values[tupleIdx] = kdeAttrValues->GetTuple(tupleIdx)[0];
			}
	cudaError_t cudaStatus = computeKDE_Cuda(input_values, output_values, size);

	outputDataToVTK(outputFileName, output_values, attrValues);
	free(input_values);
	free(output_values);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}

/* The program expects 5 arguments:
		1. The feature name (i.e kde or feature_level_set)
		2. The velocity input file in the vtk format
		3. The first (or kde) attribute field name
		4. The second attribute field name
		5. The output file
*/
int main(int argc, char *argv[])
{
	
	if (argc < 6) { 
		std::cerr << "Usage: kde/feature_level_set vtk_input_file attribute_name_1 attribute_name_2 vkt_output_file" << std::endl;
		return 1;
	}

	string feature = argv[1];
	string velocityVTKInputFile = argv[2];
	string attributeName_1 = argv[3];
	string attributeName_2 = argv[4];
	string outputFileName = argv[5];

	if (feature == "kde") {
		computeSingleAttribute_KDE(velocityVTKInputFile, attributeName_1, attributeName_2, outputFileName);
	}
	else if (feature == "feature_level_set") {
		computeFeatureLevelSet(velocityVTKInputFile, attributeName_1, attributeName_2, outputFileName);
	}
	else {
		std::cerr << "The requested feature is not supported!";
	}
	
	return 0;
}


__global__ void convolutionKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int smooth_kernel_size = 10;
	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	float lower_bound = 0;
	float higher_bound = 0.1;

	int count = 0;
	float sum = 0;

	int mink = k - smooth_kernel_size;
	int maxk = k + smooth_kernel_size;
	int minj = j - smooth_kernel_size;
	int maxj = j + smooth_kernel_size;
	int mini = i - smooth_kernel_size;
	int maxi = i + smooth_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;
							sum = sum + input_values[tupleIdx];
							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}


__global__ void velocityAverageKernel(float *input_values, float *output_values) {
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int smooth_kernel_size = 6;
	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	int count = 0;
	float sumVX = 0, sumVY = 0, sumVZ = 0;

	int mink = k - smooth_kernel_size;
	int maxk = k + smooth_kernel_size;
	int minj = j - smooth_kernel_size;
	int maxj = j + smooth_kernel_size;
	int mini = i - smooth_kernel_size;
	int maxi = i + smooth_kernel_size;

	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;
							sumVX = sumVX + input_values[tupleIdx * 3];
							sumVY = sumVY + input_values[tupleIdx * 3 + 1];
							sumVZ = sumVZ + input_values[tupleIdx * 3 + 2];
							count++;
						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx * 3] = sumVX / count;
	output_values[tupleIdx * 3 + 1] = sumVY / count;
	output_values[tupleIdx * 3 + 2] = sumVZ / count;
}

__global__ void kdeKernelRotation(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 15;
	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	float lower_bound = 0.0;
	float higher_bound = 0.25;

	int count = 0;
	float sum = 0;

	int mink = k - kde_kernel_size;
	int maxk = k + kde_kernel_size;
	int minj = j - 3 * kde_kernel_size;
	int maxj = j + 3 * kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;

	int rotatedJ, rotatedK;
	float rotationDegreeRadius = 30 * PI / 180.0;
	int tmpj, tmpk;
	for (int subi = mini; subi < maxi; subi++)
		if (subi >= 0 && subi < dims[0]) {
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subk = mink; subk < maxk; subk++)
						if (subk >= 0 && subk < dims[2])
						{
							tmpj = subj - minj;
							tmpk = subk - mink;
							rotatedJ = int(round(tmpj * cos(rotationDegreeRadius) - tmpk * sin(rotationDegreeRadius)));
							rotatedK = int(round(tmpj * sin(rotationDegreeRadius) + tmpk * cos(rotationDegreeRadius)));

							tmpj = minj + rotatedJ;
							tmpk = mink + rotatedK;
							if (tmpj >= 0 && tmpj < dims[1] && tmpk >= 0 && tmpk < dims[2]) {
								tupleIdx = tmpk * dims[0] * dims[1] + tmpj * dims[0] + subi;

								if (input_values[tupleIdx] >= lower_bound && input_values[tupleIdx] <= higher_bound) {
									sum = sum + 1;
								}

								count++;
							}

						}
		}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}


__global__ void averageKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 10;
	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	int count = 0;
	float sum = 0;

	int mink = k - 5;
	int maxk = k + 5;
	int minj = j - 3 * kde_kernel_size;
	int maxj = j + 3 * kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;


							sum = sum + input_values[tupleIdx];


							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}


__global__ void kdeKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 8;
	int dims[3];

	dims[0] = 96;  //192;
	dims[1] = 128; // 128;
	dims[2] = 128; //128;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	float lower_bound = 0.139035;
	float higher_bound = 0.5793143510818481;

	int count = 0;
	float sum = 0;

	int mink = k - kde_kernel_size;
	int maxk = k + kde_kernel_size;
	int minj = j - 3 * kde_kernel_size;
	int maxj = j + 3 * kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;

							if (input_values[tupleIdx] >= lower_bound && input_values[tupleIdx] <= higher_bound) {
								sum = sum + 1;
							}

							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}

__global__ void normalDensityKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 8;
	int dims[3];

	dims[0] = 96;
	dims[1] = 128;
	dims[2] = 128;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	//float lower_bound = 0.1;
	//float higher_bound = 0.3;
	float threshold = 2.5;
	int count = 0;
	float sum = 0;

	int mink = k - kde_kernel_size;
	int maxk = k + kde_kernel_size;
	int minj = j - kde_kernel_size;
	int maxj = j + kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;

							if (input_values[tupleIdx] >= threshold) {
								sum = sum + 1;
							}

							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}

__global__ void kdeGaussianKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 8;
	int dims[3];

	dims[0] = 192;
	dims[1] = 128;
	dims[2] = 128;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	float lower_bound = 0.1;
	float higher_bound = 0.3;

	int count = 0;
	float sum = 0;
	float w;
	float h = 0.56;

	int mink = k - kde_kernel_size;
	int maxk = k + kde_kernel_size;
	int minj = j - 3 * kde_kernel_size;
	int maxj = j + 3 * kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;

							if (input_values[tupleIdx] >= lower_bound && input_values[tupleIdx] <= higher_bound) {

								sum = sum + powf(2.718, -0.01*((subk - k)*(subk - k) + (subj - j)*(subj - j) + (subi - i)*(subi - i)) / (2 * h*h));
							}

							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / (count*powf(h*sqrtf(2 * 3.14), 3));

	//output_values[tupleIdx] = sum;
}

__global__ void kdeSigmodKernel(float *input_values, float *output_values)
{
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int tupleIdx;
	int kde_kernel_size = 10;
	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];

	float lower_bound = 0;
	float higher_bound = 0.1;

	int count = 0;
	float sum = 0;

	int mink = k - kde_kernel_size;
	int maxk = k + kde_kernel_size;
	int minj = j - kde_kernel_size;
	int maxj = j + kde_kernel_size;
	int mini = i - kde_kernel_size;
	int maxi = i + kde_kernel_size;
	for (int subk = mink; subk < maxk; subk++)
		if (subk >= 0 && subk < dims[2])
			for (int subj = minj; subj < maxj; subj++)
				if (subj >= 0 && subj < dims[1])
					for (int subi = mini; subi < maxi; subi++)
						if (subi >= 0 && subi < dims[0])
						{
							tupleIdx = subk * dims[0] * dims[1] + subj * dims[0] + subi;

							if (input_values[tupleIdx] >= lower_bound && input_values[tupleIdx] <= higher_bound) {
								sum = sum + 1;
							}
							else if (input_values[tupleIdx] < lower_bound) {
								float x = input_values[tupleIdx] - lower_bound;
								if (x > -6)
									sum = sum + 1 / (1 + exp(-x));

							}
							else { // *q > higher_bound
								float x = input_values[tupleIdx] - higher_bound;
								if (x < 6)
									sum = sum + 1 / (1 + exp(-x));
							}

							count++;

						}
	tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
	output_values[tupleIdx] = sum / count;
}

__global__ void distanceKernel_Min(float *firstAttr, float* secondAttr, float* thresholds, float *distanceField) {
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	int dims[3];

	dims[0] = 192;
	dims[1] = 512;
	dims[2] = 512;

	int k = tIdx / (dims[0] * dims[1]);
	int j = (tIdx % (dims[0] * dims[1])) / dims[0];
	int i = (tIdx % (dims[0] * dims[1])) % dims[0];


	int tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;

	float firstDistance = 0, secondDistance;

	if (firstAttr[tupleIdx] >= thresholds[0] && firstAttr[tupleIdx] <= thresholds[1]) {
		firstDistance = 0;
	}
	else {
		firstDistance = min(fabs(firstAttr[tupleIdx] - thresholds[0]), fabs(firstAttr[tupleIdx] - thresholds[1]));
	}

	if (secondAttr[tupleIdx] >= thresholds[2] && secondAttr[tupleIdx] <= thresholds[3]) {
		secondDistance = 0;
	}
	else {
		secondDistance = min(fabs(secondAttr[tupleIdx] - thresholds[2]), fabs(secondAttr[tupleIdx] - thresholds[3]));;
	}

	distanceField[tupleIdx] = min(firstDistance, secondDistance);
	//distanceField[tupleIdx] = secondDistance;

}


// Compute KDE for one attribute at a time
cudaError_t computeKDE_Cuda(float *input_values, float * output_values, int size) {
	float *dev_input_values = 0;
	float *dev_output_values = 0;
	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	fprintf(stderr, "Total memory require %d \n", size * sizeof(float));
	cudaStatus = cudaMemcpy(dev_input_values, input_values, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Launch a kernel on the GPU
	int threadsPerBlock = 1024;
	dim3 dimBlock(threadsPerBlock);
	dim3 dimGrid(size / threadsPerBlock);
	cudaEventRecord(start);
	/* Choose from a list of available kernels. E.g: 
		normalDensityKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
		kdeGaussianKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
		averageKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
		kdeKernelRotation << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
		kdeSigmodKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
	*/

	kdeKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);
	
	cudaEventRecord(stop);



	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeKDE_Cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kdeKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_values, dev_output_values, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU = %.20lf seconds\n", milliseconds / 1000);

Error:
	cudaFree(dev_input_values);
	cudaFree(dev_output_values);

	return cudaStatus;
}

// Helper function for using CUDA to compute the convolution kernel in parallel.
cudaError_t computeConvolutionKernel_Cuda(float *input_values, float * output_values, int size) {
	float *dev_input_values = 0;
	float *dev_output_values = 0;
	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	fprintf(stderr, "Total memory require %d \n", size * sizeof(float));
	cudaStatus = cudaMemcpy(dev_input_values, input_values, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	// Launch a kernel on the GPU
	int threadsPerBlock = 1024;
	dim3 dimBlock(threadsPerBlock);
	dim3 dimGrid(size / threadsPerBlock);
	cudaEventRecord(start);
	convolutionKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);

	cudaEventRecord(stop);



	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kdeKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_values, dev_output_values, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU = %.20lf seconds\n", milliseconds / 1000);

Error:
	cudaFree(dev_input_values);
	cudaFree(dev_output_values);

	return cudaStatus;
}


// Helper function for using CUDA to compute the distance field in parallel.
cudaError_t computeDistanceField_Cuda(float *firstAttr, float *secondAttr, float thresholds[4], float *distanceField, int size) {
	float *dev_firstAttr = 0;
	float *dev_secondAttr = 0;
	float *dev_distanceField = 0;
	float *dev_thresholds = 0;
	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_firstAttr, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_secondAttr, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_distanceField, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thresholds, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	fprintf(stderr, "Total memory require %d \n", size * sizeof(float));
	cudaStatus = cudaMemcpy(dev_firstAttr, firstAttr, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_secondAttr, secondAttr, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thresholds, thresholds, 4 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	// Launch a kernel on the GPU.
	int threadsPerBlock = 1024;
	dim3 dimBlock(threadsPerBlock);
	dim3 dimGrid(size / threadsPerBlock);
	cudaEventRecord(start);
	distanceKernel_Min << <dimGrid, dimBlock >> > (dev_firstAttr, dev_secondAttr, dev_thresholds, dev_distanceField);

	cudaEventRecord(stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "distanceKernel_Min launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kdeKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(distanceField, dev_distanceField, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU = %.20lf seconds\n", milliseconds / 1000);

Error:
	cudaFree(dev_firstAttr);
	cudaFree(dev_secondAttr);
	cudaFree(dev_distanceField);

	return cudaStatus;
}

// Helper function for using CUDA to compute the average velocity in parallel.
cudaError_t computeAverageVelocity_Cuda(float *input_values, float * output_values, int size)
{
	float *dev_input_values = 0;
	float *dev_output_values = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output_values, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	fprintf(stderr, "Total memory require %d \n", size * sizeof(float));
	cudaStatus = cudaMemcpy(dev_input_values, input_values, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	// Launch a kernel on the GPU
	int threadsPerBlock = 1024;
	dim3 dimBlock(threadsPerBlock);
	dim3 dimGrid((size / 3) / threadsPerBlock);
	cudaEventRecord(start);
	velocityAverageKernel << <dimGrid, dimBlock >> > (dev_input_values, dev_output_values);

	cudaEventRecord(stop);



	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "velocityAverageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kdeKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_values, dev_output_values, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU = %.20lf seconds\n", milliseconds / 1000);

Error:
	cudaFree(dev_input_values);
	cudaFree(dev_output_values);

	return cudaStatus;
}

// Compute the standard deviation
float computeStandardDeviation(string inputVTKFile, string fieldName) {
	vtkSmartPointer<vtkStructuredPoints> attrPoints = readVTKInputData(inputVTKFile);
	vtkSmartPointer<vtkDataArray> attrValues = attrPoints->GetPointData()->GetArray(fieldName.c_str());
	int tupleIdx;
	double sum = 0;
	int N = dims[0] * dims[1] * dims[2];
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				sum += attrValues->GetTuple(tupleIdx)[0];
			}
	float mean = sum / N;
	sum = 0;
	for (int k = 0; k < dims[2]; k++)
		for (int j = 0; j < dims[1]; j++)
			for (int i = 0; i < dims[0]; i++) {
				tupleIdx = k * dims[0] * dims[1] + j * dims[0] + i;
				sum += pow((attrValues->GetTuple(tupleIdx)[0] - mean), 2);
			}
	float standardDeviation = sqrt(sum / N);
	float bandwidth = 1.06*standardDeviation*pow(N, -1 / 5);
	std::cout << "standardDeviation " << standardDeviation << std::endl;
	std::cout << "bandwidth " << bandwidth << std::endl;
	return standardDeviation;
}