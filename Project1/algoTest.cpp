#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.hpp>

#include <algorithm>
#include <vector>
#include <iterator>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cl;

const int COLS = 3;
const int ROWS = 3;
const int COLSconv = COLS - 2;
const int ROWSconv = ROWS - 2;
const string IMAGE_PATH = "gt.jpg";

Mat input;
Mat defaultOutput = Mat::zeros(ROWS - 2, COLS - 2, CV_8UC1);

Mat getImage(string path, int rows, int cols);
int getIntensityMatrix(int* intensityMat, Mat image, int rows, int cols);
void setZero(int* matrix, int rows, int cols);
void matPrint(int* matrix, int rows, int cols);
double doConvSc(int* intensityMat, int* filtredMat, int* kern, int rows, int cols);
double doConvCl(int* host_a, int* cOpenCL, int* kern, unsigned long long numBytes, int rows, int cols);

int main() {
	int* intensityMat = new int[ROWS * COLS];
	setZero(intensityMat, ROWS, COLS);

	int* filtredMat = new int[ROWSconv * COLSconv];
	int* filtredMatCl = new int[ROWSconv * COLSconv];
	setZero(filtredMat, ROWSconv, COLSconv);
	setZero(filtredMatCl, ROWSconv, COLSconv);

	//���� ��� ������
	int* kern = new int[9];
	for (int i = 0; i < 9; i++)
	{
		kern[i] = rand() % 2;
	}
	//�������� ���������� ��������� ������
	if ((intensityMat == NULL) && (filtredMat == NULL) && (filtredMatCl == NULL)) {
		cout << "Error: can't allocate memory for the array" << endl;
		return -1;
	}

	input = getImage(IMAGE_PATH, ROWS, COLS);
	//�������� ���������� ���������� �����������
	if (!input.data) {
		cout << "Error: the image wasn't correctly loaded." << endl;
		return -2;
	}

	//imshow("Input", input);
	//����� kern
	getIntensityMatrix(intensityMat, input, ROWS, COLS);
	for (int i = 0; i < 9; i++) {
		cout << kern[i] << " ";
		if (((i + 1) % 3) == 0) {
			cout << endl;
		}
	}
	cout << "***" << endl;
	cout << "Size: " << COLS << "x" << ROWS << endl;
	cout << "Intensity mat" << endl;
	matPrint(intensityMat, ROWS, COLS);
	double defaultTime = doConvSc(intensityMat, filtredMat, kern, ROWSconv, COLSconv);
	cout << "filtred mat:" << endl;
	matPrint(filtredMat, ROWSconv, COLSconv);

	cout << "CPU multiplication time taken: " << defaultTime << " sec" << endl;

	//Mat resultIMG = Mat(ROWSconv, COLSconv, CV_32SC1, (uchar*)filtredMat);
	//imwrite("result.jpg", resultIMG);
	//defaultOutput = getImage("result.jpg", ROWSconv, COLSconv);
	//imshow("Output", defaultOutput);
	//cout << filtredMat[0];
	cout << "DONE" << endl;
	//waitKey(0);
	system("pause");
	return 0;
}

Mat getImage(string path, int rows, int cols) {
	// ������ ����������� � ���������� ��� � ������ �����������, � openCV ������ �������� � BGR �������
	Mat image = imread(path, IMREAD_COLOR);
	resize(image, image, { cols, rows }, INTER_NEAREST);
	return image;
}

int getIntensityMatrix(int* intensityMat, Mat image, int rows, int cols) {
	// We iterate over all pixels of the image
	for (int r = 0; r < rows; r++) {
		// We obtain a pointer to the beginning of row r
		Vec3b* pixelColor = image.ptr<Vec3b>(r);
		for (int c = 0; c < cols; c++) {
			intensityMat[r * cols + c] = (pixelColor[c][0] +
				pixelColor[c][1] + pixelColor[c][2]) / 3;
		}
	}
	return *intensityMat;
}


double doConvSc(int* intensityMat, int* filtredMat, int* kern, int rows, int cols)
{
	double start = omp_get_wtime();
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			cout << row << " " << col << "rowcol" << endl;
			for (int r_ = -1; r_ <= 1; r_++) {
				for (int c_ = -1; c_ <= 1; c_++) {
					cout << "index: " << (row + r_ + 1) * (cols + 2) + col + c_ + 1 << endl;
					cout << intensityMat[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)] << endl;
					filtredMat[row * cols + col] += intensityMat[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)];
					cout << "filtred mat at " << (row + r_ + 1) * (cols + 2) + col + c_ + 1 << " += " << intensityMat[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)] << " = " << filtredMat[row * cols + col] << endl;
				}
			}
		}
	}
	double end = omp_get_wtime();
	return (end - start);
}



void matPrint(int* matrix, int rows, int cols) {
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			cout << matrix[r * cols + c] << " ";
		}
		cout << endl;
	}
}
void setZero(int* matrix, int rows, int cols) {
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			matrix[r * cols + c] = 0;
		}
	}
}
bool matrixComp(int* a, int* b)
{
	bool if_equal = true;
	for (int i = 0; i < ROWSconv; i++)
	{
		for (int j = 0; j < COLSconv; j++)
		{
			if (a[i * COLSconv + j] != b[i * COLSconv + j])
			{
				if_equal = false;
				break;
			}
		}
	}
	return if_equal;
}