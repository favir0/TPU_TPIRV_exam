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
#include <thread>

using namespace cv;
using namespace std;
using namespace cl;

const int THREAD_NUMBER = 6;
const int COLS = 4096;
const int ROWS = 4096;
const int COLSconv = COLS - 2;
const int ROWSconv = ROWS - 2;
const string IMAGE_PATH = "gt.jpg";

Mat input;
Mat defaultOutput = Mat::zeros(ROWS - 2, COLS - 2, CV_8UC1);


string get_program_text();
Program create_program(Context cont);
Device create_device();

bool matrixComp(int* a, int* b);
Mat getImage(string path, int rows, int cols);
int getIntensityMatrix(int* intensityMat, Mat image, int rows, int cols);
void setZero(int* matrix, int rows, int cols);
void matPrint(int* matrix, int rows, int cols);
double doConvSc(int* intensityMat, int* filtredMat, int* kern, int rows, int cols);
double doConvCl(int* host_a, int* cOpenCL, int* kern, unsigned long long numBytes, int rows, int cols);
void doConvTh(int* intensityMat, int* filtredMat, int* kern, int iter_, int rows, int cols);

int main() {
	thread threads[THREAD_NUMBER];

	int* intensityMat = new int[ROWS * COLS];
	setZero(intensityMat, ROWS, COLS);

	int* filtredMat = new int[ROWSconv * COLSconv];
	int* filtredMatCl = new int[ROWSconv * COLSconv];
	int* filtredMatTh = new int[ROWSconv * COLSconv];
	setZero(filtredMat, ROWSconv, COLSconv);
	setZero(filtredMatTh, ROWSconv, COLSconv);
	setZero(filtredMatCl, ROWSconv, COLSconv);

	//Ядро для свёртки
	int* kern = new int[9];
	for (int i = 0; i < 9; i++)
	{
		kern[i] = rand() % 2;
	}
	//Проверка успешности выделения памяти
	if ((intensityMat == NULL) && (filtredMat == NULL) && (filtredMatCl == NULL)) {
		cout << "Error: can't allocate memory for the array" << endl;
		return -1;
	}
	input = getImage(IMAGE_PATH, ROWS, COLS);
	//Проверка успешности считывания изображения
	if (!input.data) {
		cout << "Error: the image wasn't correctly loaded." << endl;
		return -2;
	}
	imshow("Input", input);
	//Вывод kern
	getIntensityMatrix(intensityMat, input, ROWS, COLS);
	for (int i = 0; i < 9; i++) {
		cout << kern[i] << " ";
		if (((i + 1) % 3) == 0) {
			cout << endl;
		}
	}
	cout << "***" << endl;
	cout << "Size: " << COLS << "x" << ROWS << endl;

	//Default
	double defaultTime = doConvSc(intensityMat, filtredMat, kern, ROWSconv, COLSconv);

	//OpenCL
	double openCLtime = doConvCl(intensityMat, filtredMatCl, kern, COLS * ROWS * sizeof(int), ROWSconv, COLSconv);

	//Thread
	double start = omp_get_wtime();
	for (int i = 0; i < THREAD_NUMBER; i++) {
		threads[i] = thread(doConvTh, intensityMat, filtredMatTh, kern, i, ROWSconv, COLSconv);
	}

	for (int i = 0; i < THREAD_NUMBER; i++) {
		threads[i].join();
	}
	double end = omp_get_wtime();
	double multiThreadTime = end - start;

	if (matrixComp(filtredMat, filtredMatCl) && matrixComp(filtredMat, filtredMatTh)) {
		cout << "Matrices are the same!" << endl;
	}
	else {
		cout << "Matrices AREN'T the same!!!" << endl;
		return 0;
	}

	cout << "CPU multiplication time taken: " << defaultTime << " sec" << endl;
	cout << THREAD_NUMBER << " threads CPU multiplication time taken: " << multiThreadTime << " sec" << endl;
	cout << "OpenCL multiplication time taken: " << openCLtime * 1000 << " ms" << endl;
	cout << "CPU_single / CPU_multi ratio: " << (defaultTime / multiThreadTime) << endl;
	cout << "CPU / OpenCL ratio: " << (defaultTime / openCLtime) << endl;
	cout << "Thread / OpenCL ratio: " << multiThreadTime / openCLtime << endl;

	Mat resultIMG = Mat(ROWSconv, COLSconv, CV_32SC1, (uchar*)filtredMat);
	imwrite("result.jpg", resultIMG);
	defaultOutput = getImage("result.jpg", ROWSconv, COLSconv);
	imshow("Output", defaultOutput);
	cout << filtredMat[0];
	cout << "DONE" << endl;
	waitKey(0);
	system("pause");
	return 0;
}

Device create_device() {
	//get all platforms (drivers)
	vector<Platform> all_platforms;
	Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	Platform default_platform = all_platforms[0];
	cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	//get default device of the default platform
	vector<Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cout << "Available devices: " << endl;
	for (int i = 0; i < all_devices.size(); i++) {
		long long memSize = round((double)(all_devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) / 1024.0 / 1024.0 / 1024.0);
		cout << i << ". " << all_devices[i].getInfo<CL_DEVICE_NAME>() << endl;
		cout << "\tGlobal mem size: " << memSize << " GiB" << endl;
		cout << "\tDriver version: " << all_devices[i].getInfo<CL_DRIVER_VERSION>() << endl;
		cout << "\tDevice version: " << all_devices[i].getInfo<CL_DEVICE_VERSION>() << endl;
		cout << "\tMax work group size: " << all_devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
		cout << "\tMax compute units: " << all_devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
	}
	cout << "Choose device (print number): 0";
	int inp;
	inp = 0;
	Device default_device = all_devices[inp];
	cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
	return default_device;
}

Program create_program(Context cont) {
	Program::Sources sources;
	string codeStr = get_program_text();
	sources.push_back({ codeStr.c_str(), codeStr.length() });
	Program program(cont, sources);
	return program;
}

string get_program_text() {
	cout << "Ifstream started : reading kernel code...";
	ifstream input_stream("examGPU.cl");
	if (!input_stream)
	{
		cout << endl << "Error with opening .cl file" << endl;
		exit(1);
	}
	string out = string((istreambuf_iterator<char>(input_stream)), istreambuf_iterator<char>());
	input_stream.close();
	cout << "Complete! Ifstream has closed." << endl;
	//cout << "Code: " << endl << out << endl;
	return out;
}

Mat getImage(string path, int rows, int cols) {
	// Импорт изображения и приведение его к нужной размерности, в openCV каналы хранятся в BGR формате
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

double doConvCl(int* host_a, int* cOpenCL, int* kern, unsigned long long numBytes, int rows, int cols) {

	cl_int err = 0;

	// Initialize device
	Device device = create_device();

	// Create context
	Context context({ device });

	// Create program
	Program program = create_program(context);

	// Build the program
	if (program.build({ device }) != CL_SUCCESS) {
		cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		exit(1);
	}

	// create buffers on the device
	Buffer buffer_A(context, CL_MEM_READ_ONLY, numBytes);
	Buffer buffer_KERN(context, CL_MEM_READ_WRITE, sizeof(int) * 9);
	Buffer buffer_C(context, CL_MEM_READ_WRITE, rows * cols * sizeof(int));

	// Create queue to which we will push commands for the device.
	CommandQueue queue(context, device);
	// Create kernel specification (ND range)
	NDRange global(cols, rows);
	NDRange local(32, 32);
	// Write arrays A and B to the device
	double start = omp_get_wtime();
	// Send a and b to buffer
	cout << "hello" << endl;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, numBytes, host_a);
	queue.enqueueWriteBuffer(buffer_KERN, CL_TRUE, 0, sizeof(int) * 9, kern);

	int* COLS_ptr = new int;
	*COLS_ptr = cols;
	int* ROWS_ptr = new int;
	*ROWS_ptr = rows;
	// Setup kernel and args
	Kernel conv(program, "doConv", &err);
	conv.setArg(0, buffer_A);
	conv.setArg(1, buffer_C);
	conv.setArg(2, buffer_KERN);
	conv.setArg(3, sizeof(int), ROWS_ptr);
	conv.setArg(4, sizeof(int), COLS_ptr);



	cout << "STATUS " << err << endl;
	// Start multiplication
	queue.enqueueNDRangeKernel(conv, cl::NullRange, global, cl::NullRange);
	queue.finish();
	// Send result from device to host
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, rows * cols * sizeof(int), cOpenCL);
	double end = omp_get_wtime();
	double elapsedTime = end - start;
	return elapsedTime;
}

double doConvSc(int* intensityMat, int* filtredMat, int* kern, int rows, int cols)
{
	double start = omp_get_wtime();
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			for (int r_ = -1; r_ <= 1; r_++) {
				for (int c_ = -1; c_ <= 1; c_++) {
					filtredMat[row * cols + col] += intensityMat[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)];
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

void doConvTh(int* intensityMat, int* filtredMat, int* kern, int iter_, int rows, int cols) {
	for (int row = iter_; row < rows; row += THREAD_NUMBER) {
		for (int col = 0; col < cols; col++) {
			for (int r_ = -1; r_ <= 1; r_++) {
				for (int c_ = -1; c_ <= 1; c_++) {
					filtredMat[row * cols + col] += intensityMat[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)];
				}
			}
		}
	}
}