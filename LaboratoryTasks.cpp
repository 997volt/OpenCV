#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <sstream>
#include <string>
#include <list>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

enum {
	EVENT_MOUSEMOVE = 0,
	EVENT_LBUTTONDOWN = 1,
	EVENT_RBUTTONDOWN = 2,
	EVENT_MBUTTONDOWN = 3,
	EVENT_LBUTTONUP = 4,
	EVENT_RBUTTONUP = 5,
	EVENT_MBUTTONUP = 6,
	EVENT_LBUTTONDBLCLK = 7,
	EVENT_RBUTTONDBLCLK = 8,
	EVENT_MBUTTONDBLCLK = 9,
	EVENT_MOUSEWHEEL = 10,
	EVENT_MOUSEHWHEEL = 11
};



static void on_trackbar(int, void*)
{

}

#pragma region doWszystkichStare

cv::Mat image, Px, Py, Sx, Sy;

int mx, my;

static void onMouse(int event, int x, int y, int, void*)
{
	if (event == 4)
	{
		std::cout << std::endl << "lewy" << std::endl;
		cv::rectangle(image, cv::Point(x - 20, y - 20), cv::Point(x + 20, y + 20), cv::Scalar(120, 0, 155), 5);
	}

	if (event == 5)
	{
		std::cout << std::endl << "prawy" << std::endl;
		cv::circle(image, cv::Point(x, y), 25, cv::Scalar(25, 140, 60), 5);
	}

	cv::imshow("window", image);
}

static void onMouse2(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		mx = x;
		my = y;
	}

	if (event == EVENT_LBUTTONUP)
	{
		cv::rectangle(image, cv::Point(mx, my), cv::Point(x, y), cv::Scalar(120, 0, 155), 5);
			cv::imshow("window", image);
	}	
}

void OLDdoNegativeWithRandomAccess(cv::Mat &img) {
	int rows = img.rows;
	int cols = img.cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
}

void OLDdoNegativeWithForEach(cv::Mat &img) {
	img.forEach<unsigned char>([](unsigned char &p, const int *position) {
		p = 255 - p;
	});
}

void NEWdoNegativeWithRandomAccess(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;

	int rows = img.rows;
	int cols = img.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			img.at<Pixel>(i, j).val[0] = 255 - img.at<Pixel>(i, j).val[0];
			img.at<Pixel>(i, j).val[1] = 255 - img.at<Pixel>(i, j).val[1];
			img.at<Pixel>(i, j).val[2] = 255 - img.at<Pixel>(i, j).val[2];
		}
	}
}

void NEWdoNegativeWithMatIterator(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;

	for (Pixel &p : cv::Mat_<Pixel>(img)) {
		p.val[0] = 255 - p.val[0];
		p.val[1] = 255 - p.val[1];
		p.val[2] = 255 - p.val[2];
	}
}

void NEWdoNegativeWithLambda(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;

	img.forEach<Pixel>([](Pixel &p, const int * position) -> void {
		p.val[0] = 255 - p.val[0];
		p.val[1] = 255 - p.val[1];
		p.val[2] = 255 - p.val[2];
	});
}

#pragma endregion

#pragma region inne

int drawing_nr = 1;
cv::Mat img;
std::vector<cv::Point> points;
int roi_x0 = 0, roi_y0 = 0, roi_x1 = 0, roi_y1 = 0;
bool start_draw = false;

void draw_onmouse(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_RBUTTONDOWN)
	{
		cv::Mat &img = *((cv::Mat*)(param)); // 1st cast it back, then deref
		circle(img, cv::Point(x, y), 50, cv::Scalar(0, 255, 0), 2);
		drawing_nr++;
	}

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		cv::Mat &img = *((cv::Mat*)(param)); // 1st cast it back, then deref
		rectangle(img, cv::Point(x, y), cv::Point(x + 50, y + 40), cv::Scalar(0, 0, 255), 2);
		drawing_nr++;
	}
	cv::imshow("Image", img);
}

void doNegativeWithRandomAccess(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;

	int rows = img.rows;
	int cols = img.cols;

	int nr_channels = img.channels();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int ch = 0; ch < nr_channels; ch++) {
				img.at<Pixel>(i, j).val[ch] = 255 - img.at<Pixel>(i, j).val[ch];
			}
		}
	}
}

void doNegativeWithMatIterator(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;
	int nr_channels = img.channels();
	for (Pixel &p : cv::Mat_<Pixel>(img)) {
		for (int ch = 0; ch < nr_channels; ch++) {
			p.val[ch] = 255 - p.val[ch];
		}
	}
}

void doNegativeWithLambda(cv::Mat &img) {
	using Pixel = cv::Vec<uchar, 3>;

	img.forEach<Pixel>([](Pixel &p, const int * position) -> void {
		p.val[0] = 255 - p.val[0];
		p.val[1] = 255 - p.val[1];
		p.val[2] = 255 - p.val[2];
	});
}


cv::Mat custom_blur(cv::Mat img)
{
	int rows = img.rows;
	int cols = img.cols;
	int nr_channels = img.channels();
	using Pixel = cv::Vec<uchar, 3>;

	int size_of_kernel = 3;

	cv::Mat bgr[3];
	split(img, bgr);
	if (nr_channels == 1)
	{
		bgr[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
		bgr[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	}

	for (int i = size_of_kernel; i < rows - size_of_kernel; ++i) {
		for (int j = size_of_kernel; j < cols - size_of_kernel; ++j) {
			for (int ch = 0; ch < nr_channels; ch++) {
				cv::Mat kernel(bgr[ch], cv::Rect(j, i, size_of_kernel, size_of_kernel));
				double pole = cv::sum(kernel)[0];
				img.at<Pixel>(i, j).val[ch] = pole / (size_of_kernel*size_of_kernel);
			}
		}
	}
	cv::Mat bgrd[3];
	split(img, bgrd);
	imshow("b", bgrd[0]);
	cv::waitKey(0);

	return img;
}

cv::Mat progowanie_g(cv::Mat img, int x1, int y1, int x2, int y2)
{
	using Pixel = cv::Vec<uchar, 3>;
	int rows = img.rows;
	int cols = img.cols;

	int alpha = 125;
	for (int i = y1; i < y2; ++i) {
		for (int j = x1; j < x2; ++j) {
			if (img.at<Pixel>(i, j).val[1] > alpha)
				img.at<Pixel>(i, j).val[1] = 255;
			else if (img.at<Pixel>(i, j).val[1] <= alpha)
				img.at<Pixel>(i, j).val[1] = 0;
		}
	}
	return img;
}


void on_mouse(int event, int x, int y, int flags, void* param)
{
	cv::Mat &image = *((cv::Mat*)(param)); // 1st cast it back, then deref

	// Action when left button is clicked
	if (event == EVENT_LBUTTONDOWN)
	{
		if (!start_draw)
		{
			roi_x0 = x;
			roi_y0 = y;
			start_draw = true;
		}
		else {
			roi_x1 = x;
			roi_y1 = y;
			start_draw = false;
			cv::Mat current_view;

			current_view = progowanie_g(img, roi_x0, roi_y0, roi_x1, roi_y1);
			cv::imshow("progowanie", current_view);
		}
	}
	// Action when mouse is moving
	if ((event == EVENT_MOUSEMOVE) && start_draw)
	{
		// Redraw bounding box and rectangle
		cv::Mat current_view;
		image.copyTo(current_view);
		rectangle(current_view, cv::Point(roi_x0, roi_y0), cv::Point(x, y), cv::Scalar(0, 0, 255));
		imshow("progowanie", current_view);
	}
}

cv::Mat prewitt(cv::Mat img)
{
	cv::Mat px = (cv::Mat_<int>(3, 3) << -1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);

	cv::Mat py = (cv::Mat_<int>(3, 3) << -1, -1, -1,
		0, 0, 0,
		1, 1, 1);

	cv::Mat temp1, temp2;
	img.convertTo(temp1, CV_32F);
	img.convertTo(temp2, CV_32F);


	filter2D(img, temp1, -1, px);
	filter2D(img, temp2, -1, py);

	temp1.convertTo(temp1, CV_8U);
	temp2.convertTo(temp2, CV_8U);

	cv::Mat out = temp1 + temp2;

	return out;
}


cv::Mat sobel(cv::Mat img)
{
	cv::Mat px = (cv::Mat_<int>(3, 3) << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	cv::Mat py = (cv::Mat_<int>(3, 3) << -1, -2, -1,
		0, 0, 0,
		1, 2, 1);

	cv::Mat temp1, temp2;
	img.convertTo(temp1, CV_32F);
	img.convertTo(temp2, CV_32F);

	filter2D(img, temp1, -1, px);
	filter2D(img, temp2, -1, py);

	temp1.convertTo(temp1, CV_8U);
	temp2.convertTo(temp2, CV_8U);

	cv::Mat out = temp1 + temp2;

	return out;
}


cv::Mat hough_line(cv::Mat img)
{
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	Canny(gray, gray, 50, 150);

	std::vector<cv::Vec2f> lines; // will hold the results of the detection
	HoughLines(gray, lines, 1, CV_PI / 180, 120, 0, 0); // runs the actual detection

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img, pt1, pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
	}

	return img;
}

cv::Mat hough_line_prob(cv::Mat img)
{
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	Canny(gray, gray, 50, 150);

	// Probabilistic Line Transform
	std::vector<cv::Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(gray, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
	// Draw the lines

	for (size_t i = 0; i < linesP.size(); i++)
	{
		cv::Vec4i l = linesP[i];
		line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}

	return img;
}

cv::Mat hough_circles(cv::Mat img)
{
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	Canny(gray, gray, 100, 150);

	//imshow("j", gray);
	//waitKey(0);
	std::vector<cv::Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 50, 150, 60, 0, 0); //(input, output_vector, CV_HOUGH_GRADIENT , 1-inverse ratio, min centers' distance, max from canny, higher better, min r, max r)
	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(img, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(img, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);

	}
	return img;
}

#pragma endregion

#pragma region lab1
//live video capture with blur slider
int lab1zad1() {

	cv::Mat frame, blurred_frame;

	const int blur_slider_max = 100;
	int blur_slider = 0;

	cv::VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;

	cv::namedWindow("image");
	cv::createTrackbar("trackbar", "image", &blur_slider, blur_slider_max, on_trackbar);
	on_trackbar(blur_slider, 0);

	for (;;) {

		cap >> frame; // get a new frame from camera
		if (blur_slider > 0)
			cv::blur(frame, frame, cv::Size(blur_slider, blur_slider));

		cv::imshow("image", frame);
		if (cv::waitKey(30) >= 0) break;
	}
}

#pragma endregion

#pragma region lab2

//operacje na macierzach
void lab2zad1()
{
	cv::Mat matrix;

	std::cout << std::endl << "Single channel matrix:" << std::endl;
	matrix.create(4, 4, CV_8UC1);
	std::cout << "matrix.create(4, 4, CV_8UC1);" << std::endl << matrix << std::endl;
	matrix.setTo(cv::Scalar(128));
	std::cout << "matrix.setTo(Scalar(128));" << std::endl << matrix << std::endl;

	std::cout << std::endl << "Multi-channel matrix:" << std::endl;
	matrix.create(2, 2, CV_8UC3);
	std::cout << "matrix.create(2, 2, CV_8UC3);" << std::endl << matrix << std::endl;
	matrix.setTo(cv::Scalar(1, 2, 3, 4));
	std::cout << "matrix.setTo(Scalar(1, 2, 3, 4));" << std::endl << matrix << std::endl;

	std::cout << std::endl << "Matlab style:" << std::endl;
	matrix = cv::Mat::eye(5, 5, CV_64FC1);
	std::cout << "matrix = Mat::eye(5, 5, CV_64FC1);" << std::endl << matrix << std::endl;

	std::cout << std::endl;
	matrix = cv::Mat_<double>({ 3, 3 }, { 11, 12, 13, 21, 22, 23, 31, 32, 33 });
	std::cout << "Cała macierz:" << std::endl << matrix << std::endl;
	std::cout << "Wyświetlanie tylko fragmentu macierzy:" << std::endl;
	std::cout << "Pierwszy wiersz: matrix.row(0):" << std::endl << matrix.row(0) << std::endl;
	std::cout << "Ostatnia kolumna: matrix.col(matrix.cols-1):" << std::endl << matrix.col(matrix.cols - 1) << std::endl;

	std::cout << std::endl;
}

//tworzenie konkretnej macierzy
void lab2zad2()
{
	cv::Mat matrix;

	matrix.create(cv::Size(3, 3), CV_8UC1);

	for (int i = 0; i < 3; i++)
	{
		matrix.row(i).setTo(cv::Scalar(i + 1));
	}

	std::cout << matrix << std::endl;
}

//macierze - przykład jak lepiej nie robić
void lab2zad3_przyklad()
{
	cv::Mat mat1, mat2;
	mat1.create(cv::Size(5, 5), CV_8UC1);
	cv::randu(mat1, cv::Scalar::all(0), cv::Scalar::all(10));
	cv::Mat mat3(mat1, cv::Rect(2, 2, 2, 2));
	mat2 = mat1;
	std::cout << mat1 << std::endl << mat2 << std::endl << mat3 << std::endl;
	mat2.setTo(6);
	std::cout << mat1 << std::endl << mat2 << std::endl << mat3 << std::endl;
}

//kopiowanie macierzy (lub ich fragmentów)
void lab2zad3()
{
	cv::Mat mat1, mat2, mat3;
	mat1.create(cv::Size(5, 5), CV_8UC1);
	cv::randu(mat1, cv::Scalar::all(0), cv::Scalar::all(10));
	std::cout << "Orginal matrix:" << std::endl << mat1 << std::endl << std::endl;


	mat2 = mat1.row(0).clone();
	mat1.col(1).copyTo(mat3);
	mat1.setTo(0);
	std::cout << "Matrix changed to zeros:" << std::endl << mat1 << std::endl << std::endl;
	std::cout << "1st row of orginal matrix:" << std::endl << mat2 << std::endl << std::endl;
	std::cout << "2nd column of orginal matrix:" << std::endl << mat3 << std::endl << std::endl;
}

//negatyw
void lab2zad4()
{
	cv::Mat image;
	image = cv::imread("pies.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 10; i++)
	{
		doNegativeWithRandomAccess(image);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_secondsRA = (end - start) * 1000;

	start = std::chrono::system_clock::now();
	for (int i = 0; i < 10; i++)
	{
		OLDdoNegativeWithForEach(image);
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_secondsFE = (end - start) * 1000;

	std::cout << "Random access time (in miliseconds):" << std::endl << elapsed_secondsRA.count() << std::endl;
	std::cout << "For each time (in miliseconds):" << std::endl << elapsed_secondsFE.count() << std::endl;

	cv::namedWindow("Display window");// Create a window for display.
	cv::imshow("Display window", image);
	cv::waitKey(0);
}

//reading image, splitting and connecting
void lab2dom1()
{
	cv::Mat image = cv::imread("pies.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat image_left = image.colRange(0, image.cols / 2);
	cv::Mat image_rigth = image.colRange(image.cols / 2, image.cols);
	cv::Mat hconcat, vconcat;

	cv::hconcat(image_rigth, image_left, hconcat);
	cv::vconcat(image_left, image_rigth, vconcat);

	cv::rectangle(hconcat, cv::Point(210, 20), cv::Point(350, 250), cv::Scalar(120, 0, 155), -1);

	cv::namedWindow("image");
	cv::imshow("image", hconcat);
	cv::waitKey(0);
}

void lab2dom2()
{
	cv::Mat image = cv::imread("pies.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat image_left = image.colRange(0, image.cols / 2);
	cv::Mat image_rigth = image.colRange(image.cols / 2, image.cols);
	cv::Mat frame;

	cv::VideoCapture cap(0); // open the default camera	

	for (;;)
	{
		cap >> frame; // get a new frame from camera

		//merge images
		cv::hconcat(image_rigth, frame, frame);
		cv::hconcat(frame, image_rigth, frame);
		cv::rectangle(frame, cv::Point(210, 20), cv::Point(350, 250), cv::Scalar(120, 0, 155), -1);

		cv::imshow("image", frame);
		if (cv::waitKey(30) >= 0) break;
	}
}

#pragma endregion

#pragma region lab3

void lab3wejsciowka()
{
	cv::Mat img = cv::imread("pies.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	int rows = img.rows;
	int cols = img.cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (((i / 100) % 2) < 1)
				img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
			else
			{
				if (img.at<uchar>(i, j) > 100)
					img.at<uchar>(i, j) = 255;
				else
					img.at<uchar>(i, j) = 0;
			}

		}
	}

	cv::namedWindow("Pies");// Create a window for display.
	cv::imshow("Pies", img);
	cv::waitKey(0);
}

// different blur types 
void lab3zad1()
{
	cv::Mat img = cv::imread("lena_noise.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img_blur, img_gaussBlur, img_medianBlur, result;

	cv::blur(img, img_blur, cv::Size(3, 3));
	cv::GaussianBlur(img, img_gaussBlur, cv::Size(3, 3), 3, 3);
	cv::medianBlur(img, img_medianBlur, 3);

	std::vector<cv::Mat> matrices = { img, img_blur, img_gaussBlur, img_medianBlur };
	cv::hconcat(matrices, result);

	cv::hconcat(img, img_blur, img);
	cv::hconcat(img, img_gaussBlur, img);
	cv::hconcat(img, img_medianBlur, img);

	cv::namedWindow("baba");// Create a window for display.
	cv::imshow("baba", img);
	cv::waitKey(0);
}

// morfologic operations
void lab3zad2()
{
	cv::Mat img = cv::imread("morphology.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img_erode, img_dilate, result;

	cv::erode(img, img_erode, cv::Mat(), cv::Point(-1, -1), 3);
	cv::dilate(img, img_dilate, cv::Mat(), cv::Point(-1, -1), 3);

	cv::hconcat(img, img_erode, img);
	cv::hconcat(img, img_dilate, img);

	cv::namedWindow("litera");
	cv::imshow("litera", img);
	cv::waitKey(0);
}


void my_blur(cv::Mat &src, cv::Mat &dst)
{
	int rows = src.rows;
	int cols = src.cols;
	int temp = 0;

	for (int i = 0; i < rows - 2; ++i) {
		for (int j = 0; j < cols - 2; ++j) {
			//dst.at<uint>(i, j) = src.at<uint>(i, j);
		}
	}
}

#pragma endregion

#pragma region lab4

//creating rectangles or circles on click
void lab4zad1()
{
	image = cv::imread("pies.jpg");

	cv::namedWindow("window");// Create a window for display.

	cv::setMouseCallback("window", onMouse2, 0);

	cv::imshow("window", image);
	cv::waitKey(0);
}

void lab4zad2()
{
	image = cv::imread("pies.jpg");

	cv::namedWindow("window");// Create a window for display.

	cv::imshow("window", image);
	cv::waitKey(0);
}

#pragma endregion

#pragma region lab5

void defineMasks()
{
	Px.create(cv::Size(3, 3), CV_8SC1);
	for (int i = 0; i < 3; i++)
	{
		Px.col(i).setTo(cv::Scalar(i - 1));
	}

	Py.create(cv::Size(3, 3), CV_8SC1);
	for (int i = 0; i < 3; i++)
	{
		Py.row(i).setTo(cv::Scalar(i - 1));
	}

	Sx.create(cv::Size(3, 3), CV_8SC1);
	for (int i = 0; i < 3; i++)
	{
		Sx.col(i).setTo(cv::Scalar(i - 1));
	}
	Sx.row(1) = Sx.row(1) * 2;

	Sy.create(cv::Size(3, 3), CV_8SC1);
	for (int i = 0; i < 3; i++)
	{
		Sy.row(i).setTo(cv::Scalar(i - 1));
	}
	Sy.col(1) = Sy.col(1) * 2;
}

//simple edge detections
void lab5zad1()
{
	defineMasks();

	image = cv::imread("pies.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::namedWindow("window");

	cv::Mat filteredX, filteredY, filtered, result;

	cv::filter2D(image, filteredX, CV_32F, Px, cv::Point(-1, -1));
	cv::filter2D(image, filteredY, CV_32F, Py, cv::Point(-1, -1));

	filteredX.convertTo(filteredX, CV_8U);
	filteredY.convertTo(filteredY, CV_8U);

	int rows = filteredX.rows;
	int cols = filteredX.cols;

	result = filteredX;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result.at<uchar>(i, j) = sqrt(
				filteredX.at<uchar>(i, j) * filteredX.at<uchar>(i, j)
				+ filteredY.at<uchar>(i, j) * filteredY.at<uchar>(i, j) );
		}
	}

	cv::hconcat(filteredX, filteredY, filtered);

	cv::imshow("window", result);
	cv::waitKey(0);
}

//sobel, prewitt and canny edge detection
void lab5zad2()
{
	img = cv::imread("pies.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat prewitt_img = prewitt(img);
	cv::Mat sobel_img = sobel(img);
	cv::Mat img_canny;
	Canny(img, img_canny, 50, 150);

	cv::imshow("prewitt", prewitt_img);
	cv::imshow("sobel", sobel_img);
	cv::imshow("canny", img_canny);
	cv::waitKey(0);
}

void lab5zad3()
{
	cv::Mat img_canny, out, hough_circle, k;

	img = cv::imread("bubbles.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat img2 = cv::imread("bubbles.jpg", CV_LOAD_IMAGE_COLOR);

	cv::Mat h_line = hough_line(img);
	cv::Mat h_line_prob = hough_line_prob(img);
	cv::Mat h_circle = hough_circles(img);


	cv::imshow("hough_lines", h_line);
	cv::imshow("kough_lines_prob", h_line_prob);
	cv::imshow("hough_circle", h_circle);

	cv::waitKey(0);
}

void lab5dom1()
{

}

#pragma endregion

#pragma region lab6

cv::Mat loadImage(int number)
{
	std::stringstream ss;
	if(number < 10) 
		ss << "img_21130751_000" << number << ".bmp";
	else if (number < 100)
		ss << "img_21130751_00" << number << ".bmp";
	else if (number < 1000)
		ss << "img_21130751_0" << number << ".bmp";
	std::string path = ss.str();

	cv::Mat result = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	return result;
}

void lab6zad1()
{
	std::vector<std::vector<cv::Point2f>> corners;
	std::vector<cv::Point2f> tempCorner;	

	for (int i = 0; i < 115; i++)
	{
		cv::Mat loaded = loadImage(i);
		int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;		

		if (cv::findChessboardCorners(loaded, cv::Size(8, 5), tempCorner, flags))
		{
			//corners.push_back(tempCorner);
			cornerSubPix(loaded, tempCorner, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		}
	}
}

#pragma endregion

#pragma region lab7

void lab7zad1(bool perspective)
{
	cv::Mat image = cv::imread("marker.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	std::vector<cv::Point2f> corners;
	int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;

	if (cv::findChessboardCorners(image, cv::Size(8, 5), corners, flags))
	{
		cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	
	cv::Mat transformedImage;

	if (perspective)
	{
		std::vector<cv::Point2f> cornersIn
			= { corners[0] , corners[7] ,corners[32] ,corners[39] };

		std::vector<cv::Point2f> cornersOut
			= { cv::Point2f(100,100), cv::Point2f(240,100) ,
				cv::Point2f(100,180), cv::Point2f(240,180) };

		cv::Mat perspectiveMat = cv::getPerspectiveTransform(cornersIn, cornersOut);
		cv::warpPerspective(image, transformedImage, perspectiveMat, cv::Size(400, 300));
	}
	else
	{
		std::vector<cv::Point2f> cornersIn
			= { corners[0] , corners[7] ,corners[32] };


		std::vector<cv::Point2f> cornersOut
			= { cv::Point2f(100,100), cv::Point2f(240,100) , cv::Point2f(100,180) };

		cv::Mat affineMat = cv::getAffineTransform(cornersIn, cornersOut);
		cv::warpAffine(image, transformedImage, affineMat, cv::Size(400, 300));
	}	

	cv::imshow("window", transformedImage);
	cv::waitKey(0);
}

void lab7zad2()
{
	cv::Mat image = cv::imread("marker.jpg", CV_LOAD_IMAGE_GRAYSCALE);
}

#pragma endregion

#pragma region lab8 - detekcja ruchu

const int trackbar_max = 255;
int trackbar_value = 0;

void backroundUpdate(cv::Mat &background, cv::Mat current)
{
	cv::Mat diff;
	cv::compare(background, current, diff, cv::CMP_GT);
	cv::imshow("img", diff);
	cv::waitKey();
}

int lab8zad1()
{
	cv::Mat background, current, foreground, image;

	cv::namedWindow("image");	
	cv::createTrackbar("trackbar", "image", &trackbar_value, trackbar_max, on_trackbar);
	on_trackbar(trackbar_value, 0);

	cv::VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;

	while (true)
	{
		char key = cv::waitKey(100);

		if (key == 'a')
		{
			cap >> background; // get a new frame from camera
			cv::cvtColor(background, background, cv::COLOR_RGB2GRAY);
		}

		if (!background.empty()) 
		{
			cap >> current; // get a new frame from camera
			cv::cvtColor(current, current, cv::COLOR_RGB2GRAY);

			cv::absdiff(background, current, foreground);		
			cv::threshold(foreground, foreground, trackbar_value, 255, cv::THRESH_BINARY);
			cv::dilate(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 3);
			cv::erode(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 3);
			

			cv::hconcat(background, current, image);
			cv::hconcat(image, foreground, image);

			cv::imshow("image", image);
		}		
	}
}

int lab8zad2()
{
	cv::Mat background, current, foreground, image;

	cv::VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	
	cap >> background; // get a new frame from camera
	cv::cvtColor(background, background, cv::COLOR_RGB2GRAY);

	while (true)
	{
		cap >> current;
		cv::cvtColor(current, current, cv::COLOR_RGB2GRAY);
			
		backroundUpdate(background, current);		
	}
	
}

#pragma endregion

int main(int, char**) 
{	
	lab8zad2();
	return 0;
}
