#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <utility>

void makeBackgroundBlack(cv::Mat &image, std::pair <cv::Point, int> circle)
{
	int rows = image.rows;
	int cols = image.cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (pow(circle.first.x - j, 2) + pow(circle.first.y - i, 2) > pow(circle.second, 2))
				image.at<uchar>(i, j) = 0;
		}
	}
}

void cropBlackBars(cv::Mat &image)
{
	int xMin = 0;
	int xMax = image.cols;
	int yMin = 0;
	int yMax = image.rows;
	
	//cut top
	for (int i = 0; i < yMax; ++i) {
		for (int j = 0; j < xMax; ++j) {
			if (image.at<uchar>(i, j) != 0)
			{
				yMin = i;
				break;
			}
		}
		if (yMin != 0) break;
	}

	//cut bottom
	for (int i = yMax-1; i > -1; --i) {
		for (int j = 0; j < xMax; ++j) {
			if (image.at<uchar>(i, j) != 0)
			{
				yMax = i;
				break;
			}
		}
		if (yMax != image.rows) break;
	}

	//cut left
	for (int i = 0; i < xMax; ++i) {
		for (int j = 0; j < yMax; ++j) {
			if (image.at<uchar>(j,i) != 0)
			{
				xMin = i;
				break;
			}
		}
		if (xMin != 0) break;
	}

	//cut rigth
	for (int i = xMax-1; i > -1; --i) {
		for (int j = yMin; j < yMax; ++j) {
			if (image.at<uchar>(j, i) != 0)
			{
				xMax = i;
				break;
			}
		}
		if (xMax != image.cols) break;
	}

	cv::Rect myROI(xMin, yMin, xMax-xMin, yMax-yMin);

	image = image(myROI);
}

cv::Mat imagePreprocessing(cv::Mat image)
{
	cv::Mat result;	

	//converting to grayscale 
	cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);

	//blurring gray image - get rid of noise
	cv::GaussianBlur(result, result, cv::Size(9, 9), 0);
	cv::medianBlur(result, result, 5);

	return result;
}

/// scaling image to have input pixels vertically
void scaleImage(cv::Mat &image, int pixels)
{
	double scale = (double)pixels / (double)image.rows;
	cv::resize(image, image, cv::Size(0, 0), scale, scale);
}

std::pair <cv::Point, int> getCircle(cv::Mat image)
{
	std::pair <cv::Point, int> result;

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, image.rows);	

	//converting circle data to point and radius
	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);	

		result = std::pair <cv::Point, int>(center, radius);
	}	

	return result;
}

void drawCircle(cv::Mat &image, std::pair <cv::Point, int> circle)
{
	// draw the circle center
	cv::circle(image, circle.first, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	// draw the circle outline
	cv::circle(image, circle.first, circle.second, cv::Scalar(0, 0, 255), 3, 8, 0);
}

void findClockHands(cv::Mat &image)
{
	cv::Canny(image, image, 50, 200);
	image.convertTo(image, CV_8U);

	cv::Mat hands;

	// Standard Hough Line Transform
	std::vector<cv::Vec4i> lines; // will hold the results of the detection
	cv::HoughLinesP(image, lines, 1, CV_PI / 180, 60, 50, 5);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 150, 3, CV_AA);
	}
}

int main(int argc, char** argv)
{
	cv::Mat img = cv::imread("zegar3.jpg", CV_LOAD_IMAGE_COLOR);
	scaleImage(img, 800);
	cv::imshow("1", img);

	cv::Mat gray = imagePreprocessing(img);		
	cv::imshow("2", gray);

	std::pair <cv::Point, int> circle = getCircle(gray);
	drawCircle(img, circle);
	cv::imshow("3", img);

	cv::Mat polar;
	cv::warpPolar(gray, polar, cv::Size(1000,1000), circle.first, 1000, cv::WARP_POLAR_LINEAR);
	cv::imshow("polar", polar);

	makeBackgroundBlack(gray, circle);
	cv::imshow("4", gray);
	cropBlackBars(gray);
	scaleImage(gray, 400);
	cv::imshow("5", gray);
	
	findClockHands(gray);
	cv::imshow("6", gray);

	//cv::imshow("gray", gray);
	cv::waitKey(0);
	return 0;
}
