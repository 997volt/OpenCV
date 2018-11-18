#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <utility>

cv::Mat makeBackgroundBlack(cv::Mat image, cv::Point center, int radius)
{
	using Pixel = cv::Vec<uchar, 3>;

	image.forEach<Pixel>([](Pixel &p, const int * position) -> void {
		p.val[0] = 255 - p.val[0];
		p.val[1] = 255 - p.val[1];
		p.val[2] = 255 - p.val[2];
	});

	return image;
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

cv::Mat scaleImage(cv::Mat image, int pixels)
{
	//scaling image to have input pixels vertically
	double scale = (double)pixels / (double)image.rows;
	cv::resize(image, image, cv::Size(0, 0), scale, scale);

	return image;
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

cv::Mat drawCircle(cv::Mat image, std::pair <cv::Point, int> circle)
{
	// draw the circle center
	cv::circle(image, circle.first, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	// draw the circle outline
	cv::circle(image, circle.first, circle.second, cv::Scalar(0, 0, 255), 3, 8, 0);

	return image;
}

int main(int argc, char** argv)
{
	cv::Mat img = cv::imread("zegar1.jpg", CV_LOAD_IMAGE_COLOR);

	img = scaleImage(img, 800);

	cv::Mat gray = imagePreprocessing(img);	

	std::pair <cv::Point, int> circle = getCircle(gray);
	img = drawCircle(img, circle);

	//making backgroud black

	//showing results
	cv::imshow("circles", img);
	cv::imshow("gray", gray);
	cv::waitKey(0);
	return 0;
}
