#ifndef _MYIMAGE_
#define _MYIMAGE_ 

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class MyImage{
	public:
		MyImage(int webCamera);
		MyImage();
		Mat srcLR; // store processed image
		Mat src; // source image
		Mat srcPrev; // previous image
		Mat bw; // binary image
		Mat patchImg; // patch image
		Mat secondPatchImg;
		cv::Point fingerTipLoc;
		cv::Point secondTipLoc;
		cv::Point2f firstMatchLoc;
		cv::Point2f secondMatchLoc;
		float firstMatchScore;
		float secondMatchScore;
		vector<Mat> bwList;
		VideoCapture cap;		
		int cameraSrc; 
		void initWebCamera(int i);
};



#endif
