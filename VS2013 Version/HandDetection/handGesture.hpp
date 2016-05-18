#ifndef _HAND_GESTURE_
#define _HAND_GESTURE_ 

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "main.hpp"
#include "myImage.hpp"

enum GestureState { IDLE, ONE_FINGER, TOUCH, TWO_FINGERS, ZOOM_IN, ZOOM_OUT, PANNING, OTHERS };

using namespace cv;
using namespace std;

class HandGesture{
	public:
		MyImage m;
		HandGesture();
		vector<vector<Point> > contours;
		vector<vector<int> >hullI;
		vector<vector<Point> >hullP;
		vector<vector<Vec4i> > defects;	
		vector <Point> fingerTips;
		Rect rect;
		void printGestureInfo(Mat src);
		int cIdx;
		int frameNumber;
		int mostFrequentFingerNumber;
		int nrOfDefects;
		Rect bRect;
		double bRect_width;
		double bRect_height;
		bool isHand;
		bool detectIfHand();
		void initVectors();
		void getFingerNumber(MyImage *m);
		void eleminateDefects(MyImage *m);
		void getFingerTips(MyImage *m);
		void drawFingerTips(MyImage *m);
		vector<Point> oneFingerCoordinates; // store detected coor-s for the only finger 
		vector<Point> firstFingerCoordinates; // store the first detected finger coor-s 
		vector<Point> secondFingerCoordinates; // store the second detected finger coor-s  
		vector<Point2f> matchPointsCoordinates; // store patch matching points for first finger
		vector<Point2f> secondMatchPtsCoordinates; // store patch matching points for second finger
		GestureState state; 
		GestureState prevState;
	private:
		string bool2string(bool tf);
		int fontFace;
		int prevNrFingerTips;
		void checkForOneFinger(MyImage *m);
		float getAngle(Point s,Point f,Point e);	
		vector<int> fingerNumbers;
		//vector<Point2f> prevFeaturePoints;
		//vector<Point2f> currFeaturePoints;
		//vector<Point2f> optFlowCoordinates;
		void analyzeContours();
		string intToString(int number);
		void computeFingerNumber();
		void drawNewNumber(MyImage *m);
		void addNumberToImg(MyImage *m);
		vector<int> numbers2Display;
		void addFingerNumberToVector();
		Scalar numberColor;
		int nrNoFinger;
		float distanceP2P(Point a,Point b);
		void removeRedundantEndPoints(vector<Vec4i> newDefects,MyImage *m);
		void removeRedundantFingerTips();
};




#endif
