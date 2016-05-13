#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "myImage.hpp"
#include "roi.hpp"
#include "handGesture.hpp"
#include <vector>
#include <cmath>
#include "main.hpp"

//using namespace cv;
//using namespace std;

/* Global Variables  */
int fontFace = cv::FONT_HERSHEY_PLAIN;
int square_len;
int avgColor[NSAMPLES][3] ;
int c_lower[NSAMPLES][3];
int c_upper[NSAMPLES][3];
int avgBGR[3];
int nrOfDefects;
int iSinceKFInit;
struct dim{int w; int h;}boundingDim;
	cv::VideoWriter out;
cv::Mat edges;
My_ROI roi1, roi2,roi3,roi4,roi5,roi6;
std::vector <My_ROI> roi;
std::vector <KalmanFilter> kf;
std::vector <cv::Mat_<float> > measurement;

int noFingerFrameCounter, oneFingerFrameCounter, twoFingersFrameCounter, othersFrameCounter;

/* end global variables */

void init(MyImage *m){
	square_len=20;
	iSinceKFInit=0;
}

// change a color from one space to another
void col2origCol(int hsv[3], int bgr[3], cv::Mat src){
	Mat avgBGRMat=src.clone();	
	for(int i=0;i<3;i++){
		avgBGRMat.data[i]=hsv[i];	
	}
	cv::cvtColor(avgBGRMat,avgBGRMat,COL2ORIGCOL);
	for(int i=0;i<3;i++){
		bgr[i]=avgBGRMat.data[i];	
	}
}

void printText(cv::Mat src, std::string text){
	int fontFace = FONT_HERSHEY_PLAIN;
	cv::putText(src,text,Point(src.cols/2, src.rows/10),fontFace, 1.2f,Scalar(200,0,0),2);
}

void waitForPalmCover(MyImage* m){
    m->cap >> m->src;
	if (m->src.cols < 60 || m->src.rows < 40) {
		std::cout << "m->src size is too small." << std::endl;
	}
	//m->src = Mat(60, 40, CV_32F);
	flip(m->src,m->src,1);
	//std::cout << m->src.cols / 3 << " " << m->src.rows / 6 << " " << m->src.cols/3+square_len << std::endl;
	// define areas to sample hand color
	roi.push_back(My_ROI(cv::Point(m->src.cols/3, m->src.rows/6),cv::Point(m->src.cols/3+square_len,m->src.rows/6+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/4, m->src.rows/2),cv::Point(m->src.cols/4+square_len,m->src.rows/2+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/3, m->src.rows/1.5),cv::Point(m->src.cols/3+square_len,m->src.rows/1.5+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/2, m->src.rows/2),cv::Point(m->src.cols/2+square_len,m->src.rows/2+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/2.5, m->src.rows/2.5),cv::Point(m->src.cols/2.5+square_len,m->src.rows/2.5+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/2, m->src.rows/1.5),cv::Point(m->src.cols/2+square_len,m->src.rows/1.5+square_len),m->src));
	roi.push_back(My_ROI(cv::Point(m->src.cols/2.5, m->src.rows/1.8),cv::Point(m->src.cols/2.5+square_len,m->src.rows/1.8+square_len),m->src));
	
	
	for(int i =0;i<50;i++){
    	m->cap >> m->src;
		cv::flip(m->src,m->src,1);
		// draw N roi on image
		for(int j=0;j<NSAMPLES;j++){
			roi[j].draw_rectangle(m->src);
		}
		std::string imgText = std::string("Cover rectangles with palm");
		printText(m->src,imgText);	
		
		if(i==30){
			cv::imwrite("..\\images\\waitforpalm1.jpg",m->src);
		}

		cv::imshow("img1", m->src);
		out << m->src;
        if(cv::waitKey(30) >= 0) break;
	}
}

int getMedian(std::vector<int> val){
  int median;
  size_t size = val.size();
  std::sort(val.begin(), val.end());
  if (size  % 2 == 0)  {
      median = val[size / 2 - 1] ;
  } else{
      median = val[size / 2];
  }
  return median;
}


void getAvgColor(MyImage *m,My_ROI roi,int avg[3]){
	cv::Mat r;
	roi.roi_ptr.copyTo(r);
	std::vector<int>hm;
	std::vector<int>sm;
	std::vector<int>lm;
	// generate vectors
	for(int i=2; i<r.rows-2; i++){
    	for(int j=2; j<r.cols-2; j++){
    		hm.push_back(r.data[r.channels()*(r.cols*i + j) + 0]) ;
        	sm.push_back(r.data[r.channels()*(r.cols*i + j) + 1]) ;
        	lm.push_back(r.data[r.channels()*(r.cols*i + j) + 2]) ;
   		}
	}
	avg[0]=getMedian(hm);
	avg[1]=getMedian(sm);
	avg[2]=getMedian(lm);
}

void average(MyImage *m){
	m->cap >> m->src;
	cv::flip(m->src,m->src,1);
	for(int i=0;i<100;i++){
		m->cap >> m->src;
		cv::flip(m->src,m->src,1);
		cv::cvtColor(m->src,m->src,ORIGCOL2COL);
		for(int j=0;j<NSAMPLES;j++){
			getAvgColor(m,roi[j],avgColor[j]);
			roi[j].draw_rectangle(m->src);
		}	
		cv::cvtColor(m->src,m->src,COL2ORIGCOL);
		std::string imgText = std::string("Finding average color of hand");
		printText(m->src,imgText);	
		cv::imshow("img1", m->src);
        if(cv::waitKey(30) >= 0) break;
	}
}

void initTrackbars(){
	for(int i=0;i<NSAMPLES;i++){
		c_lower[i][0]=12;
		c_upper[i][0]=7;
		c_lower[i][1]=30;
		c_upper[i][1]=40;
		c_lower[i][2]=80;
		c_upper[i][2]=80;
	}
	cv::createTrackbar("lower1","trackbars",&c_lower[0][0],255);
	cv::createTrackbar("lower2","trackbars",&c_lower[0][1],255);
	cv::createTrackbar("lower3","trackbars",&c_lower[0][2],255);
	cv::createTrackbar("upper1","trackbars",&c_upper[0][0],255);
	cv::createTrackbar("upper2","trackbars",&c_upper[0][1],255);
	cv::createTrackbar("upper3","trackbars",&c_upper[0][2],255);
}


void normalizeColors(MyImage * myImage){
	// copy all boundries read from trackbar
	// to all of the different boundries
	for(int i=1;i<NSAMPLES;i++){
		for(int j=0;j<3;j++){
			c_lower[i][j]=c_lower[0][j];	
			c_upper[i][j]=c_upper[0][j];	
		}	
	}
	// normalize all boundries so that 
	// threshold is whithin 0-255
	for(int i=0;i<NSAMPLES;i++){
		if((avgColor[i][0]-c_lower[i][0]) <0){
			c_lower[i][0] = avgColor[i][0] ;
		}if((avgColor[i][1]-c_lower[i][1]) <0){
			c_lower[i][1] = avgColor[i][1] ;
		}if((avgColor[i][2]-c_lower[i][2]) <0){
			c_lower[i][2] = avgColor[i][2] ;
		}if((avgColor[i][0]+c_upper[i][0]) >255){ 
			c_upper[i][0] = 255-avgColor[i][0] ;
		}if((avgColor[i][1]+c_upper[i][1]) >255){
			c_upper[i][1] = 255-avgColor[i][1] ;
		}if((avgColor[i][2]+c_upper[i][2]) >255){
			c_upper[i][2] = 255-avgColor[i][2] ;
		}
	}
}

void produceBinaries(MyImage *m){	
	cv::Scalar lowerBound;
	cv::Scalar upperBound;
	cv::Mat foo;
	for(int i=0;i<NSAMPLES;i++){
		normalizeColors(m);
		lowerBound=cv::Scalar( avgColor[i][0] - c_lower[i][0] , avgColor[i][1] - c_lower[i][1], avgColor[i][2] - c_lower[i][2] );
		upperBound=cv::Scalar( avgColor[i][0] + c_upper[i][0] , avgColor[i][1] + c_upper[i][1], avgColor[i][2] + c_upper[i][2] );
		m->bwList.push_back(cv::Mat(m->srcLR.rows,m->srcLR.cols,CV_8U));	
		cv::inRange(m->srcLR,lowerBound,upperBound,m->bwList[i]);	
	}
	m->bwList[0].copyTo(m->bw);
	for(int i=1;i<NSAMPLES;i++){
		m->bw+=m->bwList[i];	
	}
	medianBlur(m->bw, m->bw,7);
}

void initWindows(MyImage m){
    namedWindow("trackbars",CV_WINDOW_KEEPRATIO);
    namedWindow("img1",CV_WINDOW_FULLSCREEN);
	namedWindow("img2", CV_WINDOW_FULLSCREEN);
}

void showWindows(MyImage m){
	pyrDown(m.bw,m.bw);
	pyrDown(m.bw,m.bw);
	cv::Rect roi( cv::Point( 3*m.src.cols/4,0 ), m.bw.size());
	std::vector<cv::Mat> channels;
	cv::Mat result;
	for(int i=0;i<3;i++)
		channels.push_back(m.bw);
	cv::merge(channels,result);
	result.copyTo( m.src(roi));
	cv::imshow("img1",m.src);	
}

int findBiggestContour(std::vector<std::vector<cv::Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

void myDrawContours(MyImage *m,HandGesture *hg){
	drawContours(m->src,hg->hullP,hg->cIdx,cv::Scalar(200,0,0),2, 8, std::vector<cv::Vec4i>(), 0, cv::Point());




	rectangle(m->src,hg->bRect.tl(),hg->bRect.br(),Scalar(0,0,200));
	vector<Vec4i>::iterator d=hg->defects[hg->cIdx].begin();
	int fontFace = FONT_HERSHEY_PLAIN;
		
	
	vector<Mat> channels;
		Mat result;
		for(int i=0;i<3;i++)
			channels.push_back(m->bw);
		merge(channels,result);
		//drawContours(result,hg->contours,hg->cIdx,cv::Scalar(0,200,0),6, 8, vector<Vec4i>(), 0, Point());
		drawContours(result,hg->hullP,hg->cIdx,cv::Scalar(0,0,250),10, 8, vector<Vec4i>(), 0, Point());

		
	while( d!=hg->defects[hg->cIdx].end() ) {
   	    Vec4i& v=(*d);
	    int startidx=v[0]; Point ptStart(hg->contours[hg->cIdx][startidx] );
   		int endidx=v[1]; Point ptEnd(hg->contours[hg->cIdx][endidx] );
  	    int faridx=v[2]; Point ptFar(hg->contours[hg->cIdx][faridx] );
	    float depth = v[3] / 256;
   /*	
		line( m->src, ptStart, ptFar, Scalar(0,255,0), 1 );
	    line( m->src, ptEnd, ptFar, Scalar(0,255,0), 1 );
   		circle( m->src, ptFar,   4, Scalar(0,255,0), 2 );
   		circle( m->src, ptEnd,   4, Scalar(0,0,255), 2 );
   		circle( m->src, ptStart,   4, Scalar(255,0,0), 2 );
*/
   		circle( result, ptFar,   9, Scalar(0,205,0), 5 );
		
		
	    d++;

   	 }
	 imwrite("..\\images\\contour_defects_before_eliminate.jpg",result);

}

void makeContours(MyImage *m, HandGesture* hg){
	Mat aBw;
	cv::pyrUp(m->bw,m->bw);
	m->bw.copyTo(aBw);
	cv::findContours(aBw,hg->contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	//std::cout << "Current contour size is " << (int)hg->contours.size() << std::endl;
	hg->initVectors(); 
	hg->cIdx=findBiggestContour(hg->contours); // record the biggest contour index in hg->cIdx
	//std::cout << "biggest contour has a size of " << hg->contours[hg->cIdx].size() << std::endl;
	if(hg->cIdx!=-1){
        //approxPolyDP( Mat(hg->contours[hg->cIdx]), hg->contours[hg->cIdx], 11, true );
		// use the biggest contour that has been found
		hg->bRect=boundingRect(Mat(hg->contours[hg->cIdx])); // get the bounding box of the contour	
		cv::convexHull(Mat(hg->contours[hg->cIdx]),hg->hullP[hg->cIdx],false,true); // return convex hull points
		cv::convexHull(Mat(hg->contours[hg->cIdx]),hg->hullI[hg->cIdx],false,false); // return indices of convex hull points
		approxPolyDP( Mat(hg->hullP[hg->cIdx]), hg->hullP[hg->cIdx], 18, true );
		if(hg->contours[hg->cIdx].size()>3 ){
			cv::convexityDefects(hg->contours[hg->cIdx],hg->hullI[hg->cIdx],hg->defects[hg->cIdx]);
			hg->eleminateDefects(m);
		}
		bool isHand = hg->detectIfHand();

		hg->printGestureInfo(m->src);
		hg->fingerTips.clear();
		if(isHand){	
			hg->getFingerTips(m);
			hg->drawFingerTips(m);
			myDrawContours(m,hg);
		}
	}
}

void switchState(HandGesture* hg) {
	// hand gesture state machine
	if (hg->fingerTips.size() == 1) {
		oneFingerFrameCounter++;
		if (oneFingerFrameCounter >= 8) {
			// reset other frame counters
			noFingerFrameCounter = 0;
			twoFingersFrameCounter = 0;
			othersFrameCounter = 0;
			// switch state to ONE_FINGER
			hg->state = ONE_FINGER;
		}
	}
	else if (hg->fingerTips.size() == 0) {
		noFingerFrameCounter++;
		if (noFingerFrameCounter >= 5) {
			// reset other frame counters
			oneFingerFrameCounter = 0;
			othersFrameCounter = 0;
			// switch state to IDLE
			hg->state = IDLE;
		}
	}
	else if (hg->fingerTips.size() == 2) {
		twoFingersFrameCounter++;
		if (twoFingersFrameCounter >= 10 ||
			(oneFingerFrameCounter >= 5 && twoFingersFrameCounter >= 5)) {
			// reset other frame counters
			noFingerFrameCounter = 0;
			oneFingerFrameCounter = 0;
			othersFrameCounter = 0;
			// switch state to TWO_FINGERS
			hg->state = TWO_FINGERS;
		}
	}
	else {
		othersFrameCounter++;
		if (othersFrameCounter >= 5) {
			// reset other frame counters
			noFingerFrameCounter = 0;
			oneFingerFrameCounter = 0;
			// switch state to OTHERS
			hg->state = OTHERS;
		}
	}

	std::cout << hg->fingerTips.size() << " finger(s) detected in current frame." << std::endl;
	std::cout << "noFingerFrameCounter = " << noFingerFrameCounter << std::endl;
	std::cout << "oneFingerFrameCounter = " << oneFingerFrameCounter << std::endl;
	std::cout << "twoFingerFrameCounter = " << twoFingersFrameCounter << std::endl;
	std::cout << "othersFrameCounter = " << othersFrameCounter << std::endl;

	switch (hg->state)
	{
	case GestureState::IDLE:
		std::cout << "Current state is IDLE." << std::endl;
		break;
	case GestureState::ONE_FINGER:
		std::cout << "Current state is ONE_FINGER." << std::endl;
		break;
	case GestureState::TWO_FINGERS:
		std::cout << "Currnet state is TWO_FINGERS." << std::endl;
		break;
	case GestureState::OTHERS:
		std::cout << "Current state is OTHERS." << std::endl;
		break;
	default:
		std::cout << "Current state is not recognized." << std::endl;
		break;
	}
}

void patchMatchingTracker(MyImage *m, HandGesture* hg) {
	cv::Mat src_copy;
	m->src.copyTo(src_copy);

	// run patch matching if a patch image is already found
	if (!m->patchImg.empty()) {
		cv::imwrite("..\\images\\patch_image_current_1.jpg", m->patchImg);
		/*cv::Rect regionToCompare(m.fingerTipLoc.x - 20, m.fingerTipLoc.y - 20, 80, 80);
		cv::Mat imageToCompare = m.src(regionToCompare);*/
		cv::Mat result; // matrix to store matching result
		int result_cols = m->src.cols - m->patchImg.cols + 1;
		int result_rows = m->src.rows - m->patchImg.rows + 1;
		/*int result_cols = imageToCompare.cols - m.patchImg.cols + 1;
		int result_rows = imageToCompare.rows - m.patchImg.rows + 1;*/
		result.create(result_rows, result_cols, CV_32FC1);
		// do patch matching and normalization
		//cv::flip(m.patchImg, m.patchImg, 1);
		cv::matchTemplate(m->src, m->patchImg, result, CV_TM_CCOEFF_NORMED);
		cv::normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		// localize the best match with minMaxLoc
		double minVal, maxVal;
		cv::Point minLoc, maxLoc, matchLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = maxLoc;
		/*matchLoc.x = m.fingerTipLoc.x - 100 + matchLoc.x;
		matchLoc.y = m.fingerTipLoc.y - 100 + matchLoc.y;*/
		m->firstMatchLoc = matchLoc;
		std::cout << "First matching point coordinates: " << matchLoc.x << " " << matchLoc.y << std::endl;

		// store matching points 
		if (hg->matchPointsCoordinates.size() > 0) {
			if (sqrt(pow(hg->matchPointsCoordinates.back().x - matchLoc.x, 2) +
				pow(hg->matchPointsCoordinates.back().y - matchLoc.y, 2)) < 50) {
				if (hg->matchPointsCoordinates.size() == 30) {
					hg->matchPointsCoordinates.erase(hg->matchPointsCoordinates.begin());
				}
				hg->matchPointsCoordinates.push_back(matchLoc);
			}
			else {
				hg->matchPointsCoordinates.clear();
			}
		}
		else {
			hg->matchPointsCoordinates.push_back(matchLoc);
		}

		// display matching points and trajectory
		if ((matchLoc.x + m->patchImg.cols) < m->src.cols &&
			(matchLoc.y + m->patchImg.rows) < m->src.rows) {

			cv::rectangle(src_copy, matchLoc, cv::Point(matchLoc.x + m->patchImg.cols,
				matchLoc.y + m->patchImg.rows), cv::Scalar(255, 0, 0), 2, 8, 0);
			if (hg->matchPointsCoordinates.size() >= 2) {
				for (int i = 0; i < hg->matchPointsCoordinates.size() - 1; i++) {
					cv::line(src_copy, hg->matchPointsCoordinates[i], hg->matchPointsCoordinates[i + 1], cv::Scalar(255, 0, 0), 2, 8);
				}
			}
			//cv::imshow("img2", src_copy);
		}
	}

	// run patch matching on second patch image
	if (!m->secondPatchImg.empty()) {
		cv::imwrite("..\\images\\patch_image_current_2.jpg", m->secondPatchImg);
		cv::Mat result; // matrix to store matching result
		int result_cols = m->src.cols - m->secondPatchImg.cols + 1;
		int result_rows = m->src.rows - m->secondPatchImg.rows + 1;
		result.create(result_rows, result_cols, CV_32FC1);
		cv::matchTemplate(m->src, m->secondPatchImg, result, CV_TM_CCOEFF_NORMED);
		cv::normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		// localize the best match with minMaxLoc
		double minVal, maxVal;
		cv::Point minLoc, maxLoc, matchLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = maxLoc;
		m->secondMatchLoc = matchLoc;
		std::cout << "Second matching point coordinates: " << matchLoc.x << " " << matchLoc.y << std::endl;

		// if first and second match location are twoo close, same finger is probably detected
		if (sqrt(pow(m->firstMatchLoc.x - m->secondMatchLoc.x, 2) + 
			pow(m->firstMatchLoc.y - m->secondMatchLoc.y, 2)) > 20) {

			// store matching points 
			if (hg->secondMatchPtsCoordinates.size() > 0) {
				if (sqrt(pow(hg->secondMatchPtsCoordinates.back().x - matchLoc.x, 2) +
					pow(hg->secondMatchPtsCoordinates.back().y - matchLoc.y, 2)) < 50) {
					if (hg->secondMatchPtsCoordinates.size() == 30) {
						hg->secondMatchPtsCoordinates.erase(hg->secondMatchPtsCoordinates.begin());
					}
					hg->secondMatchPtsCoordinates.push_back(matchLoc);
				}
				else {
					hg->secondMatchPtsCoordinates.clear();
				}
			}
			else {
				hg->secondMatchPtsCoordinates.push_back(matchLoc);
			}

			// display matching points and trajectory
			if ((matchLoc.x + m->secondPatchImg.cols) < m->src.cols &&
				(matchLoc.y + m->secondPatchImg.rows) < m->src.rows) {

				cv::rectangle(src_copy, matchLoc, cv::Point(matchLoc.x + m->secondPatchImg.cols,
					matchLoc.y + m->secondPatchImg.rows), cv::Scalar(0, 255, 0), 2, 8, 0);
				if (hg->secondMatchPtsCoordinates.size() >= 2) {
					for (int i = 0; i < hg->secondMatchPtsCoordinates.size() - 1; i++) {
						cv::line(src_copy, hg->secondMatchPtsCoordinates[i], hg->secondMatchPtsCoordinates[i + 1], cv::Scalar(0, 255, 0), 2, 8);
					}
				}
				//cv::imshow("img2", src_copy);
			}
		}
	}
	cv::imshow("img2", src_copy);
}


int main(){
	MyImage m(0);	// init MyImage with webcamera 0	
	// check if the camera is open successfully
	if (!m.cap.isOpened()) {
		std::cout << "Camera is not working as expect." << std::endl;
	}
	HandGesture hg;
	hg.state = IDLE; // initial state is IDLE
	// init frame counters for state machine 
	oneFingerFrameCounter = 0;
	twoFingersFrameCounter = 0;
	noFingerFrameCounter = 0;
	othersFrameCounter = 0;

	init(&m);		
	m.cap >> m.src; // get a new frame from camera
    namedWindow("img1",CV_WINDOW_KEEPRATIO);
	out.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, m.src.size(), true);
	waitForPalmCover(&m); 
	average(&m);
	destroyWindow("img1");
	initWindows(m);
	initTrackbars();
	for(;;){
		std::cout << "-------------" << std::endl;
		hg.frameNumber++;
		m.cap >> m.src;
		cv::flip(m.src,m.src,1);
		
		patchMatchingTracker(&m, &hg); // call patch matching to track fingertip 

		cv::pyrDown(m.src,m.srcLR); // blur and down sampling an image
		cv::blur(m.srcLR,m.srcLR,Size(3,3));
		cv::cvtColor(m.srcLR,m.srcLR,ORIGCOL2COL); // convert the image from one color space to another
		produceBinaries(&m);
		cvtColor(m.srcLR,m.srcLR,COL2ORIGCOL);
		makeContours(&m, &hg);
		hg.getFingerNumber(&m);

		switchState(&hg); // update state with detected finger(s)

		showWindows(m);
		out << m.src;
		imwrite("..\\images\\final_result.jpg",m.src);
		m.src.copyTo(m.srcPrev); // store current frame as previous
    	if(cv::waitKey(30) == char('q')) break;
	}
	destroyAllWindows();
	out.release();
	m.cap.release();
    return 0;
}
