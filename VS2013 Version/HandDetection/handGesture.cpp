#include "handGesture.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

HandGesture::HandGesture(){
	frameNumber=0;
	nrNoFinger=0;
	fontFace = FONT_HERSHEY_PLAIN;
}

void HandGesture::initVectors(){
	hullI=vector<vector<int> >(contours.size());
	hullP=vector<vector<Point> >(contours.size());
	defects=vector<vector<Vec4i> > (contours.size());	
}

void HandGesture::analyzeContours(){
	bRect_height=bRect.height;
	bRect_width=bRect.width;
}

string HandGesture::bool2string(bool tf){
	if(tf)
		return "true";
	else
		return "false";
}

string HandGesture::intToString(int number){
		stringstream ss;
		ss << number;
		string str = ss.str();
		return str;
}

void HandGesture::printGestureInfo(Mat src){
	int fontFace = FONT_HERSHEY_PLAIN;
	//Scalar fColor(245,200,200);
	cv::Scalar fColor(0, 0, 255);
	//int xpos=src.cols/1.5;
	int xpos = src.cols / 1.6;
	//int ypos=src.rows/1.6;
	int ypos = src.rows / 1.4;
	float fontSize=0.7f;
	int lineChange=14;
	string info= "Figure info:";
	putText(src,info,Point(ypos,xpos),fontFace,fontSize,fColor);
	xpos+=lineChange;
	info=string("Number of defects: ") + string(intToString(nrOfDefects)) ;
	putText(src,info,Point(ypos,xpos),fontFace,fontSize  ,fColor);
	xpos+=lineChange;
	info=string("bounding box height, width ") + string(intToString(bRect_height)) + string(" , ") +  string(intToString(bRect_width)) ;
	putText(src,info,Point(ypos,xpos),fontFace,fontSize ,fColor);
	xpos+=lineChange;
	info=string("Is hand: ") + string(bool2string(isHand));
	putText(src,info,Point(ypos,xpos),fontFace,fontSize  ,fColor);
	xpos += lineChange;
	std::string currState;
	switch (state)
	{
	case GestureState::IDLE:
		currState = "IDLE";
		break;
	case GestureState::ONE_FINGER:
		currState = "ONE_FINGER";
		break;
	case GestureState::TOUCH:
		currState = "TOUCH";
		break;
	case GestureState::PANNING:
		currState = "PANNING";
		break;
	case GestureState::TWO_FINGERS:
		currState = "TWO_FINGERS";
		break;
	case GestureState::ZOOM_IN:
		currState = "ZOOM_IN";
		break;
	case GestureState::ZOOM_OUT:
		currState = "ZOOM_OUT";
		break;
	case GestureState::OTHERS:
		currState = "OTHERS";
		break;
	default:
		currState = "UNKNOWN";
		break;
	}
	info = std::string("Current state is: ") + currState;
	putText(src, info, Point(ypos, xpos), fontFace, fontSize, fColor);
}

bool HandGesture::detectIfHand(){
	analyzeContours();
	double h = bRect_height; 
	double w = bRect_width;
	isHand=true;
	if(fingerTips.size() > 5 ){
		std::cout << "Not hand: more than 5 fingers." << std::endl;
		isHand=false;
	}else if(h==0 || w == 0){
		std::cout << "Not hand: bounding box too small." << std::endl;
		isHand=false;
	}else if(h/w > 4 || w/h >4){
		std::cout << "Not hand: bounding box height/width or width/height ratio not in a valid range." << std::endl;
		isHand=false;	
	}else if(bRect.x<20){
		std::cout << "Not hand: bounding box too close to the edge." << std::endl;
		isHand=false;	
	}	
	return isHand;
}

float HandGesture::distanceP2P(Point a, Point b){
	float d= sqrt(fabs( pow(a.x-b.x,2) + pow(a.y-b.y,2) )) ;  
	return d;
}

// remove fingertips that are too close to 
// eachother
void HandGesture::removeRedundantFingerTips(){
	std::cout << "fingerTips size = " << fingerTips.size() << std::endl;
	vector<Point> newFingers;
	for(int i=0;i<fingerTips.size();i++){
		for(int j=i;j<fingerTips.size();j++){
			if(distanceP2P(fingerTips[i],fingerTips[j])<10 && i!=j){
			}else{
				newFingers.push_back(fingerTips[i]);	
				break;
			}	
		}	
	}
	std::cout << "newFingers size = " << newFingers.size() << std::endl;
	fingerTips.swap(newFingers);
}

void HandGesture::computeFingerNumber(){
	std::sort(fingerNumbers.begin(), fingerNumbers.end());
	int frequentNr;	
	int thisNumberFreq=1;
	int highestFreq=1;
	frequentNr=fingerNumbers[0];
	for(int i=1;i<fingerNumbers.size(); i++){
		if(fingerNumbers[i-1]!=fingerNumbers[i]){
			if(thisNumberFreq>highestFreq){
				frequentNr=fingerNumbers[i-1];	
				highestFreq=thisNumberFreq;
			}
			thisNumberFreq=0;	
		}
		thisNumberFreq++;	
	}
	if(thisNumberFreq>highestFreq){
		frequentNr=fingerNumbers[fingerNumbers.size()-1];	
	}
	mostFrequentFingerNumber=frequentNr;	
}

void HandGesture::addFingerNumberToVector(){
	int i=fingerTips.size();	
	fingerNumbers.push_back(i);
}

// add the calculated number of fingers to image m->src
void HandGesture::addNumberToImg(MyImage *m){
	int xPos=10;
	int yPos=10;
	int offset=30;
	float fontSize=1.5f;
	int fontFace = FONT_HERSHEY_PLAIN;
	for(int i=0;i<numbers2Display.size();i++){
		rectangle(m->src,Point(xPos,yPos),Point(xPos+offset,yPos+offset),numberColor, 2);	
		putText(m->src, intToString(numbers2Display[i]),Point(xPos+7,yPos+offset-3),fontFace,fontSize,numberColor);
		xPos+=40;
		if(xPos>(m->src.cols-m->src.cols/3.2)){
			yPos+=40;
			xPos=10;
		}
	}
}

// calculate most frequent numbers of fingers 
// over 20 frames
void HandGesture::getFingerNumber(MyImage *m){
	removeRedundantFingerTips();
	std::cout << "bounding box height = " << bRect.height << " width = " << bRect.width << std::endl;
	std::cout << "m->src.rows = " << m->src.rows << " m->src.cols = " << m->src.cols << std::endl;
	if(bRect.height > m->src.rows/2 && nrNoFinger>12 && isHand ){
		numberColor=Scalar(0,200,0);
		addFingerNumberToVector();
		if(frameNumber>12){
			// over 12 frames
			nrNoFinger=0;
			frameNumber=0;	
			computeFingerNumber();	
			numbers2Display.push_back(mostFrequentFingerNumber);
			fingerNumbers.clear();
		}else{
			frameNumber++;
		}
	}else{
		nrNoFinger++;
		numberColor=Scalar(200,200,200);
	}
	addNumberToImg(m);
	std::cout << "Number added to the image." << std::endl;
}

float HandGesture::getAngle(Point s, Point f, Point e){
	float l1 = distanceP2P(f,s);
	float l2 = distanceP2P(f,e);
	float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
	float angle = acos(dot/(l1*l2));
	angle=angle*180/PI;
	return angle;
}

void HandGesture::eleminateDefects(MyImage *m){
	int tolerance =  bRect_height/5;
	float angleTol=95; // angle threshold between two fingers
	vector<Vec4i> newDefects;
	int startidx, endidx, faridx;
	vector<Vec4i>::iterator d=defects[cIdx].begin();
	while( d!=defects[cIdx].end() ) {
   	    Vec4i& v=(*d);
	    startidx=v[0]; Point ptStart(contours[cIdx][startidx] );
   		endidx=v[1]; Point ptEnd(contours[cIdx][endidx] );
  	    faridx=v[2]; Point ptFar(contours[cIdx][faridx] );
		if(distanceP2P(ptStart, ptFar) > tolerance && distanceP2P(ptEnd, ptFar) > tolerance && getAngle(ptStart, ptFar, ptEnd  ) < angleTol ){
			if( ptEnd.y > (bRect.y + bRect.height -bRect.height/4 ) ){
			}else if( ptStart.y > (bRect.y + bRect.height -bRect.height/4 ) ){
			}else {
				newDefects.push_back(v);		
			}
		}	
		d++;
	}
	nrOfDefects=newDefects.size();
	defects[cIdx].swap(newDefects);
	removeRedundantEndPoints(defects[cIdx], m);
}

// remove endpoint of convexity defects if they are at the same fingertip
void HandGesture::removeRedundantEndPoints(vector<Vec4i> newDefects,MyImage *m){
	Vec4i temp;
	float avgX, avgY;
	float tolerance=bRect_width/6;
	int startidx, endidx, faridx;
	int startidx2, endidx2;
	for(int i=0;i<newDefects.size();i++){
		for(int j=i;j<newDefects.size();j++){
	    	startidx=newDefects[i][0]; Point ptStart(contours[cIdx][startidx] );
	   		endidx=newDefects[i][1]; Point ptEnd(contours[cIdx][endidx] );
	    	startidx2=newDefects[j][0]; Point ptStart2(contours[cIdx][startidx2] );
	   		endidx2=newDefects[j][1]; Point ptEnd2(contours[cIdx][endidx2] );
			if(distanceP2P(ptStart,ptEnd2) < tolerance ){
				contours[cIdx][startidx]=ptEnd2;
				break;
			}if(distanceP2P(ptEnd,ptStart2) < tolerance ){
				contours[cIdx][startidx2]=ptEnd;
			}
		}
	}
}

// convexity defects does not check for one finger
// so another method has to check when there are no
// convexity defects
void HandGesture::checkForOneFinger(MyImage *m){
	//int yTol=bRect.height/6;
	int yTol = bRect.height / 30;
	//int yTol = 0;
	cv::Point highestP;
	highestP.y=m->src.rows;
	vector<Point>::iterator d=contours[cIdx].begin();
	while( d!=contours[cIdx].end() ) {
   	    Point v=(*d);
		// find the point with minimum y value and assign to highestP
		if(v.y<highestP.y){
			highestP=v;
			//std::cout << "highestP.y = " << highestP.y << std::endl;
		}
		d++;	
	}int n=0;
	d=hullP[cIdx].begin();
	while( d!=hullP[cIdx].end() ) {
   	    Point v=(*d);
			//std::cout<<"v.x " << v.x << " v.y "<<  v.y << " highestP.y " << highestP.y<< " ytol "<<yTol<<std::endl;
		if(v.y<highestP.y+yTol && v.y!=highestP.y && v.x!=highestP.x){
			n++;
		}
		d++;	
	}
	
	if (n == 0) {
		// there's only 1 finger
		fingerTips.push_back(highestP);

		if (prevState != state) {
			oneFingerCoordinates.clear();
		}

		// check the distance from last fingertip
		if (oneFingerCoordinates.size() > 0) {
			if (sqrt(pow(oneFingerCoordinates.back().x - highestP.x, 2) +
				pow(oneFingerCoordinates.back().y - highestP.y, 2)) < 100) {
				if (oneFingerCoordinates.size() == 30) {
					oneFingerCoordinates.erase(oneFingerCoordinates.begin());
				}

				oneFingerCoordinates.push_back(highestP);
			}
		}
		else {
			oneFingerCoordinates.push_back(highestP);
		}

		std::cout << "highestP (1 finger) coordinates: " << highestP.x << " " << highestP.y << std::endl;

		m->fingerTipLoc = highestP; // store finger coordinates

		// check bounding condition before making a patch image 
		if ((highestP.x - 20) > 0 &&
			(highestP.y - 10) > 0 &&
			(highestP.x + 20) < m->src.cols &&
			(highestP.y + 30) < m->src.rows) {
			cv::Rect patchRect(highestP.x - 20, highestP.y - 10, 40, 40);
			cv::Mat patchImage = m->src(patchRect);
			patchImage.copyTo(m->patchImg);
			//m->patchImg = patchImage;
			cv::imwrite("..\\images\\patch_image.jpg", m->patchImg);
		}
	}
}

void HandGesture::drawFingerTips(MyImage *m){
	Point p;
	int k=0;
	for(int i=0;i<fingerTips.size();i++){
		p=fingerTips[i];
		putText(m->src,intToString(i),p-Point(0,30),fontFace, 1.2f,Scalar(200,200,200),2);
   		circle( m->src,p,   5, Scalar(100,255,100), 4 );
   	 }

	if (fingerTips.size() == 1) {
		std::cout << "oneFingerCoordinates size: " << oneFingerCoordinates.size() << std::endl;
		if (oneFingerCoordinates.size() >= 2) {
			for (int i = 0; i < oneFingerCoordinates.size() - 1; i++) {
				cv::line(m->src, oneFingerCoordinates[i], oneFingerCoordinates[i + 1], cv::Scalar(0, 255, 0), 2, 8);
			}
		}
	}
	else if (fingerTips.size() == 2) {
		if (firstFingerCoordinates.size() >= 2) {
			for (int i = 0; i < firstFingerCoordinates.size() - 1; i++) {
				cv::line(m->src, firstFingerCoordinates[i], firstFingerCoordinates[i + 1], cv::Scalar(0, 255, 0), 2, 8);
			}
		}

		if (secondFingerCoordinates.size() >= 2) {
			for (int i = 0; i < secondFingerCoordinates.size() - 1; i++) {
				cv::line(m->src, secondFingerCoordinates[i], secondFingerCoordinates[i + 1], cv::Scalar(255, 0, 0), 2, 8);
			}
		}
	}
}

void HandGesture::getFingerTips(MyImage *m){
	fingerTips.clear();
	int i=0;
	vector<Vec4i>::iterator d=defects[cIdx].begin();
	while( d!=defects[cIdx].end() ) {
   	    Vec4i& v=(*d);
	    int startidx=v[0]; Point ptStart(contours[cIdx][startidx] );
   		int endidx=v[1]; Point ptEnd(contours[cIdx][endidx] );
  	    int faridx=v[2]; Point ptFar(contours[cIdx][faridx] );
		if(i==0){
			fingerTips.push_back(ptStart);
			//std::cout << "ptStart coordinates: " << ptStart.x << " " << ptStart.y << std::endl;
			i++;
		}
		fingerTips.push_back(ptEnd);
		//std::cout << "ptEnd coordinates: " << ptEnd.x << " " << ptEnd.y << std::endl;
		d++;
		i++;
   	}

	if (fingerTips.size() == 2) {

		if (prevState != state) {
			firstFingerCoordinates.clear();
			secondFingerCoordinates.clear();
		}

		if (firstFingerCoordinates.size() == 30) {
			firstFingerCoordinates.erase(firstFingerCoordinates.begin());
		}
		if (secondFingerCoordinates.size() == 30) {
			secondFingerCoordinates.erase(secondFingerCoordinates.begin());
		}
		firstFingerCoordinates.push_back(fingerTips[0]);
		secondFingerCoordinates.push_back(fingerTips[1]);
		oneFingerCoordinates.clear();

		// find the close finger to the previous fingerTipLoc and update
		float dist1 = sqrt(pow(m->fingerTipLoc.x - fingerTips[0].x, 2) + pow(m->fingerTipLoc.y - fingerTips[0].y, 2));
		float dist2 = sqrt(pow(m->fingerTipLoc.x - fingerTips[1].x, 2) + pow(m->fingerTipLoc.y - fingerTips[1].y, 2));

		if (dist1 < dist2) {
			m->fingerTipLoc = fingerTips[0];
			m->secondTipLoc = fingerTips[1];
		}
		else {
			m->fingerTipLoc = fingerTips[1];
			m->secondTipLoc = fingerTips[0];
		}

		// create patch image
		if ((m->fingerTipLoc.x - 20) > 0 && (m->fingerTipLoc.y - 10) > 0 &&
			(m->fingerTipLoc.x + 20) < m->src.cols && (m->fingerTipLoc.y + 30) < m->src.rows) {
			cv::Rect patchRect(m->fingerTipLoc.x - 20, m->fingerTipLoc.y - 10, 40, 40);
			cv::Mat patchImage = m->src(patchRect);
			patchImage.copyTo(m->patchImg);
			//m->patchImg = patchImage;
			cv::imwrite("..\\images\\patch_image_2fingers_1.jpg", m->patchImg);
		}
		if ((m->secondTipLoc.x - 20) > 0 && (m->secondTipLoc.y - 10) > 0 &&
			(m->secondTipLoc.x + 20) < m->src.cols && (m->secondTipLoc.y + 30) < m->src.rows) {
			cv::Rect patchRect(m->secondTipLoc.x - 20, m->secondTipLoc.y - 10, 40, 40);
			cv::Mat patchImage = m->src(patchRect);
			patchImage.copyTo(m->secondPatchImg);
			//m->secondPatchImg = patchImage;
			cv::imwrite("..\\images\\patch_image_2fingers_2.jpg", m->secondPatchImg);
		}

	} else if (fingerTips.size()==0) {
		std::cout << "fingerTips.size is zero, call checkForOneFinger..." << std::endl;
		firstFingerCoordinates.clear();
		secondFingerCoordinates.clear();
		checkForOneFinger(m);
	}
	else {
		// clear all the previously stored points
		firstFingerCoordinates.clear();
		secondFingerCoordinates.clear();
		oneFingerCoordinates.clear();
		
		// find the close finger to the previous fingerTipLoc and update
		std::vector<float> dists(fingerTips.size());
		std::vector<float> secondDists(fingerTips.size());
		// calculate the distance for each fingertip
		for (int i = 0; i < fingerTips.size(); i++) {
			float dist = sqrt(pow(m->fingerTipLoc.x - fingerTips[i].x, 2) + pow(m->fingerTipLoc.y - fingerTips[i].y, 2));
			float secondDist = sqrt(pow(m->secondTipLoc.x - fingerTips[i].x, 2) + pow(m->secondTipLoc.y - fingerTips[i].y, 2));
			dists[i] = dist;
			secondDists[i] = secondDist;
		}
		// get the index of smallest distance
		int index = 0;
		int secondIndex = 0;
		for (int i = 0; i < dists.size(); i++) {
			if (dists[i] < dists[index]) {
				index = i;
			}
		}
		for (int i = 0; i < secondDists.size(); i++) {
			if (secondDists[i] < secondDists[secondIndex]) {
				secondIndex = i;
			}
		}
		// update with the closest finger 
		m->fingerTipLoc = fingerTips[index];
		m->secondTipLoc = fingerTips[secondIndex];

		// create patch image
		if ((m->fingerTipLoc.x - 20) > 0 && (m->fingerTipLoc.y - 10) > 0 &&
			(m->fingerTipLoc.x + 20) < m->src.cols && (m->fingerTipLoc.y + 30) < m->src.rows) {
			cv::Rect patchRect(m->fingerTipLoc.x - 20, m->fingerTipLoc.y - 10, 40, 40);
			cv::Mat patchImage = m->src(patchRect);
			patchImage.copyTo(m->patchImg);
			//m->patchImg = patchImage;
			cv::imwrite("..\\images\\patch_image_others_1.jpg", m->patchImg);
		}
		if ((m->secondTipLoc.x - 20) > 0 && (m->secondTipLoc.y - 10) > 0 &&
			(m->secondTipLoc.x + 20) < m->src.cols && (m->secondTipLoc.y + 30) < m->src.rows) {
			cv::Rect patchRect(m->secondTipLoc.x - 20, m->secondTipLoc.y - 10, 40, 40);
			cv::Mat patchImage = m->src(patchRect);
			patchImage.copyTo(m->secondPatchImg);
			//m->secondPatchImg = patchImage;
			cv::imwrite("..\\images\\patch_image_others_2.jpg", m->secondPatchImg);
		}
	}
}
