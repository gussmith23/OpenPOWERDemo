#include "cmt\CMT.h"
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace cmt;
using namespace cv;
using namespace std;

bool trackerInitialized = false;
bool running = true;

CMT cmt;

/**
 * Fields needed for initial object detection with SURF
 */ 
int minHessian = 400;
SurfFeatureDetector features;
SurfDescriptorExtractor descriptors;

// If there's no tracker yet, should we be attempting object detection?
bool attemptObjectDetection = false;

// The model image, keypoints, and descriptors of the object to detect
Mat object_image;
vector<KeyPoint> object_keypoints;
OutputArray object_descriptors;

// Scene keypoints and descriptors
vector<KeyPoint> scene_keypoints;
OutputArray scene_descriptors;


int main()
{
	
	cmt = CMT();
	features = SurfFeatureDetector(minHessian);
	descriptors = SurfDescriptorExtractor(); 

	/*
	Server connection/delegate registration goes here
	*/

	while (running); 

}

void doWorkOnData()
{
	/*
	Getting frame and converting it goes here
	*/


}

void initializeTracker()
{

}