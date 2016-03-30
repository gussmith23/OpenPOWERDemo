#include "cmt\CMT.h"
#include "opencv2\opencv.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace cmt;
using namespace cv;
using namespace std;

// function declarations
void processFrame(Mat image);
void receiveData();
void receiveData();
void initializeTracker(Mat image);
void setObjectToFind(Mat image);

bool trackerInitialized = false;
bool running = true;

CMT cmt_tracker;

/**
 * Fields needed for initial object detection with SURF
 */ 
int minHessian = 400;
SurfFeatureDetector detector;
SurfDescriptorExtractor extractor;

// If there's no tracker yet, should we be attempting object detection?
bool attemptObjectDetection = false;

// The model image, keypoints, and descriptors of the object to detect
Mat object_image;
vector<KeyPoint> object_keypoints;
Mat object_descriptors;

// Scene keypoints and descriptors
vector<KeyPoint> scene_keypoints;
Mat scene_descriptors;

double lastArea = -1.0f;
Point2d lastLocation = Point2d(-1, -1);
double scaleDelta = 1.0f;
Point2f positionDelta = Point2f(1.0f, 1.0f);
double scaleDeltaThreshold = 0.1f;
Point2f positionDeltaThreshold = Point2f(0.1f, 0.1f);

int main()
{

	cmt_tracker = CMT();
	detector = SurfFeatureDetector(minHessian);
	extractor = SurfDescriptorExtractor();

	/*
	Server connection/delegate registration goes here
	*/


	/////////// TEST CODE for testing from webcam
	Mat frostedFlakes = imread("itemsToTrack/Frosted Flakes.jpg");
	setObjectToFind(frostedFlakes);

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.0f);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480.0f);
	

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		processFrame(frame);
	
		
		 
	}


	/////////// END TEST CODE


	while (running); 
	
}

void processFrame(Mat image)
{
	Mat im_gray;
	cvtColor(image, im_gray, CV_BGR2GRAY);

	if (!trackerInitialized)
	{
		if (attemptObjectDetection)
		{
			initializeTracker(im_gray);
		}
	} 
	else
	{
		cmt_tracker.processFrame(im_gray);
	}

	if (trackerInitialized)
	{
		Rect roi = cmt_tracker.bb_rot.boundingRect();
		rectangle(image, roi, Scalar(0, 255, 0));
	}

	imshow("test", image);
	waitKey(30);
	
}

void receiveData()
{

	Mat receivedImage;

	/*
	Getting frame and converting it goes here
	*/

	
	processFrame(receivedImage);
}

/**
 * Find ROI given a greyscale image.
 * This function will need to be called multiple times on successive frames
 *	before initializing tracker, as it has to confirm that it's finding the same
 *  object multiple times.
 * Based, as always, on OpenCV's example code: 
 * http://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html?highlight=surf
 */
void initializeTracker(Mat im_gray)
{
	Mat scene_image = im_gray;

	#warning SURF call
	detector.detect(scene_image, scene_keypoints);
	#warning SURF call
	extractor.compute(scene_image, scene_keypoints, scene_descriptors);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(object_descriptors, scene_descriptors, matches);

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < object_descriptors.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < object_descriptors.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(object_keypoints[good_matches[i].queryIdx].pt);
		scene.push_back(scene_keypoints[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(object_image.cols, 0);
	obj_corners[2] = cvPoint(object_image.cols, object_image.rows); obj_corners[3] = cvPoint(0, object_image.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	int minX = 10000, maxX = 0, minY = 10000, maxY = 0;

	for (int i = 0; i < 4; i++)
	{
		Point2f pt = scene_corners[i];
		int x = pt.x, y = pt.y;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
	}

	Rect roi = Rect(minX, minY, maxX - minX, maxY - minY);

	// Debug code
	//Mat image_to_show = scene_image;
	//rectangle(image_to_show, roi, Scalar(255, 0,0));
	//imshow("ROI found by SURF", image_to_show);
	//waitKey();

	// Validity of ROI checking goes here.
	bool roiValid = false;

	// Get area and center location
	double area = roi.area();
	Point2d center = Point2d(roi.x + roi.width / 2, roi.y + roi.height / 2);

	// If we have previous measurements 
	if (lastArea >= 0.0f && lastLocation.x >= 0 && lastLocation.y >= 0)
	{
		positionDelta.x = abs(lastLocation.x - center.x) / scene_image.cols;
		positionDelta.y = abs(lastLocation.y - center.y) / scene_image.rows;
		scaleDelta = abs(lastArea - area) / lastArea;

		// Log
		printf("New deltas: position (%f, %f), scale %f\n", positionDelta.x, positionDelta.y, scaleDelta);

		// Check if valid
		if (scaleDelta < scaleDeltaThreshold && positionDelta.x < positionDeltaThreshold.x && positionDelta.y < positionDeltaThreshold.y)
		{
			roiValid = true;
			cout << "ROI determined to be valid!" << endl;
		}
		else
		{
			cout << "ROI determined to be invalid!" << endl;
		}

	}

	// Log the current measurements.
	lastArea = area;
	lastLocation = center;

	// Finally, initialize if SURF worked appropriately.
	if (roiValid)
	{
		cmt_tracker.initialize(scene_image, roi);
		trackerInitialized = true;
	}
	
}

void setObjectToFind(Mat image)
{
	object_image = image;

	#warning SURF call
	detector.detect(object_image, object_keypoints);
	#warning SURF call
	extractor.compute(object_image, object_keypoints, object_descriptors);

	attemptObjectDetection = true;
}