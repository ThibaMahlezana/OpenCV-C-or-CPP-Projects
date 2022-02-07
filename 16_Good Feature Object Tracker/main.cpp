#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    int numCorners;
    istringstream iss(argv[1]);
    iss >> numCorners;

    if(numCorners < 1){
        numCorners = 1;
    }

    RNG rng(12345);
    string windowName = "Feature based Tracker";

    // Current frame
    Mat frame, frameGray;

    char ch;

    // Creating the capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cout << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    // Scaling factor to resize the input frames from webcam
    float scalingFactor = 0.75;

    while(true){
        // Capture the current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        // Initialize the parameters for Shi-Tomasi algorithm
        vector<Point2f> corners;
        double qualityThreshold = 0.02;
        double minDist = 15;
        int blockSize = 5;
        bool useHarrisDetector = false;
        double k = 0.07;

        // Clone the input frame
        Mat frameCopy;
        frameCopy = frame.clone();

        // Apply corner detection
        goodFeaturesToTrack(frameGray, corners, numCorners, qualityThreshold, minDist, Mat(), blockSize, useHarrisDetector, k);

        int radius = 8;
        int thickness = 2;
        int lineType = 8;

        // Draw the detected corners using circles
        for(size_t i = 0; i < corners.size(); i++){
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
            circle(frameCopy, corners[i], radius, color, thickness, lineType, 0);
        }

        // Displaying the frame
        imshow(windowName, frameCopy);

        if(waitKey(30) == 27) {
            break;
        }
    }
    // Release the video capture object
    cap.release();

    // Close all windows
    destroyAllWindows();

    return 1;
}