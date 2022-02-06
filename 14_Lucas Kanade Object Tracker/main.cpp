#include <iostream>

#include "opencv2/video/tracking.hpp";
#include "opencv2/imgproc/imgproc.hpp";
##include "opencv2/highgui/highgui.hpp";

using namespace std;
using namespace cv;

bool pointTrackingFlag = false;
Point2f currentPoint;

// Function to detect mouse event
void onMouse(int event, int x, int y, int void*){
    // Detect the mouse buttom events
    if(event == EVENT_LBUTTONDOWN){
        // Assign the current (x, y) position to currentPoint
        currentPoint = Point2f((float)x, (float)y);

        // Set the tracking flag
        pointTrackingFlag = true;
    }
}

int main(int argc, char* argv){
    // Creating the capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cout << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    // Termination criteria for tracking the points
    TermCriteria terminationCriteria(TermCriteria::COUT | TermCriteria::EPS, 10, 0.02);

    // Size of the block that is used for matching
    Size windowSize(25, 25);

    // Maximum number of points that you want to  track
    const int maxNumPoints = 200;

    string windowName = "Lucas Kanade Object Tracker";
    namedWindow(windowName, 1);
    setMouseCallBack(windowName, onMouse,  0);

    Mat prevGrayImage, curGrayImage, image, frame;
    vector<Point2f> trackingPoints[2];

    // Image size scaling factor for the input frames
    float scalingFactor = 0.75;

    while(true){
        // Capture the current frame
        cap >> frame;

        // Check if the frame is empty
        if(frame.empty()){
            break;
        }

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Copy the input frame
        frame.copyTo(image);

        // Check if there are points to track
        if(!trackingPoints[0].empty()){
            // Status vector to indicate wether the flow for the corresponding features has been found
            vector<uchar> statusVector;

            // Error vector to indicate the error for the corresponding feature
            vector<float> errorVector;

            // Check if previous image is empty
            if(prevGrayImage.empty){
                curGrayImage.copyTo(prevGrayImage);
            }

            // Calculating the optical flow
            calcOpticalFlowPyrLK(prevGrayImage, curGrayImage, trackingPoints[0], trackingPoints[1], statusVector, errorVector, windowSize, 3, terminationCriteria, 0, 0.001);

            int count = 0;

            // Minimum distance between any two tracking points
            int minDist = 7;

            for(int i = 0; i < trackingPoints[1].size(); i++){
                if(pointTrackingFlag){
                    if(norm(currentPoint - trackingPoints[1][i]) <= minDist){
                        pointTrackingFlag = false;
                        continue;
                    }
                }

                // Check if the status vector is good
                if(!statusVector[i]){
                    continue;
                }

                trackingPoints[1][count++] = trackingPoints[1][i];

                // Draw a filled circle for each of the tracking points
                int radius = 8;
                int thickness = 2;
                int lineType = 8;
                circle(image, trackingPoints[1][i], radius, Scalar(0, 255, 0), thickness, lineType);
            }
            trackingPoints[1].resize(count);
        }

        // Refining the location of the feature points
        if(pointTrackingFlag && trackingPoints[1].size() < maxNumPoints){
            vector<Point2f> tempPoints;
            tempPoints.push_back(currentPoint);

            cornerSubPix(curGrayImage, tempPoints, windowSize, Size(-1,-1), terminationCriteria);

            trackingPoints[1].push_back(tempPoints[0]);
            pointTrackingFlag = false;
        }

        // Display the image
        imshow(windowName, image);

        if(waitKey(10) == 27){
            break;
        }

        // Swap the points vectors to update previous to current
        std::swap(trackingPoints[1], trackingPoints[0]);

        // Swap the points vectors to update previous to current
        cv::swap(prevGrayImage, curGrayImage);
    }

    return 0;
}