// MOTION DETECTOR WITH FRAME DIRRERENCING

#include <iostream>
#include <sstream>

// OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat frameDiff(Mat prevFrame, Mat curFrame, Mat nextFrame){
    Mat diffFrames1, diffFrames2, output;

    // Compute absolute difference between current frame and the next frame
    absdiff(nextFrame, curFrame, diffFrames1);

    // Compute absolute difference between current frame and the previous frame
    absdiff(curFrame, prevFrame, diffFrames2);

    // Bitwise AND operation between the above two diff images
    bitwise_and(diffFrames1, diffFrames2, output);

    return output;
}

Mat getFrame(VideoCapture cap, float scalingFactor){
    Mat frame, output;

    // Capture the current frame
    cap >> frame;

    // Resize the frame
    resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

    // Convert to grayscale
    cvtColor(frame, output, COLOR_BGR2GRAY);

    return output;
}

int main(int argc, char* argv[]){
    Mat frame, prevFrame, curFrame, nextFrame;
    char ch;

    // Creating the capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cerr << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    // Creating GUI windows
    namedWindow("Motion Detector");

    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;

    prevFrame = getFrame(cap, scalingFactor);
    curFrame = getFrame(cap, scalingFactor);
    nextFrame = getFrame(cap, scalingFactor);

    while(true){
        // Showing the object movement
        imshow("Object Movement", frameDiff(preFrame, curFrame, nextFrame));

        // Update the variables and grab the next frame
        prevFrame = curFrame;
        curFrame = nextFrame;
        nextFrame = getFrame(cap, scalingFactor);

        // Exit when ESC Key is pressed
        if(waitKey(0) == 27){
            break;
        }
    }

    // Releasing the video capture object
    cap.release();

    // Closing all windows
    destroyAllWindows();

    return -1;
}