#include <iostream>

// OpenCV includes
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

int main(){

    // Creating the capture object
    cout << "[INFO] Initializing webcam ..."
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cout << "[INFO] Unable to open the webcam." << endl;
        return -1;
    }

    Mat frame, hsvImage, mask, outputImage;
    char  ch;

    // Image size scaling factor for the input frames from the webcam
    float scalingFactor = 0.75;

    whil(true){

        // Initilizing the output image
        outputImage = Scalar(0, 0, 0);

        // Capture the current frame
        cap >> frame;

        // check if frame is empty
        if(frame.empty()) { break; }

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to HSV colorspace
        cvtColor(frame, hsvImage, COLOR_BGR2HSV);

        // Define the range of blue color in HSV colorspace
        Scalar lowerLimit = Scalar(60, 100, 100);
        Scalar upperLimit = Scalar(180, 255, 255);

        // Threshold the HSV image to get only blue color
        inRange(hsvImage, lowerLimit, upperLimit, mask);

        // Compute bitwise-AND of input image and mask
        bitwise_and(frame, frame, outputImage, mask=mask);

        // Run median filter on the output to smoothen it
        medianBlur(outputImage, outputImage, 5);

        // Display the input and output image
        imshow("Input", frame);
        imshow("Output", outputImage);

        if(waitKey(30) == 27) { break; }
    }

    return 0;
}