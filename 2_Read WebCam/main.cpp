#include <iostream>

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main(){
    // Creating video capture object
    VideoCapture cap(0);

    // Checking if the webcam is opened
    if(!cap.isOpened()){
        cout << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    string windowsName = "Read webcam";
    namedWindow(windowsName, 1);

    while(true){
        // Creating frame object
        Mat frame;

        // Capturing current frame
        cap >> frame;

        // Check if the frame is not empty
        if(frame.empty){
            return 0;
        }

        // Displaying the video
        imshow(windowsName, frame);

        // Exit if the user pressed ESC Key
        if(waitKey(0) == 27){
            break;
        }
    }

    // Releasing webcam
    cap.release();

    // Close all windows
    destroyAllWindows();

    return 0;
}