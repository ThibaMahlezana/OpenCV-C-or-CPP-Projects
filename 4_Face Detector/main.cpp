#include <iostream>

// OpenCV Includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/object/objectdetect.hpp"

using namespace std;
using namespace  cv;

int main(int argc, char* argv[]){
    
    // Haarscade file
    string faceCascadeName = argv[1];
    CascadeClassifier faceCascade;

    // Check and load Haarscade file
    if(!faceCascade.load(faceCascadeName)){
        cerr << "[INFO] Error loading cascade file." << endl;
        return -1;
    }

    Mat faceMask = imread(argv[2]);

    // Current frame
    Mat frame, frameGray;
    Mat frameROI, faceMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskThreshIv;
    Mat maskedFace, maskedFrame;

    // Creating capture object
    VideoCapture cap(0);

    // Check if webcam was opened
    if(!cap.isOpned()){
        cout << "[INFO] Unable to open webcam." << endl;
        return -1;
    }

    // Creating GUI window
    string windowName = "Face Detector";
    namedWindow(windowName, 1);

    // Scaling factor to resize the input frames from webcam
    float scalingFactor = 0.75;

    vector<Rect> faces;

    while(true){
        // Capture current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frame. Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to grayscale
        cvtColor(frame,frameGray, COLOR_BGR2GRAY);

        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|2, Size(30, 30));

        // Draw green rectangle aroundthe face
        for(auto& face: faces){
            Rect faceRect(face.x, face.y, face.width, face.height);
        }

        // Show the current frame
        imshow(windowName, frame);

        if(waitKey(0) == 27){
            break;
        }
    }

    // Release webcam
    cap.release();

    // Close all windows
    destroyAllWindows();

    return 0;
}