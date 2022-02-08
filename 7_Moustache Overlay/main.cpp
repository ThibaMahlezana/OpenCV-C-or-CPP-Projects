#include <iostream>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"

#define CV_HAAR_SCALE_IMAGE 2

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){

    // Haarcascade files
    string faceCascadeName(argv[1]);
    string mouthCascadeName(argv[2]);

    CascadeClassifier faceCascade, mouthCascade;

    // Check if face haarcascade file exists and load
    if(!faceCascade.load(faceCascadeName)){
        cerr << "[INFO] Error loading face cascade file." << endl;
    }

    // Check if mouth haarcascade file file exist and load
    if(!mouthCascade.load(mouthCascadeName)){
        cerr << "[INFO] Error loading mouth cascade file." << endl;
    }

    Mat mouthMask = imread(argv[3]);
    if(!mouthMask.data){
        cerr << "[INFO] Error loading moustach image." << endl;
        return -1;
    }

    Mat frame, frameGray;
    Mat frameROI, mouthMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedMouth, maskedFrame;

    // Create the capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cout << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    // Creating GUI windows
    namedWindow("Moustach Overlay");

    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;

    vector<Rect> faces;

    while(true){
        // Capture the current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to grayscale
        cvtColor(frame, frameGray, COLOR_BRG2GRAY);

        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

        vector<Point> centers;

        // Looking for mouth in the face ROI
        for(auto& face:faces){

            Mat faceROI = frameGray(face);
            vector<Rect> mouths;

            for(auto& mouth : mouths){
                Point center(face.x + mouth.x + int(mouth.width*0.5), face.y + mouth.y + int(mouth.height*0.5));
                int radius = int((mouth.width + mouth.height) * 0.25);

                // Overlay moustache
                int w = 1.8 * mouth.width;
                int h = mouth.height;
                int x = face.x + mouth.x - 0.2*w;
                int y = face.y + mouth.y + 0.65*h;
                
                frameROI = frame(Rect(x,y,w,h));
                resize(mouthMask, mouthMaskSmall, Size(w,h));
                cvtColor(mouthMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);
                threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV);
                bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
                bitwise_and(mouthMaskSmall, mouthMaskSmall, maskedMouth, grayMaskSmallThresh);
                bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
                add(maskedMouth, maskedFrame, frame(Rect(x,y,w,h)));
            }
        }

        // Displaying the current frame
        imshow("Moustache Overlay", frame);

        // Exit the program when ESC key is pressed
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