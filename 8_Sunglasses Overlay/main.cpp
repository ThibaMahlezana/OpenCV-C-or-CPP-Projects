#include <iostream>

// OpenCV Includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"

#define CV_HAAR_SCALE_IMAGE 2

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    string faceCascadeName = argv[1];
    string eyeCascadeName = argv[2];
    CascadeClassifier faceCascade, eyeCascade;

    // Check if face Haarcascade file exists and loads it
    if(!faceCascade.load(faceCascadeName)){
        cerr << "[INFO] Error loading face cascade file." << endl;
        return -1;
    }

    // Check if eye Haarcascade file exists and loads it
    if(!eyeCascade.load(eyeCascadeName)){
        cerr << "[INFO] Errorloading eye cascade file." << endl;
        return -1;
    }

    Mat eyeMask = imread(argv[3]);
    if(!eyeMask.data){
        cerr << "[INFO] Error loading mask image." << endl;
        return -1;
    }

    Mat frame, frameGray;
    Mat frameROI, eyeMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedEye, maskedFrame;

    char ch;

    // Creating the capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cerr << "[INF] Unable to open webcam" << endl;
        return -1;
    }

    // Creating GUI windows
    namedWindow("Sunglasses Overlay");

    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;

    vector<Rect> faces;

    while(true){
        // Capture the current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frameGray, COLOR_BGR2GRAY);

        // Equalize the histogram
        equalizeHist(frameGray, frameGray);

        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

        vector<Point> centers;

        // Draw green circles around the eyes
        for(int i = 0; i < faces.size(); i++){
            Mat faceROI = frameGray(faces[i]);
            vector<Rect> eyes;

            // In each face, detect eyes
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));

            // In each eye detected, compute the center
            for(int j = 0; j < eyes.size(); j++){
                Point center(faces[i].x + eyes[j].x + int(eyes[j].width*0.5), faces[i].y + eyes[j].y + int(eyes[j].height*0.5));
                centers.push_back(center);
            }
        }

        // Overlay sunglasses only if both eyes are detected
        if(centers.size() == 2){
            Point leftPoint, rightPoint;
            
            // Identify the left and right eyes
            if(centers[0].x < centers[1].x)
            {
                leftPoint = centers[0];
                rightPoint = centers[1];
            }
            else
            {
                leftPoint = centers[1];
                rightPoint = centers[0];
            }
            
            // Custom parameters to make the sunglasses fit your face. You may have to play around with them to make sure it works.
            int w = 2.3 * (rightPoint.x - leftPoint.x);
            int h = int(0.4 * w);
            int x = leftPoint.x - 0.25*w;
            int y = leftPoint.y - 0.5*h;
            
            // Extract region of interest (ROI) covering both the eyes
            frameROI = frame(Rect(x,y,w,h));
            
            // Resize the sunglasses image based on the dimensions of the above ROI
            resize(eyeMask, eyeMaskSmall, Size(w,h));
            
            // Convert the above image to grayscale
            cvtColor(eyeMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);
            
            // Threshold the above image to isolate the foreground object
            threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV);
            
            // Create mask by inverting the above image (because we don't want the background to affect the overlay)
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
            
            // Use bitwise "AND" operator to extract precise boundary of sunglasses
            bitwise_and(eyeMaskSmall, eyeMaskSmall, maskedEye, grayMaskSmallThresh);
            
            // Use bitwise "AND" operator to overlay sunglasses
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
            
            // Add the above masked images and place it in the original frame ROI to create the final image
            add(maskedEye, maskedFrame, frame(Rect(x,y,w,h)));
        }

        // Displaying the current frame
        imshow("Sunglasses Overlay", frame);

        // Exiting when ESC key is pressed
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