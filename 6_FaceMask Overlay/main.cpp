#include <iostream>

// OpenCV Includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/object/object.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){

    string faceCascadeName = argv[1];
    CascadeClassifier faceCascade;

    if(!faceCascade.load(faceCascadeName)){
        cerr << "[INFO] Error loading cascade file." << endl;
        return -1;
    }

    Mat faceMask = imread(argv[2]);
    if(!faceMask.data){
        cerr << "[INFO] Error loading mask image." << endl;
    }

    // Current frame
    Mat frame, frameGray;
    Mat frameROI, faceMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, greyMaskSmallThreshInv;
    Mat maskFace, maskedFrame;

    VideoCapture cap(0);

    if(!cap.isOpened()){
        return -1;
    }

    // Creating GUI window
    string windowName = "Face Mask Overlay";
    namedWindow(windowName, 0);

    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;

    vector<Rect> faces;

    while(true){
        // Capture the current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to grayscale
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        // Equalize the histogram
        equalizeHist(frameGray, frameGray);

        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|2, Size(30, 30));

        // Draw green rectangle around the face
        for(auto& face: faces){
            Rect faceRect(face.x, face.y, face.width, face.height);

            int x = face.x - int(0.1 * face.width);
            int y = face.y - int(0.0 * face.height);
            int w = int(1.1 * face.width);
            int h = int(1.3 * face.height);

            // Extract region of interest (ROI) covering your face
            frameROI = frame(Rect(x,y,w,h));
            
            // Resize the face mask image based on the dimensions of the above ROI
            resize(faceMask, faceMaskSmall, Size(w,h));
            
            // Convert the above image to grayscale
            cvtColor(faceMaskSmall, grayMaskSmall, COLOR_BGR2GRAY);
            
            // Threshold the above image to isolate the pixels associated only with the face mask
            threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, THRESH_BINARY_INV);
            
            // Create mask by inverting the above image (because we don't want the background to affect the overlay)
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
            
            // Use bitwise "AND" operator to extract precise boundary of face mask
            bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh);
            
            // Use bitwise "AND" operator to overlay face mask
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
            
            // Add the above masked images and place it in the original frame ROI to create the final image
            if(x>0 && y>0 && x+w < frame.cols && y+h < frame.rows){
                add(maskedFace, maskedFrame, frame(Rect(x,y,w,h)));
            }

            // Displaying the current frame
            imshow(windowName, frame);

            if(waitKey(0) == 27){
                break;
            }
        }

        // Releasing the video capture
        cap.release();

        // Close all windows
        destroyAllWindows();

        return -1;
    }
}