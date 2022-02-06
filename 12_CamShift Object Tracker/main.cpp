#include <iostream>
#include <cctype>

// OpenCV library
#include "opencv2/video/tracking.hpp";
#include "opencv2/imgproc/imgproc.hpp";
#include "opencv2/highgui/highgui.hpp";

using namespace cv;
using namespace std;

Mat image;
Point originPoint;
Rect selectedRect;
bool selectedRegion = false;
int trackingFlag = 0;

// Function to track the mouse events
void onMouse(int event, int x, int y, int, void*){
    if(selectedRegion){
        selectedRect.x = MIN(x, originPoint.x);
        selectedRect.y = MIN(y, originPoint.y);
        selectedRect.width = std::abs(x - originPoint.x);
        selectedRect.height = std::abs(y - originPoint.y);

        selectedRect &= Rect(0, 0, image.cols, image.rows);
    }

    switch(event){
        case EVENT_LBUTTONDOWN:
            originPoint = Point(x, y);
            selectedRect = Rect(x, y, 0, 0);
            selectedRegion =  true;
            break;
        case EVENT_BUTTONUP:
            selectedRegion = false;
            if(selectedRect.width > 0 && selectedRect.height > 0){
                trackingFlag = -1;
            }
            break;
    }
}

int main(){
    // Creating capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cout << "[INFO] Unable to open the webcam" << endl;
        return -1;
    }

    char ch;
    Rect trackingRect;
    float hueRanges[] = {0, 180};
    const float* histRanges = hueRanges;
    
    int minSaturation = 40;
    int minValue = 20; maxValue = 245;

    int histSize = 8;

    string windowName = "CAMShift Object Tracker";
    namedWindow(windowName, 0);
    setMouseCallBack(windowName, onMouse, 0);

    Mat frame, hsvImage, hueImage, mask, hist,  backproj;

    float scalingFactor = 0.75;

    while(true){
        
        // Capture the current frame
        cap >> frame;

        if(frame.empty) { break; }

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor. INTER_AREA);

        // Clone the input frame
        frame.copyTo(image);

        // Convert to HSV colorspace
        cvtColor(image, hsvImage, COLOR_BGR2HSV);

        if(trackingFlag){
            // Check for all the values in hsvimage that are within the specified range
            inRange(hsvImage, Scalar(0, minSaturation, minValue), Scalar(180, 256, maxValue), mask);

            // Mix the sspecified channels
            int channels[] = {0, 0};
            hueImage.create(hsvImage.size(), hsvImage.depth());
            mixChannels(&hsvImage, 1, &hueImage, 1, channels, 1);

            if(trackingFlag < 0){
                // Create images based on selected regions of interest
                Mat roi(hueImage, selectedRect);
                Mat maskroi(mask, selectedRect);

                // Compute the histogram and normalise it
                calcHist(&roi, 1, 0, maskroi, hist, 1, &histSize, &histRanges);
                normalize(hist, hist, 0, 255, NORM_MINMAX);

                trackingRect = selectedRect;
                trackingFlag = 1;
            }

            // Compute the histogram back projection 
            calcBackProject(&hueImage, 1, 0, hist, backproj, &histRanges);
            backproj &= mask;
            RotatedRect rotatedTrackingRect = CamShift(backproj, trackingRect, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

            // Check if the area of trackingRect is too small
            if(trackingRect.area() <= 1){
                // Use an offset value to make sure the trackingRect has a minimum size
                int cols = backproj.cols; 
                int rows = backproj.rows;
                int offset = MIN(rows, cols) + 1;
                trackingRect = Rect(trackingRect.x - offset, trackingRect.y - offset, trackingRect.x + offset, trackingRect.y + offset) & Rect(0, 0, cols, rows);
            }
            // Draw the ellipse on top of the image
            ellipse(image, rotatedTrackingRect, Scalar(0, 255, 0), 3, LINE_AA);
        }
        // Apply the negative effect on the selected region of interest
        if(selectedRegion && selectedRegion.width > 0 && selectedRect.height > 0){
            Mat roi(image, selectedRect);
            bitwise_not(roi, roi);
        }

        // Dispay the ouput image
        imshow(windowName, image);
        if(waitKey(30) == 27){ break; }
    }

    return 0;
}
