#include <iostream>
#include <string>
#include <sstream>

// OpenCV includes
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat color_img = imread("African_Art.jpg");

    // Check if the image exist
    if(!color_img.data){
        cout << "[INFO] Could not open or find the image" << endl;
        return -1;
    }

    // Show the image
    imshow("African Art Image", color_img);

    // Wait for any key press
    waitKey(0);
    
    return 0;
}