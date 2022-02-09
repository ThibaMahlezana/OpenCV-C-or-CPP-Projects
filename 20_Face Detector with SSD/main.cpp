#include <iostream>
#include <opencv2/dnn.hpp>
#include <opnecv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

int main(){

    String modelConfiguration = "data/res10_300x300_ssd_iter_140000.caffemodel";
    String modelBinary = "data/deploy.prototxt.txt";

    dnn::Net net = reaNetFromCaffe(modelConfiguration, modelBinary);

    if(net.empty()){
        cerr << "[INFO] Cannot load model files" << endl;
        exit(-1);
    }

    // Creating video capture object
    VideoCapture cap(0);

    if(!cap.isOpened()){
        cerr << "[INFO] Unable to open webcam" << endl;
        return -1;
    }

    while(true){
        // Creating frame object
        Mat frame;

        // Capturing current frame
        cap >> frame;
        
        // Check if frame is not empty
        if(frame.empty){
            break;
        }

        if(frame.channels() == 4){
            cvtColor(frame, frame, COLOR_BGR2BGR);
        }

        // Preparing blob
        Mat inputBlob = blobFromImage(frame, inScaleFactor, Size(inWidth, inHeight), meanVal, false, false);

        // Setting input blob
        net.setInput(inputBlob, "data");

        // Making a forward pass
        Mat detection = net.forward("detection_out");

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        ostringstream ss;
        ss << "FPS: " << 1000 / time << " ; time: " << time << "ms";
        putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));

        float confidenceThreshold = 0.75;

        for(int i = 0; i < detectionMat.rows; i++){
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold){
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = "Face: " + conf;
                int baselLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }
        // Displaying current frame
        imshow("Face Detector", frame);

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