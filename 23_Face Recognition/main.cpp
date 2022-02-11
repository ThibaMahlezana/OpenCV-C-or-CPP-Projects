// Starndard Library
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

// OpenCV Library
#include "opencv2/opencv.hpp"

// Custom libraries
#include "detectObject.h"
#include "preprocessFace.h"
#include "recognition.h"

using namespace cv;
using namespace std;

const char* facerecAlgorithm = "FaceRecognizer.Eigenfaces";
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;

const char* faceCascadeFilename = "lbpcascade_frontalface.xml";
const char* eyeCascadeFilename1 = "haarcascade_eye.xml";
const char* eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml";

const int faceWidth = 70;
const int faceHeight = faceWidth;

const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;

const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;
const char* windowName = "Face Recognition";


const bool preprocessLeftAndRightSeparately = true;
bool m_debug = false;

// Running mode for the Webcam-based interactive GUI program
enum MODES {
    MODE_STARTUP,
    MODE_DETECTION,
    MODE_COLLECTION,
    MODE_COLLECT_FACES,
    MODE_TRAINING,
    MODE_RECOGNITION,
    MODE_DELETE_ALL,
    MODE_END
};

const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!" };
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

// Position of GUI buttons
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

template <typename T> string toString(T t){
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t){
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

// Loading the face and 1 or 2 eye detection XML classifiers
void initDetectors(CascadeClassifier& faceCascade, CascadeClassifier& eyeCascade1, CascadeClassifier& eyeCascade2){
    try{
        faceCascade.load(faceCascadeFilename);
    }
    catch(cv::Exception& e){}

    if(faceCascade.empty()){
        cerr << "[ONFO] ERROR Could not load Face Detection cascade classifier" << endl;
        exit(1);
    }
    cout << "[INFO] loaded the Face Detection cascade classifier" << endl;

    try{
        eyeCascade1.load(eyeCascadeFilename1);
    }
    catch(cv::Exception& e){}

    if(eyeCascadeFilename1.empty()){
        cerr << "[INFO] ERROR Could not load 1st Eye Detection cascade classifier" << endl;
        exit(1);
    }

    try{
        eyeCascade2.load(eyeCascadeFilename2);
    }
    catch(cv::Exception& e){}

    if(eyeCascade2.empty()){
        cerr << "[INFO] Error Could not load 2nd Eye Detection cascade classifier" << endl;
    }
    else{
        cout << "Loaded the 2nd Eye Detection cascade classifier" << endl;
    }
}

// Getting access to the webcam
void initWebcam(VideoCapture& videoCapture, int cameraNumber){
    // Get access to the default camera
    try{
        VideoCapture.open(cameraNumber);
    }
    catch(cv::Exception& e){}

    if(!videoCapture.isOpened()){
        cerr << "[INFO] ERROR could not access the camera!" << endl;
        exit(1);
    }
    cout << "[INFO] loaded camera" << endl;
}

// Writting text to an image
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX){
    // Getting the text size & baseline
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    if(coord.y >= 0){
        coord.y += textSize.height;
    }
    else{
        coord.y += img.rows - baseline + 1;
    }

    if(coord.x < 0){
        coord.x += img.cols - textSize.width + 1;
    }

    // Getting the bounding box around the text
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text
    putText(img, text, coord, fontFace, fontScale, color, thickness, LINE_AA);

    return boundingRect;
}

// Drawing button into the image
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0){
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0, 0, 0));
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2 * B, rcText.height + 2 * B);

    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;

    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);

    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200, 200, 200), 1, LINE_AA);

    // Draw the actual text that will be displayed, using anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10, 55, 20));

    return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc){
    if(pt.x >= rc.x &&  pt.x <= (rc.x + rc.width - 1)){
        if(pt.y >= rc.y && pt.y <= (rc.y + rc.height -1)){
            return true
        }
    }
    return false;
}

void onMouse(int event, int x, int y, int void*){
    if (event != EVENT_LBUTTONDOWN){
        return;
    }

    // Check if the user clicked on one of the GUI buttons
    Point pt = Point(x,  y);

    // Add Person Button
    if(isPointInRect(pt, m_rcBtnAdd)){
        cout << "[INFO] User clicked [Add Button]" << endl;
        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons - 1] >= 0)) {
            // Add a new person.
            m_numPersons++;
            m_latestFaces.push_back(-1);
            cout << "Num Persons: " << m_numPersons << endl;
        }
        // Use the newly added person. Also use the newest person even if that person was empty.
        m_selectedPerson = m_numPersons - 1;
        m_mode = MODE_COLLECT_FACES;
    }

    // Delete button
    else if(isPointInRect(pt, m_rcBtnDel)){
        cout << "[INFO] User clicked [Delete All] button" << endl;
        m_mode = MODE_COLLECT_FACES;
    }

    // Debug button
    else if (isPointInRect(pt, m_rcBtnDebug)) {
        cout << "[INFO] User clicked [Debug] button." << endl;
        m_debug = !m_debug;
        cout << "[INFO] Debug mode: " << m_debug << endl;
    }

    else{
        cout << "[INFO] User clicked on the image" << endl;

        // Check if the user clicked on one of the faces in the list.
        int clickedPerson = -1;
        for (int i = 0; i < m_numPersons; i++) {
            if (m_gui_faces_top >= 0) {
                Rect rcFace = Rect(
                    m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                if (isPointInRect(pt, rcFace)) {
                    clickedPerson = i;
                    break;
                }
            }
        }

        // Change the selected person, if the user clicked on a face in the GUI.
        if (clickedPerson >= 0) {
            // Change the current person, and collect more photos for them.
            m_selectedPerson = clickedPerson;
            m_mode = MODE_COLLECT_FACES;
        }

        // Otherwise they clicked in the center.
        else {
            // Change to training mode if it was collecting faces.
            if (m_mode == MODE_COLLECT_FACES) {
                cout << "[INFO] User wants to begin training." << endl;
                m_mode = MODE_TRAINING;
            }
        }
    }
}

void recognizeAndTrainUsingWebcam(VideoCapture& videoCapture, CascadeClassifier& faceCascade, CascadeClassifier& eyeCascade1, CascadeClassifier& eyeCascade2){
    Ptr<BasicFaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
    double old_time = 0;

    m_mode = MODE_DETECTION;

    while(true){
        Mat cameraFrame;
        VideoCapture >> cameraFrame;
        if(cameraFrame.empty()){
            cerr << "[INFO] Error Could not grab the frame" << endl;
            exit(1);
        }

        Mat displayedFrame;
        cameraFrame.copyTo(displayFrame);

        int identity = -1;

        // Find a face and preprocess it to have a standard size and contrast & brightness
        Rect faceRect;
        Rect searchedLeftEye, searchedRightEye;

        Point leftRect, rightEye;
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade,
            eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye,
            &rightEye, &searchedLeftEye, &searchedRightEye);Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade,
            eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye,
            &rightEye, &searchedLeftEye, &searchedRightEye);

        bool gotFaceAndEyes = false;
        if(preprocessedFace.data){
            gotFaceAndEyes = true;
        }

        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, LINE_AA);

            // Draw light-blue anti-aliased circles for the 2 eyes.
            Scalar eyeColor = CV_RGB(0, 255, 255);
            if (leftEye.x >= 0) { // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6,
                    eyeColor, 1, LINE_AA);
            }
            if (rightEye.x >= 0) { // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6,
                    eyeColor, 1, LINE_AA);
            }
        }

        if (m_mode == MODE_DETECTION) {}

        // Collect faces mode
        else if(m_mode == MODE_COLLECT_FACES){
            // Check if we have detected a face.
            if (gotFaceAndEyes) {

                // Check if this face looks somewhat different from the previously collected face.
                double imageDiff = 10000000000.0;
                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }

                // Also record when it happened.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

                // Only process the face if it is noticeably different from the previous frame and
                // there has been noticeable time gap.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION)
                    && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
                    // Also add the mirror image to the training set, so we have more training data,
                    // as well as to deal with faces looking to the left or right.
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);

                    // Add the face images to the list of detected faces.
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedPerson);
                    faceLabels.push_back(m_selectedPerson);

                    // Keep a reference to the latest face of each person.
                    m_latestFaces[m_selectedPerson]
                        = preprocessedFaces.size() - 2; // Point to the non-mirrored face.
                    // Show the number of collected faces. But since we also store mirrored faces,
                    // just show how many the user thinks they stored.
                    cout << "Saved face " << (preprocessedFaces.size() / 2) << " for person "
                         << m_selectedPerson << endl;

                    // Make a white flash on the face, so the user knows a photo has been taken.
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90, 90, 90);

                    // Keep a copy of the processed face, to compare on next iteration.
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
        
        // Trainging Mode
        else if (m_mode == MODE_TRAINING) {

            // Check if there is enough data to train from. For Eigenfaces, we can learn just one
            // person if we want, but for Fisherfaces, we need atleast 2 people otherwise it will
            // crash!
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
                if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0)) {
                    cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is "
                            "nothing to differentiate! Collect more data ..."
                         << endl;
                    haveEnoughData = false;
                }
            }
            if (m_numPersons < 1 || preprocessedFaces.size() <= 0
                || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Warning: Need some training data before it can be learnt! Collect more "
                        "data ..."
                     << endl;
                haveEnoughData = false;
            }

            if (haveEnoughData) {
                // Start training from the collected faces using Eigenfaces or a similar algorithm.
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);

                // Show the internal face recognition data, to help debugging.
                if (m_debug)
                    showTrainingDebugData(model, faceWidth, faceHeight);

                // Now that training is over, we can start recognizing!
                m_mode = MODE_RECOGNITION;
            } 
            else {
                // Since there isn't enough training data, go back to the face collection mode!
                m_mode = MODE_COLLECT_FACES;
            }

        }

        // Recognition Mode
        else if(m_mode == MODE_RECOGNITION){
            if (gotFaceAndEyes && (preprocessedFaces.size() > 0)
                && (preprocessedFaces.size() == faceLabels.size())) {

                // Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);
                if (m_debug)
                    if (reconstructedFace.data)
                        imshow("reconstructedFace", reconstructedFace);

                // Verify whether the reconstructed face looks like the preprocessed face, otherwise
                // it is probably an unknown person.
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identify who the person is in the preprocessed face image.
                    identity = model->predict(preprocessedFace);
                    outputStr = toString(identity);
                } else {
                    // Since the confidence is low, assume it is an unknown person.
                    outputStr = "Unknown";
                }
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

                // Show the confidence rating for the recognition in the mid-top of the display.
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
                Point ptTopLeft = Point(cx - 15, BORDER);
                // Draw a gray line showing the threshold for an "unknown" person.
                Point ptThreshold = Point(
                    ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y),
                    CV_RGB(200, 200, 200), 1, LINE_AA);
                // Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence
                    = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                // Show the light-blue confidence bar.
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0, 255, 255), FILLED,
                    LINE_AA);
                // Show the gray border of the bar.
                rectangle(
                    displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200, 200, 200), 1, LINE_AA);
            }
        }

        // Delete all mode
        else if(m_mode == MODE_DELETE_ALL){
            // Restart everything!
            m_selectedPerson = -1;
            m_numPersons = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_prepreprocessedFace = Mat();

            // Restart in Detection mode.
            m_mode = MODE_DETECTION;
        }
        else {
            cerr << "[INFO] Error Invalid run mode" << endl;
            exit(1);
        }

        // Show the help, while also showing the number of collected faces
        string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION)
            help = "Click [Add Person] when ready to collect faces.";
        else if (m_mode == MODE_COLLECT_FACES)
            help = "Click anywhere to train from your " + toString(preprocessedFaces.size() / 2)
                + " faces of " + toString(m_numPersons) + " people.";
        else if (m_mode == MODE_TRAINING)
            help = "Please wait while your " + toString(preprocessedFaces.size() / 2) + " faces of "
                + toString(m_numPersons) + " people builds.";
        else if (m_mode == MODE_RECOGNITION)
            help = "Click people on the right to add more faces to them, or [Add Person] for "
                   "someone new.";
        if (help.length() > 0) {
            // Draw it with a black background and then again with a white foreground.
            // Since BORDER may be 0 and we need a negative position, subtract 2 from the border so
            // it is always negative.
            float txtSize = 0.4;
            drawString(displayedFrame, help, Point(BORDER, -BORDER - 2), CV_RGB(0, 0, 0),
                txtSize); // Black shadow.
            rcHelp = drawString(displayedFrame, help, Point(BORDER + 1, -BORDER - 1),
                CV_RGB(255, 255, 255), txtSize); // White text.
        }

        // Show the current mode.
        if (m_mode >= 0 && m_mode < MODE_END) {
            string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
            drawString(displayedFrame, modeStr, Point(BORDER, -BORDER - 2 - rcHelp.height),
                CV_RGB(0, 0, 0)); // Black shadow
            drawString(displayedFrame, modeStr, Point(BORDER + 1, -BORDER - 1 - rcHelp.height),
                CV_RGB(0, 255, 0)); // Green text
        }

        // Show the current preprocessed face in the top-center of the display.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Get a BGR version of the face, since the output is BGR color.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, COLOR_GRAY2BGR);
            // Get the destination ROI (and make sure it is within the image!).
            // min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            // Copy the pixels from src to dst.
            srcBGR.copyTo(dstROI);
        }

        // Draw an anti-aliased border around the face, even if it is not shown.
        rectangle(displayedFrame, Rect(cx - 1, BORDER - 1, faceWidth + 2, faceHeight + 2),
            CV_RGB(200, 200, 200), 1, LINE_AA);

        // Draw the GUI buttons into the main image.
        m_rcBtnAdd = drawButton(displayedFrame, "Add Person", Point(BORDER, BORDER));
        m_rcBtnDel = drawButton(displayedFrame, "Delete All",
            Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
        m_rcBtnDebug = drawButton(displayedFrame, "Debug",
            Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);

        // Show the most recent face for each of the collected people, on the right side of the
        // display.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i = 0; i < m_numPersons; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    // Get a BGR version of the face, since the output is BGR color.
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, COLOR_GRAY2BGR);
                    // Get the destination ROI (and make sure it is within the image!).
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    // Copy the pixels from src to dst.
                    srcBGR.copyTo(dstROI);
                }
            }
        }

        // Highlight the person being collected, using a red rectangle around their face.
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight,
                    displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255, 0, 0), 3, LINE_AA);
            }
        }

        // Highlight the person that has been recognized, using a green rectangle around their face.
        if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0, 255, 0), 3, LINE_AA);
        }

        // Show the camera frame on the screen.
        imshow(windowName, displayedFrame);

        // If the user wants all the debug data, show it to them!
        if (m_debug) {
            Mat face;
            if (faceRect.width > 0) {
                face = cameraFrame(faceRect);
                if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) {
                    Mat topLeftOfFace = face(searchedLeftEye);
                    Mat topRightOfFace = face(searchedRightEye);
                    imshow("topLeftOfFace", topLeftOfFace);
                    imshow("topRightOfFace", topRightOfFace);
                }
            }

            if (!model.empty())
                showTrainingDebugData(model, faceWidth, faceHeight);
        }
        char keypress = waitKey(20);

        if (keypress == VK_ESCAPE) { 
            break;
        }
    }
}

int main(int argc, char* argv[]){
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;

    // Load the face and 1 or 2 eye detection XML classifiers.
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    int cameraNumber = 0;
    if(argc > 1){
        cameraNumber = atoi(argv[1]);
    }

    // Getting acces to the webcam
    initWebcam(videoCapture, cameraNumber);

    // Setting camera resolution
    videoCapture.set(CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen
    namedWindow(windowName);
    setMouseCallback(windowName, onMouse, 0);

    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

    return 0;
}