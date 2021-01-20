#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void detectAndDisplay( Mat frame );

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, char** argv) {
    int camera_device=0;
    
    if( !face_cascade.load("haarcascade_frontalface_default.xml" ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load("haarcascade_eye.xml" ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
	};
        VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
        Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }


	return 0;

}

int getRadius(Rect eye){
    return cvRound( (eye.width + eye.height)*0.25 );
}

void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    frame = frame_gray;
    //-- Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        int eyesMax[2]={-1,-1};
        eyes_cascade.detectMultiScale( faceROI, eyes );
        if(eyes.size()>=2){
            if(getRadius(eyes[0])>=getRadius(eyes[1])){
                eyesMax[0]=0;
                eyesMax[1]=1;
            }else{
                eyesMax[0]=1;
                eyesMax[1]=0;
            }
        }
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            int radius = getRadius(eyes[j]);
            if(eyes.size()>=2){
                if(j!=eyesMax[0] && j!=eyesMax[1])
                    if(radius > getRadius(eyes[eyesMax[0]])){
                        eyesMax[0] = j;
                    }
                    else if(radius > getRadius(eyes[eyesMax[1]])){
                        eyesMax[1] = j;    
                    }
            }else{
                Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
                //circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
            }
        }
        if(eyes.size()>=2){
            for ( size_t j = 0; j < 2; j++ ){
                int radius = getRadius(eyes[eyesMax[j]]);
             Point eye_center( faces[i].x + eyes[eyesMax[j]].x + eyes[eyesMax[j]].width/2, faces[i].y + eyes[eyesMax[j]].y + eyes[eyesMax[j]].height/2 );
                //circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );   
                
                //iris detection
                Mat eyeROI = faceROI( eyes[eyesMax[j]] );
                Canny(eyeROI, eyeROI,35,3*35);
                vector<Vec3f> circles;
                HoughCircles(eyeROI, circles, HOUGH_GRADIENT, 1,60,200, 20, 0, 0);
                for( size_t i = 0; i < circles.size(); i++ ){
                    cout << "circles detected"<<endl;
                    Vec3i c = circles[i];
                    Point center = Point(faces[i].x + eyes[eyesMax[j]].x + c[0], faces[i].y + eyes[eyesMax[j]].y + c[1]);
                            circle( frame, center, 1,Scalar(0,100,100), 3, LINE_AA);
                            int radius = c[2];
                            circle( frame, center, radius, Scalar(255,0,255), 3, LINE_AA);
                            
                }
            }
        }
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
