#include <iostream>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
 }

void face_recognition()
{

}

void edge_detection()
{
	/// Load an image
	src = imread("hello_world.jpg");

	if( !src.data )
	{
		return;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );

	/// Convert the image to grayscale
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Create a window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	/// Create a Trackbar for user to enter threshold
	createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

	/// Show the image
	CannyThreshold(0, 0);
}

int main()
{
	const char *classifer = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml";

	CvHaarClassifierCascade* cascade = 0;
	cascade = (CvHaarClassifierCascade*) cvLoad(classifer, 0, 0, 0 );

	if(!cascade)
	{
		std::cerr<<"error: cascade error!!"<<std::endl;
		return -1;
	}

	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);

	if(!storage)
	{
		std::cerr<<"error: storage error!!"<<std::endl;
		return -2;
	}

	//edge detection 먼저
	edge_detection();


	//이 아래는 얼굴 인식
	
	//이미지를 로드
	IplImage *frame = cvLoadImage("hello_world.jpg",CV_LOAD_IMAGE_COLOR);
	
	//얼굴 검출
	CvSeq *faces = cvHaarDetectObjects(frame, cascade, storage, 1.4, 1, 0);

	//검출된 얼굴 Rectangle 그리기
	for(int i=0; i<faces->total; i++){
		CvRect *r = 0;
		r = (CvRect*) cvGetSeqElem(faces, i);
		cvRectangle(frame, cvPoint(r->x, r->y), cvPoint(r->x+r->width, r->y+r->height), cvScalar(0,255,0), 3, CV_AA, 0);
	}

	//Window에 frame 출력
	cvShowImage("haar example (exit = esc)",frame);
	cvWaitKey(0);

	//자원 해제
	cvReleaseMemStorage(&storage);
	cvReleaseHaarClassifierCascade(&cascade);
	cvDestroyWindow("haar example (exit = esc)");
	return 0;
}