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
  imshow( window_name, detected_edges );
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

void grab_cut(int x, int y, int width, int height)
{
	cv::Mat image= cv::imread("hello_world.jpg");
	cv::namedWindow("Original Image");
	cv::imshow("Original Image",image);

	// 입력 영상에 대한 부분 전경/배경 레이블을 지정하는 방법
	// 전경 객체 내부를 포함하는 내부 직사각형을 정의
	cv::Rect rectangle(x, y, width, height);
	// 경계 직사각형 정의
	// 직사각형 밖의 화소는 배경으로 레이블링

	// 입력 영상과 자체 분할 영상 외에 cv::grabCut 함수를 호출할 때
	// 이 알고리즘에 의해 만든 모델을 포함하는 두 행렬의 정의가 필요
	cv::Mat result; // 분할 (4자기 가능한 값)
	cv::Mat bgModel, fgModel; // 모델 (초기 사용)
	cv::grabCut (image,    // 입력 영상
		result,    // 분할 결과
		rectangle,   // 전경을 포함하는 직사각형
		bgModel, fgModel, // 모델
		5,     // 반복 횟수
		cv::GC_INIT_WITH_RECT); // 직사각형 사용
	// cv::CC_INT_WITH_RECT 플래그를 이용한 경계 직사각형 모드를 사용하도록 지정

	// cv::GC_PR_FGD 전경에 속할 수도 있는 화소(직사각형 내부의 화소 초기값)
	// cv::GC_PR_FGD와 동일한 값을 갖는 화소를 추출해 분할한 이진 영상을 얻음
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// 전경일 가능성이 있는 화소를 마크한 것을 가져오기
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	// 결과 영상 생성
	image.copyTo(foreground, result);
	// 배경 화소는 복사되지 않음

	cv::namedWindow("Result");
	cv::imshow("Result", result);

	cv::namedWindow("Foreground");
	cv::imshow("Foreground", foreground);
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
	CvSeq *faces = cvHaarDetectObjects(frame, cascade, storage, 1.4 , 1, 0);

	
	//검출된 얼굴 Rectangle 그리기
	for(int i=0; i<faces->total; i++)//얼굴 인식된 개수가 faces->total로 들어감. 여기서 우리가 쓸 사진엔 얼굴 한개밖에 없으므로 그닥 상관 없을듯
	{
		CvRect *r = 0;
		r = (CvRect*) cvGetSeqElem(faces, i);
		cvRectangle(frame, cvPoint(r->x, r->y), cvPoint(r->x+r->width, r->y+r->height), cvScalar(0,255,0), 3, CV_AA, 0);

		grab_cut(r->x, r->y + (0.1)*(r->height), r->width, r->height);//얼굴 크롭하기
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