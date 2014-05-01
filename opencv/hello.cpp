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

cv::Mat result;

enum classified_hairstyle
{

};

void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  int threshold_value = 60;
  Canny( detected_edges, detected_edges, threshold_value, threshold_value*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  
  src.copyTo( dst, detected_edges);
  imshow( window_name, detected_edges );
 }

void edge_detection()
{
	/// Load an image
	src = imread("hello_world.jpg");

	/// Create a matrix of the same type and size as src (for dst)
	dst.create( src.size(), src.type() );

	/// Convert the image to grayscale
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Create a window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

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
	
	cv::Mat bgModel, fgModel; // 모델 (초기 사용)
	cv::grabCut (image,    // 입력 이미지
		result,    // 분할 결과
		rectangle,   // 전경을 포함하는 직사각형
		bgModel, fgModel, // 모델
		3,     // 반복 횟수
		cv::GC_INIT_WITH_RECT); // 직사각형 사용

	// cv::GC_PR_FGD 전경에 속할 수도 있는 화소(직사각형 내부의 화소 초기값)
	// cv::GC_PR_FGD와 동일한 값을 갖는 화소를 추출해 분할한 이진 영상을 얻음
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// 전경일 가능성이 있는 화소를 마크한 것을 가져오기
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	// 결과 이미지 생성
	image.copyTo(foreground, result);
	// 배경 화소는 복사되지 않음
	

	int front_hair_rower_bound_line, front_hair_rower_bound_y;
	for(int i = 0; i < result.rows; i++)
	{
		int j=0;
		while(j < result.cols)
		{
			if(result.at<uchar>(i,j) == 255)
			{
				//여기서부터 윗머리 경계 나올때까지 startingx값을 빼나갈 것이다.(여기는 앞머리 아랫쪽 경계)
				front_hair_rower_bound_line=i;
				front_hair_rower_bound_y=j;
				break;
			}
			else
			{
				j++;
			}
		}
		if(j != result.cols)
		{
			break;
		}
	}

	//윗머리 끝 좌표
	int hair_upper_edge_row=0;
	int hair_upper_edge_col=0;

	for(int i = 0; i < detected_edges.rows; i++)
	{
		int j=0;
		while(j < detected_edges.cols)
		{
			if(detected_edges.at<uchar>(i,j) == 255)
			{
				//여기가 머리 윗쪽 경계
				hair_upper_edge_row=i;
				hair_upper_edge_col=j;
				break;
			}
			else
			{
				j++;
			}
		}
		if(hair_upper_edge_col!=0 && hair_upper_edge_row!=0)
		{
			break;
		}
	}

	//턱 경계 찾아야 함
	int chin_line_bound = front_hair_rower_bound_line+1;
	
	while(chin_line_bound < result.rows)
	{
		int j=0;
		while(j < result.cols)
		{
			if(result.at<uchar>(chin_line_bound,j) == 255)
			{
				break;
			}
			else
			{
				j++;
			}
		}
		if(j == result.cols)
		{
			//여기가 턱 끝임
			chin_line_bound -= 1;
			break;
		}
		else
			chin_line_bound++;
	}

	//옆머리 아랫쪽 경계를 찾아야 함
	//오른쪽 옆머리의 두께를 위에서부터 내려가며 측정해나가며 0이 되는 순간 그곳의 row, column 좌표를 기록한다
	int left_hair_width_warning=0;
	int left_hair_lower_bound=front_hair_rower_bound_line;
	while(left_hair_lower_bound < chin_line_bound)
	{
		//우선 result의 col 좌표를 찾아야함
		int face_left_boundary=0;
		while(face_left_boundary < result.cols)
		{
			if(result.at<uchar>(left_hair_lower_bound,face_left_boundary) == 255)
			{
				break;
			}
			else
				face_left_boundary++;
		}
		//찾은 col 왼쪽 경계에서부터 detected_edges에 있는 옆 경계까지의 거리를 측정해야 함
		int hair_left_boundary = face_left_boundary;
		while(hair_left_boundary>0)
		{
			if(detected_edges.at<uchar>(left_hair_lower_bound,hair_left_boundary) == 255)
			{
				break;
			}
			else
				hair_left_boundary--;
		}
		int left_hair_width = face_left_boundary - hair_left_boundary;
		if(left_hair_width < 2)
		{
			break;
		}

		left_hair_lower_bound++;
	}

	//옆머리 기장 판정
	if( left_hair_lower_bound < chin_line_bound && left_hair_lower_bound >= hair_upper_edge_row + (chin_line_bound - hair_upper_edge_row)*(2.0/3.0) )
	{
		//옆머리 짧다
	}
	else if(left_hair_lower_bound >= chin_line_bound)
	{
		//옆머리 길다
	}
	else
	{
		//옆머리 없다
	}

	//앞머리 기장 판정
		

	//이쪽에서 아마 result 가 흰색이고 detected_edges에서도 흰색인 부분을 찾는다면 두 이미지를 합성하여 머리 경계를 찾아낼수 있을 것이다
	for(int i = 0; i < detected_edges.rows; i++)
	{
		int j=0;
		while(j < detected_edges.cols)
		{
			if(result.at<uchar>(i,j) != 255 && detected_edges.at<uchar>(i,j) == 255)
				result.at<uchar>(i,j) = 255;
			
			j++;
			// do something with BGR values...
		}
	}
	cv::namedWindow("Result");
	cv::imshow("Result", result);
	

}

int main()
{
	const char *faceclassifer = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml";
	CvHaarClassifierCascade* facecascade = 0;
	facecascade = (CvHaarClassifierCascade*) cvLoad(faceclassifer, 0, 0, 0 );

	const char *eyeclassifer = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml";
	CvHaarClassifierCascade* eyecascade = 0;
	eyecascade = (CvHaarClassifierCascade*) cvLoad(eyeclassifer, 0, 0, 0 );

	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);

	//edge detection 먼저
	edge_detection();

	//이 아래는 얼굴 인식
	
	//이미지를 로드
	IplImage *frame = cvLoadImage("hello_world.jpg",CV_LOAD_IMAGE_COLOR);
	
	//얼굴 검출
	CvSeq *faces = cvHaarDetectObjects(frame, facecascade, storage, 1.4 , 1, 0);

	//검출된 얼굴 Rectangle 받아오기(얼굴은 하나만 입력되고 인식됐다고 가정)
	CvRect *face_rect = 0;
	face_rect = (CvRect*) cvGetSeqElem(faces, 0);


	//얼굴 이미지에서 눈을 찾아야 할 것 같음
	cv::Mat face_mat= cv::imread("hello_world.jpg");

	// Transform it into the C++ cv::Mat format
	cv::Mat image(face_mat); 

	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	CvRect quarter_of_face;
	quarter_of_face.x = face_rect -> x;
	quarter_of_face.y = face_rect -> y;
	quarter_of_face.width = (face_rect -> width)*0.5;
	quarter_of_face.height = (face_rect -> height)*0.5;


	cv::Mat cropped_face = image(quarter_of_face);

	//cropped_face를 IplImage로 변환해야 함
	IplImage* face_iplimage = &IplImage(cropped_face);

	//face_iplimage에서 눈깔을 찾는다
	CvSeq *eye = cvHaarDetectObjects(face_iplimage, eyecascade, storage, 1.4 , 1, 0);

	//검출된 눈깔 Rectangle 받아오기(눈알은 하나만 입력되고 인식됐다고 가정)
	CvRect *eye_rect = 0;
	eye_rect = (CvRect*) cvGetSeqElem(eye, 0);

	//eye_rect의 좌표를 기억해놓아서 앞머리의 상대적 길이를 판정할 때 도움을 받아야 함 
	cvRectangle(frame, cvPoint( face_rect->x + eye_rect->x, (face_rect->y)+(eye_rect->y) ), cvPoint(face_rect->x + eye_rect->x+eye_rect->width, (face_rect->y)+(eye_rect->y)+eye_rect->height), cvScalar(0,255,0), 3, CV_AA, 0);
	

	
	
	cvRectangle(frame, cvPoint(face_rect->x, face_rect->y), cvPoint(face_rect->x+face_rect->width, face_rect->y+face_rect->height), cvScalar(0,255,0), 3, CV_AA, 0);
	grab_cut(face_rect->x, face_rect->y + (0.1)*(face_rect->height), face_rect->width, face_rect->height);//얼굴 크롭하기



	//Window에 frame 출력
	cvShowImage("haar example (exit = esc)",frame);
	cvWaitKey(0);


	//자원 해제
	cvReleaseMemStorage(&storage);
	cvReleaseHaarClassifierCascade(&facecascade);
	cvDestroyWindow("haar example (exit = esc)");
	
	return 0;
}