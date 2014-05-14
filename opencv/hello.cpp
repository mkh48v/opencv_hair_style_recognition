#include <iostream>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat detected_edges;

int ratio = 3;
int kernel_size = 3;

int eye_upper_bound;

enum front_hair_style
{
	no_front_hair,
	short_front_hair,
	long_front_hair
};
enum side_hair_style
{
	short_left_hair,
	long_left_hair
};

void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur( src_gray, detected_edges, Size(3,3) );

	int threshold_value = 60;
	Canny( detected_edges, detected_edges, threshold_value, threshold_value*ratio, kernel_size );

	cv::namedWindow("Result");
	cv::imshow("Result", detected_edges);

}

void edge_detection()
{
	/// Load an image
	src = imread("hello_world.jpg");
	
	/// Convert the image to grayscale
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Show the image
	CannyThreshold(0, 0);
}



int get_left_front_hair_lower_bound(int head_upper_bound, int left_eye_right_bound) 
{
	int front_left_hair_lower_bound = head_upper_bound + (4*(eye_upper_bound-head_upper_bound)/5.0);


	while(front_left_hair_lower_bound > head_upper_bound + (eye_upper_bound-head_upper_bound)/2.0)
	{
		if(detected_edges.at<uchar>( front_left_hair_lower_bound, left_eye_right_bound ) == 255)
		{
			break;
		}
		else
		{
			front_left_hair_lower_bound--;
		}
	}


	if(front_left_hair_lower_bound <=  head_upper_bound + (eye_upper_bound-head_upper_bound)/2.0)
	{
		//앞머리가 긴거임(여기까지 오도록 경계가 높을리가 없음)
		front_left_hair_lower_bound = head_upper_bound + (4*(eye_upper_bound-head_upper_bound)/5.0);
	}
	return front_left_hair_lower_bound;
}

int get_right_front_hair_lower_bound(int head_upper_bound, int right_eye_left_bound) 
{
	int front_right_hair_lower_bound = head_upper_bound + (13*(eye_upper_bound-head_upper_bound)/15.0);

	while(front_right_hair_lower_bound > head_upper_bound + (eye_upper_bound-head_upper_bound)/2.0 + 5)
	{
		if(detected_edges.at<uchar>( front_right_hair_lower_bound, right_eye_left_bound ) == 255)
		{
			break;
		}
		else
		{
			front_right_hair_lower_bound--;
		}
	}

	if(front_right_hair_lower_bound <= head_upper_bound + (eye_upper_bound-head_upper_bound)/2.0 + 5)
	{
		//앞머리가 긴거임(여기까지 오도록 경계가 높을리가 없음)
		front_right_hair_lower_bound = head_upper_bound + (13*(eye_upper_bound-head_upper_bound)/15.0);
	}
	return front_right_hair_lower_bound;
}



int get_head_upper_bound() 
{
	int hair_upper_boundary_row = 0;
	int hair_upper_edge_col=0;
	for(int i = 0; i < detected_edges.rows; i++)
	{
		int j=0;
		while(j < detected_edges.cols)
		{
			if(detected_edges.at<uchar>(i,j) == 255)
			{
				//여기가 머리 윗쪽 경계
				hair_upper_boundary_row=i;
				hair_upper_edge_col=j;
				break;
			}
			else
			{
				j++;
			}
		}
		if(hair_upper_edge_col!=0 && hair_upper_boundary_row!=0)
		{
			break;
		}
	}
	return hair_upper_boundary_row;
}

int get_chin_line_bound(CvRect* face_rect) 
{
	int chin_line_bound = (face_rect->y + face_rect->height + 1);

	while( chin_line_bound < (int)(face_rect->y + face_rect->height * (5/4.0)) )
	{
		if(detected_edges.at<uchar>( chin_line_bound, (int)( face_rect->x + (face_rect->width/2.0) ) ) == 255)
		{
			break;
		}
		else
		{
			chin_line_bound++;
		}
	}
	if( chin_line_bound >= (int)(face_rect->y + face_rect->height * (5/4.0)) )
	{
		chin_line_bound = (face_rect->y + face_rect->height + 1);
	}
	return chin_line_bound;
}



int get_left_hair_lower_bound(int left_hair_lower_bound, int chin_line_bound, int face_left_boundary, int detection_bound) 
{
	//초기화하는 부분
	int face_left_boundary_save = face_left_boundary;
	while(left_hair_lower_bound < chin_line_bound)
	{
		while(face_left_boundary > detection_bound)
		{
			if(detected_edges.at<uchar>(left_hair_lower_bound,face_left_boundary) == 255)
			{
				break;
			}
			else
			{
				face_left_boundary--;
			}
		}
		if(face_left_boundary == detection_bound)
		{
			face_left_boundary = face_left_boundary_save;
			left_hair_lower_bound++;
			continue;
		}
		else
		{
			break;
		}
	}
	int hair_left_boundary = face_left_boundary - 7;
	


	//왼쪽 머리 아랫쪽 경계 찾기
	int hair_left_bound_save = hair_left_boundary;
	while(left_hair_lower_bound < chin_line_bound)
	{
		//머리 왼쪽 경계 찾는 부분
		while(hair_left_boundary>detection_bound)
		{
			if(detected_edges.at<uchar>(left_hair_lower_bound,hair_left_boundary) == 255)
			{
				break;
			}
			else
			{
				hair_left_boundary--;
			}
		}
		if(hair_left_boundary == detection_bound)
		{
			hair_left_boundary = hair_left_bound_save;
			left_hair_lower_bound++;
			continue;
		}
		else
		{
			hair_left_bound_save = hair_left_boundary;
		}



		int left_hair_width = face_left_boundary - hair_left_boundary;
		if(left_hair_width < 2)
		{
			break;
		}
		else//왼쪽머리 오른쪽 경계 찾는 부분
		{
			left_hair_lower_bound++;
			face_left_boundary_save = face_left_boundary;
			face_left_boundary = hair_left_boundary + 1;
			while(face_left_boundary < detected_edges.cols)
			{
				if(detected_edges.at<uchar>(left_hair_lower_bound,face_left_boundary) == 255)
				{
					break;
				}
				else
				{
					face_left_boundary++;
				}
			}

			if(face_left_boundary == detected_edges.cols)
			{
				face_left_boundary = face_left_boundary_save;
				hair_left_boundary = face_left_boundary - 1;
				left_hair_lower_bound++;
				continue;
			}

			int left_hair_width = face_left_boundary - hair_left_boundary;
			if(left_hair_width < 2)
			{
				break;
			}
			else
			{
				hair_left_boundary = face_left_boundary-1;
			}
		}
	}
	//루프 끝



	return left_hair_lower_bound;
}



void show_recommended_hairstyle( front_hair_style detected_front_hair, side_hair_style detected_side_hair ) 
{
	//앞머리 긴 여자
	if(detected_front_hair == long_front_hair && detected_side_hair == long_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontlong_sidelong.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}
	else if(detected_front_hair == long_front_hair && detected_side_hair == short_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontlong_sidemedium.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}

	//앞머리 짧은 여자
	else if(detected_front_hair == short_front_hair && detected_side_hair == long_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontmedium_sidelong.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}
	else if(detected_front_hair == short_front_hair && detected_side_hair == short_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontmedium_sidemedium.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}

	//앞머리 없는 여자
	else if(detected_front_hair == no_front_hair && detected_side_hair == long_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontshort_sidelong.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}
	else if(detected_front_hair == no_front_hair && detected_side_hair == short_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontshort_sidemedium.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}
}

side_hair_style judge_side_hairstyle( int left_hair_lower_bound, int chin_line_bound, int hair_upper_boundary_row) 
{
	side_hair_style detected_side_hair;
	//옆머리 기장 판정
	if( left_hair_lower_bound < chin_line_bound && left_hair_lower_bound >= hair_upper_boundary_row + (chin_line_bound - hair_upper_boundary_row)*(4.0/5.0) )
	{
		//옆머리 짧다
		detected_side_hair = short_left_hair;
	}
	else if(left_hair_lower_bound >= chin_line_bound)
	{
		//옆머리 길다
		detected_side_hair = long_left_hair;
	}
	else
	{
		detected_side_hair = short_left_hair;
	}
	return detected_side_hair;
}

front_hair_style judge_front_hairstyle(int front_hair_lower_bound, int hair_upper_boundary_row) 
{
	front_hair_style detected_front_hair;
	//앞머리 기장 판정
	if( front_hair_lower_bound > hair_upper_boundary_row + 2*(eye_upper_bound-hair_upper_boundary_row)/3.0 )
	{
		//앞머리 길다
		detected_front_hair = long_front_hair;
	}
	else if(
		front_hair_lower_bound <= hair_upper_boundary_row + 2*(eye_upper_bound-hair_upper_boundary_row)/3.0
		&& front_hair_lower_bound > hair_upper_boundary_row + (eye_upper_bound-hair_upper_boundary_row)/2.0
		)
	{
		//앞머리 짧다
		detected_front_hair = short_front_hair;
	}
	else
	{
		//앞머리 없다
		detected_front_hair =  no_front_hair;
	}
	return detected_front_hair;
}

int main()
{
	const char *faceclassifer = "haarcascade_frontalface_alt.xml";
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
	IplImage *pic = cvLoadImage("hello_world.jpg",CV_LOAD_IMAGE_COLOR);
	
	//얼굴 검출
	CvSeq *detected_face = cvHaarDetectObjects(pic, facecascade, storage, 1.4 , 1, 0);

	//검출된 얼굴 Rectangle 받아오기(얼굴은 하나만 입력되고 인식됐다고 가정)
	CvRect *face_rect = 0;
	face_rect = (CvRect*) cvGetSeqElem(detected_face, 0);

	//얼굴 이미지에서 눈을 찾아야 할 것 같음
	cv::Mat face_mat= cv::imread("hello_world.jpg");
	// Transform it into the C++ cv::Mat format
	cv::Mat image(face_mat); 


	CvRect left_quarter_of_face;
	left_quarter_of_face.x = face_rect -> x;
	left_quarter_of_face.y = face_rect -> y;
	left_quarter_of_face.width = (face_rect -> width)*0.5;
	left_quarter_of_face.height = (face_rect -> height)*0.5;

	CvRect right_quarter_of_face;
	right_quarter_of_face.x = face_rect -> x + (face_rect -> width)*0.5;
	right_quarter_of_face.y = face_rect -> y;
	right_quarter_of_face.width = (face_rect -> width)*0.5;
	right_quarter_of_face.height = (face_rect -> height)*0.5;

	cv::Mat cropped_left_face = image(left_quarter_of_face);
	cv::Mat cropped_right_face = image(right_quarter_of_face);

	//cropped_face를 IplImage로 변환해야 함
	IplImage* left_face_iplimage = &IplImage(cropped_left_face);
	IplImage* right_face_iplimage = &IplImage(cropped_right_face);

	//face_iplimage에서 눈깔을 찾는다
	CvSeq* left_eye = cvHaarDetectObjects(left_face_iplimage, eyecascade, storage, 1.4 , 1, 0);
	CvSeq* right_eye = cvHaarDetectObjects(right_face_iplimage, eyecascade, storage, 1.4 , 1, 0);

	//검출된 눈깔 Rectangle 받아오기(눈알은 하나만 입력되고 인식됐다고 가정)
	CvRect *left_eye_rect = 0;
	left_eye_rect = (CvRect*) cvGetSeqElem(left_eye, 0);

	CvRect *right_eye_rect = 0;
	right_eye_rect = (CvRect*) cvGetSeqElem(right_eye, 0);


	//eye_rect의 좌표를 기억해놓아서 앞머리의 상대적 길이를 판정할 때 도움을 받아야 함 
	cvRectangle(pic, cvPoint( face_rect->x + left_eye_rect->x, (face_rect->y)+(left_eye_rect->y) ), cvPoint(face_rect->x + left_eye_rect->x+left_eye_rect->width, (face_rect->y)+(left_eye_rect->y)+left_eye_rect->height), cvScalar(0,255,0), 3, CV_AA, 0);

	cvRectangle(pic, cvPoint(face_rect->x, face_rect->y), cvPoint(face_rect->x+face_rect->width, face_rect->y+face_rect->height), cvScalar(0,255,0), 3, CV_AA, 0);



	eye_upper_bound = (face_rect->y) + (left_eye_rect->y);
	
	//윗머리 끝 라인
	int hair_upper_boundary_row = get_head_upper_bound();

	//앞머리 라인
	int front_hair_lower_bound = get_left_front_hair_lower_bound(hair_upper_boundary_row, face_rect->x + left_eye_rect->x + left_eye_rect->width);

	//회사에서 부수적으로 요구하는 정보
	int front_hair_left_lower_bound = get_left_front_hair_lower_bound(hair_upper_boundary_row, face_rect->x + left_eye_rect->x + left_eye_rect->width);
	int front_hair_right_lower_bound = get_right_front_hair_lower_bound(hair_upper_boundary_row, face_rect->x + (int)(face_rect->width / 2.0) + right_eye_rect->x);

	int chin_line_bound = get_chin_line_bound(face_rect);



	int left_hair_lower_bound = get_left_hair_lower_bound((int) ( eye_upper_bound + (chin_line_bound - eye_upper_bound)*(3.0/5.0) - 1 ), chin_line_bound, face_rect->x + left_eye_rect->x + (left_eye_rect->width/2.0), face_rect->x+3);



	side_hair_style detected_side_hair = judge_side_hairstyle(left_hair_lower_bound, chin_line_bound, hair_upper_boundary_row);

	front_hair_style detected_front_hair = judge_front_hairstyle(front_hair_lower_bound, hair_upper_boundary_row);

	show_recommended_hairstyle(detected_front_hair, detected_side_hair);

	cvShowImage("haar example (exit = esc)",pic);
	cvWaitKey(0);

	cvDestroyWindow("haar example (exit = esc)");

	return 0;
}