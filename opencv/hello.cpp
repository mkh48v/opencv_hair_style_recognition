#include <opencv/cv.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <windows.h>

using namespace cv;

int eye_upper_bound;

enum front_hair_style
{
	no_front_hair,
	long_front_hair,
	left_front_long_right_front_short,
	left_front_short_right_front_long
};
enum side_hair_style
{
	short_left_hair,
	long_left_hair,
	tied_hair
};

int get_head_upper_bound(CvRect* face_rect)
{
	return face_rect->y - (int)(face_rect->height * 0.5);
}

int get_front_hair_lower_bound(cv::Mat gray_scale_mat, int hair_upper_bound, int vertical_standard_line, int horizontal_lower_bound) 
{
	int horizontal_lower_bound_save = horizontal_lower_bound;
	//처음 horizontal_lower_bound로부터 위의 세 픽셀의 gray_scale값의 평균을 낸다.
	double criteria_color = 0;
	criteria_color += gray_scale_mat.at<uchar>(horizontal_lower_bound,vertical_standard_line);
	criteria_color += gray_scale_mat.at<uchar>(horizontal_lower_bound - 1,vertical_standard_line);
	criteria_color += gray_scale_mat.at<uchar>(horizontal_lower_bound - 2,vertical_standard_line);
	criteria_color /= 3.0;

	while(horizontal_lower_bound > hair_upper_bound + (horizontal_lower_bound_save - hair_upper_bound)/3 )
	{
		if( gray_scale_mat.at<uchar>(horizontal_lower_bound,vertical_standard_line) < criteria_color - 100)
		{
			return horizontal_lower_bound;
		}
		else
		{
			horizontal_lower_bound--;
		}
	}
	//만약 hair_upper_bound까지 머리 색깔의 변화가 없다면, 앞머리는 매우 긴 것이다.
	return horizontal_lower_bound_save;
}

int get_chin_line_bound(CvRect* face_rect) 
{
	return face_rect->y + face_rect->height + (int)(face_rect->height * 0.05);
}

int get_upper_hair_color(int vertical_center_of_hair, int front_hair_lower_bound, int head_upper_bound, cv::Mat gray_scale_mat)
{
	int sum_of_hair_color = 0;
	double mean_of_hair_color = gray_scale_mat.at<uchar>(front_hair_lower_bound + 3,vertical_center_of_hair);
	sum_of_hair_color = mean_of_hair_color;
	int size_of_sample = 1;
	std::vector<int> myvector;

	while(front_hair_lower_bound > head_upper_bound)
	{
		myvector.push_back(gray_scale_mat.at<uchar>(front_hair_lower_bound,vertical_center_of_hair));
		if( gray_scale_mat.at<uchar>(front_hair_lower_bound,vertical_center_of_hair) > mean_of_hair_color + 30)
		{
			front_hair_lower_bound--;
		}
		else if(size_of_sample > 30)
		{
			break;
		}
		else if(gray_scale_mat.at<uchar>(front_hair_lower_bound,vertical_center_of_hair) < mean_of_hair_color)
		{
			sum_of_hair_color += gray_scale_mat.at<uchar>(front_hair_lower_bound,vertical_center_of_hair);
			size_of_sample++;
			mean_of_hair_color = sum_of_hair_color/(double)size_of_sample;
			front_hair_lower_bound--;
		}
		else
		{
			front_hair_lower_bound--;
		}
	}
	return mean_of_hair_color;
}

int get_left_hair_color(cv::Mat gray_scale_mat, int starting_left_pos, int right_iter_bound, int searching_line)
{
	int sum_of_hair_color = 0;
	double mean_of_hair_color = gray_scale_mat.at<uchar>(searching_line, starting_left_pos);
	sum_of_hair_color = mean_of_hair_color;
	int size_of_sample = 1;
	std::vector<int> myvector;

	while(starting_left_pos < right_iter_bound)
	{
		myvector.push_back(gray_scale_mat.at<uchar>(searching_line,starting_left_pos));
		if( gray_scale_mat.at<uchar>(searching_line,starting_left_pos) > mean_of_hair_color + 50)
		{
			starting_left_pos++;
		}
		else if(size_of_sample > 30)
		{
			break;
		}
		else if(gray_scale_mat.at<uchar>(searching_line,starting_left_pos) <= mean_of_hair_color + 50)
		{
			sum_of_hair_color += gray_scale_mat.at<uchar>(searching_line,starting_left_pos);
			size_of_sample++;
			mean_of_hair_color = sum_of_hair_color/(double)size_of_sample;
			starting_left_pos++;
		}
		else
		{
			starting_left_pos++;
		}
	}
	return mean_of_hair_color;
}

int get_left_hair_lower_bound(int front_hair_color, int chin_line_bound, int starting_lower_bound, int left_iter_bound, int right_iter_bound, cv::Mat gray_scale_mat) 
{
	std::vector<int> myvector;

	//옆머리가 귀 아래에 있는지 없는지 먼저 확인한다.
	int left_hair_left_bound = left_iter_bound;
	while(left_hair_left_bound < right_iter_bound)
	{
		if( gray_scale_mat.at<uchar>(starting_lower_bound,left_hair_left_bound) < front_hair_color + 30 )
		{
			break;
		}
		else
		{
			left_hair_left_bound++;
		}
	}
	if(left_hair_left_bound == right_iter_bound)
	{
		return 0;//옆머리가 없으므로 0을 리턴해서 옆머리가 아주 짧다고 인식하게 만든다
	}

	//옆머리 색 평균을 내야 함(윗머리보다 노이즈가 많은 거 같음)
	int left_hair_color = get_left_hair_color(gray_scale_mat, left_hair_left_bound, right_iter_bound, starting_lower_bound);

	int warning_count = 0;
	while(starting_lower_bound < chin_line_bound)
	{
		int hair_pixel_count = 0;
		int iter = right_iter_bound;//피부에서부터 검색 시작한다.
		while(iter > left_iter_bound)
		{
			if( gray_scale_mat.at<uchar>(starting_lower_bound,iter) < left_hair_color + 30 )
			{
				hair_pixel_count++;
			}

			if(hair_pixel_count> 2)
			{
				break;
			}
			else
			{
				iter--;
			}
		}
		if(hair_pixel_count <= 2)
		{
			warning_count++;
			if(warning_count > 5)
			{
				break;
			}
			else
			{
				starting_lower_bound++;
			}
		}
		else
		{
			starting_lower_bound++;
		}
	}

	return starting_lower_bound;
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
	else if(detected_front_hair == long_front_hair && detected_side_hair == tied_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontlong_sideshort.png");
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
	else if(detected_front_hair == no_front_hair && detected_side_hair == tied_hair)
	{
		cv::Mat hairtyle_image= cv::imread("frontshort_sideshort.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}

	//앞머리가 비대칭이고 옆머리가 긴 경우
	else if(detected_front_hair == left_front_long_right_front_short && detected_side_hair == long_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("leftfrontlong_rightfrontshort_sidelong.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}
	else if(detected_front_hair == left_front_short_right_front_long && detected_side_hair == long_left_hair)
	{
		cv::Mat hairtyle_image= cv::imread("leftfrontshort_rightfrontlong_sidelong.png");
		cv::namedWindow("recommended hairstyle");
		cv::imshow("recommended hairstyle",hairtyle_image);
	}

	//옆머리가 길지 않은 경우는 이모티콘이 없으므로, 앞머리가 그냥 길다고 판정해버린다
	else if(detected_front_hair == left_front_long_right_front_short && detected_side_hair != long_left_hair)
	{
		show_recommended_hairstyle(long_front_hair, detected_side_hair);
	}
	else if(detected_front_hair == left_front_short_right_front_long && detected_side_hair != long_left_hair)
	{
		show_recommended_hairstyle(long_front_hair, detected_side_hair);
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
		detected_side_hair = tied_hair;
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
	else
	{
		//앞머리 없다
		detected_front_hair =  no_front_hair;
	}
	return detected_front_hair;
}

int determine_front_hair_bound(int mid_front, int left_front, int right_front)
{
	int judged_value = 0;
	return judged_value;
}

int main(int argc, char* argv[])
{
	LARGE_INTEGER liCounter1, liCounter2, liFrequency;

	QueryPerformanceFrequency(&liFrequency);  // retrieves the frequency of the high-resolution performance counter    
	QueryPerformanceCounter(&liCounter1);         // Start


	
	const char *faceclassifer = "haarcascade_frontalface_alt.xml";
	CvHaarClassifierCascade* facecascade = 0;
	facecascade = (CvHaarClassifierCascade*) cvLoad(faceclassifer, 0, 0, 0 );

	const char *eyeclassifer = "haarcascade_eye.xml";
	CvHaarClassifierCascade* eyecascade = 0;
	eyecascade = (CvHaarClassifierCascade*) cvLoad(eyeclassifer, 0, 0, 0 );

	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);


	//이미지를 로드
	IplImage *pic = cvLoadImage("hello_world.jpg",CV_LOAD_IMAGE_COLOR);
	if(pic == NULL)
	{
		printf("사진 파일을 찾을 수 없습니다. Enter키를 누르시면 종료합니다.\n");
		while(1)
		{
			if (std::cin.get() == '\n')
			{
				return 0;
			}
		}
	}

	//이 아래는 얼굴 인식
	
	//얼굴 검출
	CvSeq *detected_face = cvHaarDetectObjects(pic, facecascade, storage, 1.4 , 1, 0);
	if(detected_face == NULL)
	{
		printf("사진 파일에서 얼굴을 찾을 수 없습니다. Enter키를 누르시면 종료합니다.\n");
		while(1)
		{
			if (std::cin.get() == '\n')
			{
				return 0;
			}
		}
	}
	//검출된 얼굴 Rectangle 받아오기(얼굴은 하나만 입력되고 인식됐다고 가정)
	CvRect *face_rect = 0;
	face_rect = (CvRect*) cvGetSeqElem(detected_face, 0);

	//얼굴 이미지에서 눈을 찾아야 할 것 같음
	cv::Mat face_mat= cv::imread("hello_world.jpg");

	cv::Mat pic_gray;
	cvtColor( face_mat, pic_gray, CV_BGR2GRAY );


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
	if(left_eye == NULL)
	{
		printf("사진에서 왼쪽 눈을 찾을 수 없습니다. Enter키를 누르시면 종료합니다.\n");
		while(1)
		{
			if (std::cin.get() == '\n')
			{
				return 0;
			}
		}
	}
	CvSeq* right_eye = cvHaarDetectObjects(right_face_iplimage, eyecascade, storage, 1.4 , 1, 0);
	if(right_eye == NULL)
	{
		printf("사진에서 오른쪽 눈을 찾을 수 없습니다. Enter키를 누르시면 종료합니다.\n");
		while(1)
		{
			if (std::cin.get() == '\n')
			{
				return 0;
			}
		}
	}

	//검출된 눈깔 Rectangle 받아오기(눈알은 하나만 입력되고 인식됐다고 가정)
	CvRect *left_eye_rect = 0;
	left_eye_rect = (CvRect*) cvGetSeqElem(left_eye, 0);

	CvRect *right_eye_rect = 0;
	right_eye_rect = (CvRect*) cvGetSeqElem(right_eye, 0);
	
	eye_upper_bound = (face_rect->y) + (left_eye_rect->y);
	
	//윗머리 끝 라인
	int hair_upper_bound = get_head_upper_bound(face_rect);
	//턱선
	int chin_line_bound = get_chin_line_bound(face_rect);

	//스펙트럼을 이용한 앞머리 경계 판정
	if(hair_upper_bound < 0)
	{
		hair_upper_bound = 0;
	}

	int front_hair_left_lower_bound = get_front_hair_lower_bound(pic_gray, hair_upper_bound, face_rect->x + left_eye_rect->x + left_eye_rect->width, eye_upper_bound - (int)(0.5*left_eye_rect->height) );
	int front_hair_right_lower_bound = get_front_hair_lower_bound(pic_gray, hair_upper_bound, face_rect->x + (int)(face_rect->width / 2.0) + right_eye_rect->x, eye_upper_bound - (int)(0.5*left_eye_rect->height) );
	int mid_front_hair_lower_bound = get_front_hair_lower_bound(pic_gray, hair_upper_bound, face_rect->x + (int)(face_rect->width / 2.0), eye_upper_bound - (int)(0.5*left_eye_rect->height) );
	

	int mid_left_pos = face_rect->x + left_eye_rect->x + left_eye_rect->width + face_rect->x + (int)(face_rect->width / 2.0);
	mid_left_pos/=2;
	int mid_right_pos = face_rect->x + (int)(face_rect->width / 2.0) + right_eye_rect->x + face_rect->x + (int)(face_rect->width / 2.0);
	mid_right_pos/=2;

	int mid_left_front_hair_lower_bound = get_front_hair_lower_bound(pic_gray, hair_upper_bound, mid_left_pos, eye_upper_bound - (int)(0.5*left_eye_rect->height) );
	int mid_right_front_hair_lower_bound = get_front_hair_lower_bound(pic_gray, hair_upper_bound, mid_right_pos, eye_upper_bound - (int)(0.5*left_eye_rect->height) );

	front_hair_style mid_front_hair_style = judge_front_hairstyle(mid_front_hair_lower_bound, hair_upper_bound);
	front_hair_style left_front_hair_style = judge_front_hairstyle(front_hair_left_lower_bound, hair_upper_bound);
	front_hair_style right_front_hair_style = judge_front_hairstyle(front_hair_right_lower_bound, hair_upper_bound);

	front_hair_style mid_left_front_hair_style = judge_front_hairstyle(mid_left_front_hair_lower_bound, hair_upper_bound);
	front_hair_style mid_right_front_hair_style = judge_front_hairstyle(mid_right_front_hair_lower_bound, hair_upper_bound);


	front_hair_style judged_front_hair_style;
	int hair_color = 0;
	if(mid_left_front_hair_style == long_front_hair && mid_right_front_hair_style == long_front_hair && mid_front_hair_style == no_front_hair)//앞머리 판정을 정확하게 하기 위한 조건문
	{
		hair_color = get_upper_hair_color(mid_right_pos, mid_right_front_hair_style, hair_upper_bound, pic_gray);
		judged_front_hair_style = long_front_hair;
	}

	//왼쪽 앞머리 짧고 오른쪽 앞머리 긴 경우
	else if(
		left_front_hair_style == no_front_hair && mid_left_front_hair_style == no_front_hair
		&& mid_front_hair_style == long_front_hair && mid_right_front_hair_style == long_front_hair && right_front_hair_style == long_front_hair
		)
	{
		hair_color = get_upper_hair_color(face_rect->x + (int)(face_rect->width / 2.0), mid_front_hair_lower_bound, hair_upper_bound, pic_gray);
		judged_front_hair_style = left_front_short_right_front_long;
	}
	//오른쪽 앞머리 짧고 왼쪽 앞머리 긴 경우
	else if(
		left_front_hair_style == long_front_hair && mid_left_front_hair_style == long_front_hair && mid_front_hair_style == long_front_hair
		&& mid_right_front_hair_style == no_front_hair && right_front_hair_style == no_front_hair
		)
	{
		hair_color = get_upper_hair_color(face_rect->x + (int)(face_rect->width / 2.0), mid_front_hair_lower_bound, hair_upper_bound, pic_gray);
		judged_front_hair_style = left_front_long_right_front_short;
	}
	else
	{
		hair_color = get_upper_hair_color(face_rect->x + (int)(face_rect->width / 2.0), mid_front_hair_lower_bound, hair_upper_bound, pic_gray);
		judged_front_hair_style = mid_front_hair_style;
	}

	//옆머리 경계 판정
	int left_iter_bound = face_rect->x - (face_rect->width)/2;
	if(left_iter_bound < 0)
	{
		left_iter_bound = 0;
	}
	int left_hair_lower_bound = get_left_hair_lower_bound(hair_color, chin_line_bound, (int) ( eye_upper_bound + (chin_line_bound - eye_upper_bound)*(3.0/5.0) - 1 ), left_iter_bound, face_rect->x + left_eye_rect->x + (left_eye_rect->width/2), pic_gray);


	side_hair_style detected_side_hair = judge_side_hairstyle(left_hair_lower_bound, chin_line_bound, hair_upper_bound);

	show_recommended_hairstyle(judged_front_hair_style, detected_side_hair);

	QueryPerformanceCounter(&liCounter2);//타이머 끝



	//소요 시간을 파일의 마지막 줄에 기록
	FILE *fp;
	fp = fopen( "elapsed_time.txt", "a");
	fprintf(fp, "%lf\n", (double)(liCounter2.QuadPart - liCounter1.QuadPart) / (double)liFrequency.QuadPart);
	fclose( fp);



	cvShowImage("original image (exit = esc)",pic);
	cvWaitKey(0);

	cvDestroyWindow("original image (exit = esc)");

	return 0;
}