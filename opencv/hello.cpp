#include <iostream>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

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