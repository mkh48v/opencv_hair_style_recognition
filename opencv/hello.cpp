#include <opencv\cv.h>
#include <opencv\highgui.h>

int main() {
	IplImage *image = cvLoadImage("test.jpg");

	cvShowImage("Test",image);
	cvWaitKey(0);

	cvReleaseImage(&image);
}
