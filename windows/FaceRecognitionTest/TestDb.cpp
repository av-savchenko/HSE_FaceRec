#include "stdafx.h"

#include "TestDb.h"
#include "FaceImage.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const int width = 160, height = 160;

void create_model_and_test_images(Mat& model_image, Mat& test_image){
	const int delta_test=3;
	const int color_delta_test=20;
	model_image = Mat::zeros(height, width, CV_8UC3);
	test_image = Mat::zeros(height, width, CV_8UC3);
	model_image.setTo(Scalar(255));
	test_image.setTo(Scalar(255));
	
    /*circle(	image, Point(501,  10), 5, Scalar(  0), thickness, lineType);
    circle(	image, Point(255,  10), 5, Scalar(100), thickness, lineType);
    circle(	image, Point(501, 255), 5, Scalar(100), thickness, lineType);
    circle(	image, Point( 10, 501), 5, Scalar(100), thickness, lineType);*/
	for(int i=0;i<50;++i){
		int left=((float)rand())/RAND_MAX*(width-1);
		int right=((float)rand())/RAND_MAX*(width-1);
		int top=((float)rand())/RAND_MAX*(height-1);
		int bottom=((float)rand())/RAND_MAX*(height-1);
		int color=rand()%255;
		rectangle(model_image,Point(left,top), Point(right,bottom),Scalar(color,color,color),-1);

		left+=delta_test-rand()%(2*delta_test);
		if(left<0)
			left=0;
		if(left>=width-1)
			left=width-1;

		right+=delta_test-rand()%(2*delta_test);
		if(right<0)
			right=0;
		if(right>=width-1)
			right=width-1;

		top+=delta_test-rand()%(2*delta_test);
		if(top<0)
			top=0;
		if(top>=height-1)
			top=height-1;

		bottom+=delta_test-rand()%(2*delta_test);
		if(bottom<0)
			bottom=0;
		if(bottom>=height-1)
			bottom=height-1;


		color+=color_delta_test-rand()%(2*color_delta_test);
		if(color<0)
			color=0;
		if(color>=255)
			color=255;
		rectangle(test_image,Point(left,top), Point(right,bottom),Scalar(color,color,color),-1);
	}
}

void randomize_image(Mat& image, int num_of_random_points){
	for(int i=0;i<num_of_random_points;++i){
		int point_width=2;
		int left=((float)rand())/RAND_MAX*(width-point_width);
		int top=((float)rand())/RAND_MAX*(height-point_width);
		int color=rand()%256;
		circle(	image, Point(left, top), point_width, Scalar(color,color,color), -1);
	}
}

void change_test_image(Mat& test_image){
    int histoGranularityW=width/POINTS_IN_W;
    int histoGranularityH=height/POINTS_IN_H;
	
	int beta=0;
	const float start_alpha=0.75;

    for(int i=0;i<POINTS_IN_H;++i)
        for(int j=0;j<POINTS_IN_W;++j){
			float alpha=start_alpha+2*(1-start_alpha)*rand()/RAND_MAX;
			for(int di=0;di<histoGranularityH;++di){
                    int x=i*histoGranularityH+di;
                    for(int dj=0;dj<histoGranularityW;++dj){
                        int y=j*histoGranularityW+dj;
						test_image.at<uchar>(y,x) = saturate_cast<uchar>( alpha*( test_image.at<uchar>(y,x) ) + beta );
					}
			}
		}
}
void fill_image(){
	srand(time(NULL));
    Mat model_image, test_image;

	create_model_and_test_images(model_image,test_image);

	randomize_image(model_image, 200);

	change_test_image(test_image);
	randomize_image(test_image, 200);

	//cvSaveImage("result.jpg", &(IplImage(model_image)));

	//namedWindow("Original Image", 1);
	//namedWindow("New Image", 1);
	
	/// Show stuff
	imshow("Original Image", model_image);
	imshow("New Image", test_image);

    waitKey(0);
}

#include <sstream>
void load_model_and_test_images(vector<FaceImage*>& dbImages, vector<FaceImage*>& testImages){
	const int CLASSES_NUMBER=5000;
	ostringstream str;
	for(int i=0;i<CLASSES_NUMBER;++i){
		str.str("");
		str << "class " << i;
		Mat model_image, test_image;

		create_model_and_test_images(model_image,test_image);
		int models_count = rand() % 3 + 1;
		for (int j = 0; j < models_count; ++j){
			randomize_image(model_image, 30);
			dbImages.push_back(new FaceImage(model_image, str.str()));
		}
		
		//change_test_image(test_image);
		randomize_image(test_image, 70);
		testImages.push_back(new FaceImage(test_image,str.str()));

		//change_test_image(test_image);
		/*randomize_image(test_image, 70);
		testImages.push_back(new FaceImage(test_image, str.str()));*/

	}
}