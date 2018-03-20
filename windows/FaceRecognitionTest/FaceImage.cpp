#include "stdafx.h"

#include "FaceImage.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;

//#define ORTHO_SERIES_HISTOS
#ifdef ORTHO_SERIES_HISTOS
#define ORTHO_SERIES_CORRECT
#endif

static void median(int* pixels, int width,int height, int colorCount){
    int *prev_pixels=new int[width*height*colorCount];
    int i,j,ind;
    for(j=0;j<width;++j){
        for(i=0;i<height;++i){
            ind=(i*width+j)*colorCount+0;
            prev_pixels[ind]=pixels[ind];
        }
    }
    const int MEDIAN_DIFF=1;
    std::vector<int> neighbors;
    for(int k=0;k<1;++k)
        for(i=1;i<height-1;++i){
            for(j=1;j<width-1;++j){
                ind=(i*width+j)*colorCount;
                neighbors.clear();
                for(int i1=(i-MEDIAN_DIFF);i1<=(i+MEDIAN_DIFF);++i1)
                    if(i1>=0 && i1<height)
                        for(int j1=(j-MEDIAN_DIFF);j1<=(j+MEDIAN_DIFF);++j1)
                            if(j1>=0 && j1<width)
                                neighbors.push_back(prev_pixels[i1*width+j1]);

                neighbors.push_back(pixels[ind+k]);
                std::sort(neighbors.begin(),neighbors.end());
                pixels[ind+k]=neighbors[neighbors.size()/2];  
			}
        }
   delete[] prev_pixels;
}
static void equalizeAllPixels(int* pixels, int width,int height, int colorCount){
	const int max=255;
	const int min=0;
	int i,j,k,l,ind;
	int histo[max+1];
	int sqrSize=height*width;
	memset(histo,0,sizeof(histo));
	for(i=0;i<height;++i)
		for(j=0;j<width;++j){
			ind=(i*width+j)*colorCount;
			++histo[pixels[ind]];
		}
	for(l=1;l<=max;++l){
		histo[l]+=histo[l-1];
	}
	for(i=0;i<height;++i)
		for(j=0;j<width;++j){
			ind=(i*width+j)*colorCount;
			pixels[ind]=min+(max-min)*histo[pixels[ind]]/sqrSize;
		}
}
static void varSmoothing(int* pixels, int width,int height, int colorCount){
    float *vars=new float[width*height*colorCount];
	int i,j,ind;
	const int VAR_DIFF=2;
	float minVar=100000,maxVar=0;
    for(int k=0;k<1;++k)
        for(i=1;i<height-1;++i){
            for(j=1;j<width-1;++j){
                ind=(i*width+j)*colorCount;
				float mean=pixels[ind+k],var=pixels[ind+k]*pixels[ind+k];
				int count=1;
				for(int i1=(i-VAR_DIFF);i1<=(i+VAR_DIFF);++i1)
                    if(i1>=0 && i1<height)
                        for(int j1=(j-VAR_DIFF);j1<=(j+VAR_DIFF);++j1)
                            if(j1>=0 && j1<width){
								mean+=pixels[i1*width+j1];
								var+=pixels[i1*width+j1]*pixels[i1*width+j1];
								++count;
							}
				mean/=count;
				var = fast_sqrt((var - mean * mean * count) / count);
				if(var<minVar)
					minVar=var;
				if(var>maxVar)
					maxVar=var;
				vars[ind+k]=var;
			}
        }
	for(int k=0;k<1;++k)
		for(j=0;j<width;++j){
			for(i=0;i<height;++i){
				ind=(i*width+j)*colorCount+0;
				pixels[ind+k]=maxVar==minVar?0:(int)(255*(vars[ind+k]-minVar)/(maxVar-minVar));
			}
		}
   delete[] vars;
}

static void randomizeImage(int* pixels, int width,int height, int colorCount, int rndImageRange){
    int i,j,ind;
    for(j=0;j<width;++j){
        for(i=0;i<height;++i){
            ind=(i*width+j)*colorCount+0;
			int rnd = rand();
			if (rndImageRange>0){
				pixels[ind] += rnd % (2 * rndImageRange) - rndImageRange;
				if (pixels[ind]<0)
					pixels[ind] = 0;
				if (pixels[ind]>255)
					pixels[ind] = 255;
			}
        }
    }
}

#if DISTANCE==LOCAL_DESCRIPTORS
	cv::DescriptorExtractor* FaceImage::extractor=new cv::SiftDescriptorExtractor();
	cv::FeatureDetector * FaceImage::detector=new cv::SiftFeatureDetector();
#endif


const float PI=4.*atan(1.0);

FaceImage::FaceImage(const char* imageFileName, const std::string& pName, int pW, int pH, int rndImageRange, ImageTransform img_transform) :
        personName(pName),
        fileName(imageFileName),
		pointsInW(pW), pointsInH(pH)
{
	//std::cout << imageFileName << '\n';
	const int scale = 1;
    Mat img = imread( imageFileName, CV_LOAD_IMAGE_COLOR );
    int width=img.cols/scale;
	int height = img.rows / scale;
#ifndef USE_DNN_FEATURES
	width = height = 96;
	if ((width%pointsInW) != 0) {
		if ((width%pointsInW)<pointsInW / 2)
			width = (width / pointsInW)*pointsInW;
		else
			width = ((width / pointsInW) + 1)*pointsInW;
	}
	if ((height%pointsInH) != 0) {
		if ((height%pointsInH)<pointsInH / 2)
			height = (height / pointsInH)*pointsInH;
		else
			height = ((height / pointsInH) + 1)*pointsInH;
	}
#elif defined(USE_RGB_DNN)
	width = height = 224;
	//width = 96; height = 112;
#else
	width = height = 128;
#endif

	resize(img, img, Size(width,height));

	init(img, width, height, rndImageRange, img_transform);
}
FaceImage::FaceImage(cv::Mat& img, const std::string& pName, int pW, int pH) :
        personName(pName),
		fileName(""),
		pointsInW(pW), pointsInH(pH)
{
    init(img,img.cols,img.rows);
}

FaceImage::FaceImage():
        personName(""),
		fileName(""),
		pointsInW(POINTS_IN_W), pointsInH(POINTS_IN_H)
{
	int i,j,k,ind;
	histos = new float[COLORS_COUNT*pointsInH*pointsInW*HISTO_SIZE];
	kernel_histos = kernel_histos_function = histosSum = 0;

	for(k=0;k<COLORS_COUNT;++k)
	for (i = 0; i<pointsInH; ++i)
		for (j = 0; j<pointsInW; ++j){
                for(ind=0;ind<HISTO_SIZE;++ind){
					histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 1.0 / HISTO_SIZE;//[k][i][j][ind]
                }
			}
}

FaceImage::FaceImage(int* p, int w, int h, const std::string& pName, int pW, int pH) :
	pixels(p),
	personName(pName),
	fileName(""),
	pointsInW(pW), pointsInH(pH),
	width(w), height(h)
{
	initHistos(p);
}		
#ifdef USE_DNN_FEATURES
FaceImage::FaceImage(std::string fn, std::string pn, std::vector<float>& features, bool normalize) :
fileName(fn),personName(pn),featureVector(FEATURES_COUNT)
{
	float sum = 1;
	if (normalize) {
		sum = 0;
#if DISTANCE!=EUC
		for (int i = 0; i < FEATURES_COUNT; ++i)
			sum += features[i];
#else
		for (int i = 0; i < FEATURES_COUNT; ++i)
			sum += features[i] * features[i];
		sum = sqrt(sum);
#endif
	}
	//std::cout << "sum=" << sum << std::endl;
	for (int i = 0; i < FEATURES_COUNT; ++i)
		featureVector[i] = features[i] / sum;
}


static std::string readString(std::istream& file)
{
	unsigned char len;
	file.read((char*)&len, 1);

	char* buffer = new char[len];
	file.read(buffer, len);

	std::string str(buffer, len);
	delete[] buffer;

	return str;
}

static void writeString(std::ostream& file, std::string str)
{
	unsigned char len = (char)str.length();
	file.write((char*)&len, 1);
	file.write(str.c_str(), len);
}
FaceImage* FaceImage::readFaceImage(std::ifstream& file)
{
	std::string fileName = readString(file);
	std::string personName = readString(file);
	//cout << fileName << ' ' << personName << '\n';

	std::vector<float> features(FEATURES_COUNT);
	file.read((char*)&features[0], sizeof(float)*FEATURES_COUNT);

	return new FaceImage(fileName, personName, features);
}

void FaceImage::writeFaceImage(std::ofstream& file)
{
	writeString(file, fileName);
	writeString(file, personName);

	file.write((char*)&featureVector[0], sizeof(float)*FEATURES_COUNT);
}
#endif

FaceImage::~FaceImage(){
	delete[] histos;
	delete[] histosSum;

	delete[] pixels;
}

FaceImage* FaceImage::nextInPyramid(double scale){
	//return new FaceImage(fileName.c_str(), personName, (int)(pointsInW / scale), (int)(pointsInH / scale));
	int newWidth = width;// (int)(width / scale);
	int newHeight = height;// (int)(height / scale);
	int *new_pixels = new int[width*height*COLORS_COUNT];
	memmove(new_pixels, pixels, width*height*COLORS_COUNT*sizeof(int));
	FaceImage* res = new FaceImage(new_pixels, width, height, personName, (int)(pointsInW / scale), (int)(pointsInH / scale));
	return res;
}

static int* get_gamma_vals(double gamma){
	static int lut[256];
	double inverse_gamma = 1.0 / gamma;
	for (int i = 0; i < 256; i++)
		lut[i] = (int)(pow((double)i / 255.0, gamma) * 255.0);
	return lut;

}
static int* lut = get_gamma_vals(0.2);

static Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static Mat tan_triggs_preprocessing(Mat X,
	float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
	int sigma1 = 2) {
	// Convert to floating point:
	X.convertTo(X, CV_32FC1);
	// Start preprocessing:
	Mat I;
	pow(X, gamma, I);
	// Calculate the DOG Image:
	if(false){
		Mat gaussian0, gaussian1;
		// Kernel Size:
		int kernel_sz0 = (3 * sigma0);
		int kernel_sz1 = (3 * sigma1);
		// Make them odd for OpenCV:
		kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
		kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
		GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_CONSTANT);
		GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_CONSTANT);
		subtract(gaussian0, gaussian1, I);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(abs(I), alpha, tmp);
			meanI = mean(tmp).val[0];

		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(min(abs(I), tau), alpha, tmp);
			meanI = mean(tmp).val[0];
		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	// Squash into the tanh:
	{
		for (int r = 0; r < I.rows; r++) {
			for (int c = 0; c < I.cols; c++) {
				I.at<float>(r, c) = tanh(I.at<float>(r, c) / tau);
			}
		}
		I = tau * I;
	}
	return norm_0_255(I);
}

/*
copied from https://github.com/MasteringOpenCV/code/blob/master/Chapter8_FaceRecognition/preprocessFace.cpp
Author Shervin Emami
*/
static void equalizeLeftAndRightHalves(Mat &faceImg)
{
	// It is common that there is stronger light from one half of the face than the other. In that case,
	// if you simply did histogram equalization on the whole face then it would make one half dark and
	// one half bright. So we will do histogram equalization separately on each face half, so they will
	// both look similar on average. But this would cause a sharp edge in the middle of the face, because
	// the left half and right half would be suddenly different. So we also histogram equalize the whole
	// image, and in the middle part we blend the 3 images together for a smooth brightness transition.

	int w = faceImg.cols;
	int h = faceImg.rows;

	// 1) First, equalize the whole face.
	Mat wholeFace;
	equalizeHist(faceImg, wholeFace);

	// 2) Equalize the left half and the right half of the face separately.
	int midX = w / 2;
	Mat leftSide = faceImg(Rect(0, 0, midX, h));
	Mat rightSide = faceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	// 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			int v;
			if (x < w / 4) {          // Left 25%: just use the left face.
				v = leftSide.at<uchar>(y, x);
			}
			else if (x < w * 2 / 4) {   // Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the whole face as it moves further right along the face.
				float f = (x - w * 1 / 4) / (float)(w*0.25f);
				v = cvRound((1.0f - f) * lv + (f)* wv);
			}
			else if (x < w * 3 / 4) {   // Mid-right 25%: blend the right face & whole face.
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the right-side face as it moves further right along the face.
				float f = (x - w * 2 / 4) / (float)(w*0.25f);
				v = cvRound((1.0f - f) * wv + (f)* rv);
			}
			else {                  // Right 25%: just use the right face.
				v = rightSide.at<uchar>(y, x - midX);
			}
			faceImg.at<uchar>(y, x) = v;
		}// end x loop
	}//end y loop
}
static Mat normalizeMat(const Mat& srcMat) {
	Mat result;
	if (srcMat.channels() >= 3)
	{
		Mat ycrcb;

		cvtColor(srcMat, ycrcb, CV_BGR2YCrCb);

		std::vector<Mat> channels;
		split(ycrcb, channels);

		equalizeLeftAndRightHalves(channels[0]);

		merge(channels, ycrcb);

		cvtColor(ycrcb, result, CV_YCrCb2BGR);
	}
	else {
		result = srcMat.clone();
		equalizeLeftAndRightHalves(result);
	}
	return result;
}
void FaceImage::init(cv::Mat& img, int w, int h, int rndImageRange, ImageTransform img_transform){
	//std::cout << fileName << '\n';
	width = w;
	height = h;
#if DISTANCE==LOCAL_DESCRIPTORS
	detector->detect(img,keypoints);
	extractor->compute(img,keypoints,descriptors);
#else
    int i,j,k;
	Mat dst;
#if 1
	//cv::GaussianBlur(img, dst, Size(1, 1),1);
	/*
	for (i = 0; i < width; ++i){
		for (j = 0; j < height; ++j){
			Vec3b& val = img.at<Vec3b>(Point(i, j));
			int greyColor = (int)(0.11*val[0] + .56*val[1] + .33*val[2]);
			int rnd = rand();
			if (rndImageRange>0){
				greyColor += rnd % (2 * rndImageRange) - rndImageRange;
				if (greyColor<0)
					greyColor = 0;
				if (greyColor>255)
					greyColor = 255;
			}
			val[0] = val[1] = val[2] = 
				greyColor;
				//lut[greyColor];
		}
	}
	*/
#if 1 || !defined(USE_DNN_FEATURES)
	cv::medianBlur(img, dst, 3);
#else
	dst = img;
#endif

	cv::Mat colorMat = dst.clone();

	cvtColor(dst, dst, CV_RGB2GRAY);
	//dst = tan_triggs_preprocessing(dst);
	
#if defined(USE_DNN_FEATURES) && defined(USE_RGB_DNN)
	cv::Mat srcMat=colorMat;
#else
	cv::Mat srcMat = dst;
#endif
	cv::Mat transformedMat;
	std::string dst_path = "D:\\img";
	switch (img_transform) {
	case ImageTransform::NONE:
		transformedMat = srcMat;
		break;
	case ImageTransform::FLIP:
		transformedMat = cv::Mat(srcMat.rows, srcMat.cols, srcMat.type());
		cv::flip(srcMat, transformedMat, 1);
		dst_path += "_flipped";
		break;
	case ImageTransform::NORMALIZE:
		transformedMat=normalizeMat(srcMat);
		dst_path += "_normalized"; 
		break;
	}
	//imwrite( dst_path+".jpg", transformedMat);
	/*char c;
	std::cin>>c;*/
#endif
	pixelMat = dst;

#ifndef USE_DNN_FEATURES

	pixels = new int[width*height*COLORS_COUNT];
	for (i = 0; i < width; ++i) {
		for (j = 0; j < height; ++j) {
			//Vec3b val = dst.at<Vec3b>(Point(i, j));
			uchar val = transformedMat.at<uchar>(Point(i, j));
			for (k = 0; k < COLORS_COUNT; ++k)
				pixels[(j*width + i)*COLORS_COUNT + k] = val;
		}
	}

	//equalizeAllPixels(pixels, width, height, COLORS_COUNT);
	float mean = 0, std_var = 0;
	for (i = 0; i < width; ++i) {
		for (j = 0; j < height; ++j) {
			//dst.at<uchar>(Point(i, j)) = pixels[(j*width + i)*COLORS_COUNT + 0];
			mean += pixels[(j*width + i)*COLORS_COUNT + 0];
			std_var += pixels[(j*width + i)*COLORS_COUNT + 0] * pixels[(j*width + i)*COLORS_COUNT + 0];
		}
	}
	mean /= width*height;
	std_var = std_var / (width*height) - mean*mean;
	std_var = (std_var>0) ? sqrt(std_var) : 0;
	for (i = 0; i < width; ++i) {
		for (j = 0; j < height; ++j) {
			for (k = 0; k < COLORS_COUNT; ++k) {
				float val = 128;
				if (std_var>0) {
					val = 128 + ((pixels[(j*width + i)*COLORS_COUNT + k] - mean) * 127 / (6 * std_var));
					if (val < 0)
						val = 0;
					else if (val >= 255)
						val = 255;
				}
				pixels[(j*width + i)*COLORS_COUNT + k] = (int)val;
			}
		}
	}
	initHistos(pixels);
	delete[] pixels; pixels = 0;
#else
	featureVector.resize(FEATURES_COUNT);
	DnnFeatureExtractor::GetInstance()->extractFeatures(transformedMat, &featureVector[0]);
	/*float sum = 0;
	for (int i = 0; i < FEATURES_COUNT; ++i)
		sum += featureVector[i] * featureVector[i];
	sum = sqrt(sum);
	for (int i = 0; i < FEATURES_COUNT; ++i)
		featureVector[i] /= sum;
		*/
#endif

#endif
}


#define eps 0.0001

// unit vectors used to compute gradient orientation
float uu[9] = { 1.0000,
0.9397,
0.7660,
0.500,
0.1736,
-0.1736,
-0.5000,
-0.7660,
-0.9397 };
float vv[9] = { 0.0000,
0.3420,
0.6428,
0.8660,
0.9848,
0.9848,
0.8660,
0.6428,
0.3420 };

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// takes a double color image and a bin size 
// returns HOG features
float* FaceImage::extractHOG(int& feat_size) {
	int *im = pixels;
	// memory for caching orientation histograms & their norms
	int blocks[2] = { pointsInH, pointsInW };
	int histoGranularityH = height / pointsInH;
	int histoGranularityW = width / pointsInW;
	/*blocks[0] = (int)round((float)width / (float)sbin);
	blocks[1] = (int)round((float)height / (float)sbin);*/
	float *hist = new float[blocks[0] * blocks[1] * 18];
	float *norm = new float[blocks[0] * blocks[1]];
	for (int i = 0; i < blocks[0] * blocks[1] * 18; ++i)
		hist[i] = 0;
	for (int i = 0; i < blocks[0] * blocks[1]; ++i)
		norm[i] = 0;

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0] - 2, 0);
	out[1] = max(blocks[1] - 2, 0);
	out[2] = 18;// 27 + 4 + 1;
	feat_size = out[0] * out[1] * out[2];
	float *feat = new float[feat_size];
	for (int i = 0; i < feat_size; ++i)
		feat[i] = 0;

	int visible[2];
	visible[0] = height;// blocks[0] * sbin;
	visible[1] = width;// blocks[1] * sbin;

	for (int x = 1; x < visible[1] - 1; x++) {
		for (int y = 1; y < visible[0] - 1; y++) {
			int *s = im + min(x, height - 2)*width + min(y, width - 2);
			int dy = *(s + 1) - *(s - 1);
			int dx = *(s + width) - *(s - width);
			int v = dx*dx + dy*dy;

			// snap to one of 18 orientations
			float best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++) {
				float dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o + 9;
				}
			}

			// add to 4 histograms around pixel using linear interpolation
			float xp = ((float)x + 0.5) / histoGranularityW - 0.5;
			float yp = ((float)y + 0.5) / histoGranularityH - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			float vx0 = xp - ixp;
			float vy0 = yp - iyp;
			float vx1 = 1.0 - vx0;
			float vy1 = 1.0 - vy0;
			v = sqrt(v);

			if (ixp >= 0 && iyp >= 0) {
				*(hist + ixp*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx1*vy1*v;
			}

			if (ixp + 1 < blocks[1] && iyp >= 0) {
				*(hist + (ixp + 1)*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx0*vy1*v;
			}

			if (ixp >= 0 && iyp + 1 < blocks[0]) {
				*(hist + ixp*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx1*vy0*v;
			}

			if (ixp + 1 < blocks[1] && iyp + 1 < blocks[0]) {
				*(hist + (ixp + 1)*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx0*vy0*v;
			}
		}
	}

	// compute energy in each block by summing over orientations
	for (int o = 0; o < 9; o++) {
		float *src1 = hist + o*blocks[0] * blocks[1];
		float *src2 = hist + (o + 9)*blocks[0] * blocks[1];
		float *dst = norm;
		float *end = norm + blocks[1] * blocks[0];
		while (dst < end) {
			*(dst++) += (*src1 + *src2);// *(*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// compute features
	for (int x = 0; x < out[1]; x++) {
		for (int y = 0; y < out[0]; y++) {
			float *dst = feat + x*out[0] + y;
			float *src, *p, n1, n2, n3, n4;

			p = norm + (x + 1)*blocks[0] + y + 1;
			n1 = 1.0 / /*sqrt*/(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + (x + 1)*blocks[0] + y;
			n2 = 1.0 / /*sqrt*/(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y + 1;
			n3 = 1.0 / /*sqrt*/(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y;
			n4 = 1.0 / /*sqrt*/(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);

			float t1 = 0;
			float t2 = 0;
			float t3 = 0;
			float t4 = 0;

			// contrast-sensitive features
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 18; o++) {
				float h1 = min(*src * n1, 0.2);
				float h2 = min(*src * n2, 0.2);
				float h3 = min(*src * n3, 0.2);
				float h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0] * out[1];
				src += blocks[0] * blocks[1];
			}
		}
	}

	delete[] hist;
	delete[] norm;

	float sum = eps;
	for (int i = 0; i < feat_size; ++i)
		sum += feat[i];
	for (int i = 0; i < feat_size; ++i){
		feat[i] /= sum;
	}
	return feat;
}

static int LBPH_size = 0;
static int* get_uniform_map(){
	static int uniform_map[256];
	int bits[8];
	for (int i = 0; i < 256; ++i){
		int prev_bit, cur_bit, count = 0;
		prev_bit = (i&(1 << 7)) ? 1 : 0;
		for (int j = 1; j<8; ++j){
			cur_bit = (i&(1 << (7 - j))) ? 1 : 0;
			if (prev_bit != cur_bit)
				++count;
			prev_bit = cur_bit;
		}
		if (count <= 2)
			uniform_map[i] = LBPH_size++;
		else
			uniform_map[i] = -1;
	}
	for (int i = 0; i < 256; ++i){
		if (uniform_map[i] == -1)
			uniform_map[i] = LBPH_size;
	}
	++LBPH_size;
	return uniform_map;
}
static int* uniform_map = get_uniform_map();
template<typename T> float* FaceImage::extractLBP(int& feat_size, T* image){
	int k = 0;

	int w1 = width , h1 = height ;
	int* lbp_image = new int[width*height*COLORS_COUNT];
	for (int j = 0; j < height; ++j){
		for (int i = 0; i < width; ++i){
			lbp_image[(j*width + i)*COLORS_COUNT + k] = 0;
		}
	}
	for (int i = 2; i < width - 2; ++i){
		for (int j = 2; j < height - 2; ++j){
			int res = 0;
			if (image[(j*w1 + i - 2)*COLORS_COUNT + k] >= image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 7;
			if (image[((j + 2)*w1 + i)*COLORS_COUNT + k] >= image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 5;
			if (image[(j*w1 + i + 2)*COLORS_COUNT + k] >= image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 3;
			if (image[((j - 2)*w1 + i)*COLORS_COUNT + k] >= image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 1;

			if ((image[((j + 1)*w1 + i - 1)*COLORS_COUNT + k] + image[((j + 2)*w1 + i - 2)*COLORS_COUNT + k])
				>= 2 * image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 6;
			if ((image[((j + 1)*w1 + i + 1)*COLORS_COUNT + k] + image[((j + 2)*w1 + i + 2)*COLORS_COUNT + k])
				>= 2 * image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 4;
			if ((image[((j - 1)*w1 + i + 1)*COLORS_COUNT + k] + image[((j - 2)*w1 + i + 2)*COLORS_COUNT + k])
				>= 2 * image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 2;
			if ((image[((j - 1)*w1 + i - 1)*COLORS_COUNT + k] + image[((j - 2)*w1 + i - 2)*COLORS_COUNT + k])
				>= 2 * image[(j*w1 + i)*COLORS_COUNT + k])
				res += 1 << 0;

			lbp_image[((j - 0)*width + i - 0)*COLORS_COUNT + k] = uniform_map[res];
		}
	}
	//WARNING!!! POINTS_IN_W, POINTS_IN_H are not used
	int pointsInW = 8;
	int pointsInH = 8;
	feat_size = pointsInW*pointsInH*LBPH_size;
	float* lbp_histo = new float[feat_size];
	memset(lbp_histo, 0, feat_size*sizeof(float));
	int histoGranularityW = width / pointsInW;
	int histoGranularityH = height / pointsInH;

	for (int k = 0; k < COLORS_COUNT; ++k){
		for (int i = 0; i < pointsInH; ++i){
			for (int j = 0; j < pointsInW; ++j){
				int superInd = LBPH_size*(i*pointsInW + j);
				for (int di = 0; di < histoGranularityH; ++di){
					int x = i*histoGranularityH + di;
					if (x >= height){
						break;
					}
					for (int dj = 0; dj < histoGranularityW; ++dj){
						int y = j*histoGranularityW + dj;
						if (y >= width){
							break;
						}
						int ind = lbp_image[x*width + y];
						if (ind != -1)
							++lbp_histo[superInd + ind];
					}
				}

				float sum = 0;
				for (int ind = 0; ind < LBPH_size; ++ind)
					sum += lbp_histo[superInd + ind];
				if (sum>0){
					float norm = sum;
					sum = 0;
					for (int ind = 0; ind < LBPH_size; ++ind){
						lbp_histo[superInd + ind] /= norm;
						if (lbp_histo[superInd + ind] > 0.2)
							lbp_histo[superInd + ind] = 0.2;
						sum += lbp_histo[superInd + ind];
					}
					norm = sum;
					for (int ind = 0; ind < LBPH_size; ++ind){
						lbp_histo[superInd + ind] /= norm;
					}
				}
			}
		}
	}

	delete[] lbp_image;
	return lbp_histo;
}

static Mat gabor_kernel[8];
static Ptr<Mat>* createFilterBank(){
	static Ptr<Mat> gabor_filter[8];
	float lambda = 4;
	float sigma = .5622*lambda;
	float thetas[] = { 0, 45, 90, 135 };
	float gamma = 1;
	for (int i = 0; i < 4; ++i){
		float theta = thetas[i] * CV_PI / 180;
		//Mat kernel;
		gabor_kernel[2*i]=getGaborKernel(cv::Size(13, 13), sigma, theta, lambda, gamma, 0, CV_32F);
		/*gabor_filter[2 * i] = createLinearFilter(CV_32F, CV_32F, gabor_kernel[2 * i], cv::Point(-1, -1),
			0, BORDER_REFLECT, BORDER_REFLECT, Scalar(0));*/
		gabor_kernel[2 * i+1] = getGaborKernel(cv::Size(13, 13), sigma, theta, lambda, gamma, -CV_PI / 2, CV_32F);
		/*gabor_filter[2 * i + 1] = createLinearFilter(CV_32F, CV_32F, gabor_kernel[2 * i + 1], cv::Point(-1, -1),
			0, BORDER_REFLECT, BORDER_REFLECT, Scalar(0));*/
	}
	return gabor_filter;

}
static Ptr<Mat>* gabor_filter = createFilterBank();

float* FaceImage::extractGabor(int& feat_size){

	Mat img;
	pixelMat.convertTo(img, CV_32F);
#if 1
	int w = 48, h=48;
#else
	int w = width, h = height;
#endif
	resize(img, img, Size(w, h));
	feat_size = w*h * 4;
	float* gabor_features = new float[feat_size];
	std::vector<float> res;
	Mat dest0(img.rows, img.cols, CV_32F), dest1(img.rows, img.cols, CV_32F);
	for (int i = 0; i < 4; ++i){
		filter2D(img, dest0, CV_32F, gabor_kernel[2*i]);
		//gabor_filter[2 * i]->apply(img, dest0);
		cv::pow(dest0, 2, dest0);
		//filter2D(img, dests[1], CV_32F, gabor_kernel[2*i+1]);
		//gabor_filter[2 * i+1]->apply(img, dest1);
		filter2D(img, dest1, CV_32F, gabor_kernel[2 * i + 1]);
		cv::pow(dest1, 2, dest1);
		cv::add(dest0, dest1, dest0);
		cv::pow(dest0, 0.5, dest0);
		/*double maxVal;
		cv::minMaxLoc(dest0, 0, &maxVal);
		dest0 /= (float)maxVal;*/

		float sum = 0.;
		for (int r = 0; r < h; ++r){
			for (int c = 0; c < w; ++c){
				gabor_features[i*w*h + r*w + c] = dest0.at<float>(r, c);
				sum += gabor_features[i*w*h + r*w + c];
			}
		}
		if (sum>0){
			//sum = sqrt(sum);
			for (int j = 0; j < w*h; ++j)
				gabor_features[i*w*h + j] /= sum;
		}
	}
	return gabor_features;
}
void FaceImage::initHistos(int* pixels)
{
	int i, j, k, di, dj, x, y, superInd, ind;

	const float min_weight = 0.1f;
	weights.resize(pointsInH*pointsInW);
	for (int row = 0; row<pointsInH; ++row){
		for (int j = 0; j < pointsInW; ++j){
			if ((row >= pointsInH / 2 && row <= 3 * pointsInH / 4)){
				if (j<pointsInW / 2)
					weights[row*pointsInW+j] = min_weight + (1 - min_weight) * 2 * j / (pointsInW - 1);
				else
					weights[row*pointsInW+j] = min_weight + (1 - min_weight) * 2 * (pointsInW - j - 1) / (pointsInW - 1);
			}
			else
				weights[row*pointsInW+j] = 1.f;
		}
	}

	//equalizeAllPixels(pixels,width,height,COLORS_COUNT);
	//median(pixels,width,height,COLORS_COUNT);
	//varSmoothing(pixels,width,height,COLORS_COUNT);
	int histo_length = COLORS_COUNT*pointsInH*pointsInW*HISTO_SIZE;
	histos = new float[3 * histo_length];
	kernel_histos = &histos[histo_length];
	kernel_histos_function = &histos[2*histo_length];
	histosSum = new float[COLORS_COUNT*pointsInH*pointsInW];

	int histoGranularityW = width / pointsInW;
	int histoGranularityH = height / pointsInH;
	blockSize = histoGranularityH*histoGranularityW;
	//const float PI=4*atan(1.);
	std::vector<float> mags;
	mags.reserve(blockSize);
	const int rate = 1;
#ifdef ORTHO_SERIES_HISTOS
	int PROJECT_SIZE = (int)pow(histoGranularityH*histoGranularityW, 1.0 / 3);
	if (PROJECT_SIZE<10)
		PROJECT_SIZE = 10;
	vector<double> a(PROJECT_SIZE), b(PROJECT_SIZE);
#endif

	for (k = 0; k<COLORS_COUNT; ++k)
	for (i = 0; i<pointsInH; ++i)
	for (j = 0; j<pointsInW; ++j){
		superInd = i*pointsInW + j;
#ifdef USE_GRADIENTS
#ifdef ORTHO_SERIES_HISTOS
		for (int projInd = 0; projInd<PROJECT_SIZE; ++projInd)
			a[projInd] = b[projInd] = 0;
#endif
		mags.clear();
		for (di = 0; di<histoGranularityH; ++di){
			x = i*histoGranularityH + di;
			if (x >= height - 1){
				break;
			}
			for (dj = 0; dj<histoGranularityW; ++dj){
				if ((di + dj) % rate != 0)
					continue;
				y = j*histoGranularityW + dj;
				if (y >= width - 1){
					break;
				}
				ind = x*width + y;
				int dfdx = (pixels[(x + 1)*width + y] - pixels[x*width + y + 1]);
				int dfdy = (pixels[x*width + y] - pixels[(x + 1)*width + y + 1]);
				//float magSqr=dfdx*dfdx+dfdy*dfdy;
				//float mag=labs(dfdx)>labs(dfdy)?labs(dfdx):labs(dfdy);
				float mag = labs(dfdx) + labs(dfdy);
				mags.push_back(mag);
			}
		}
		std::sort(mags.begin(), mags.end());
		float maxMag = mags[(mags.size() - 1) * 9 / 10] + 0.1;

		for (ind = 0; ind<HISTO_SIZE; ++ind){
			histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 0;
		}
		float sum = 0;
		for (di = 0; di<histoGranularityH; ++di){
			x = i*histoGranularityH + di;
			if (x >= height - 1){
				break;
			}
			for (dj = 0; dj<histoGranularityW; ++dj){
				if ((di + dj) % rate != 0)
					continue;
				y = j*histoGranularityW + dj;
				if (y >= width - 1){
					break;
				}
				ind = x*width + y;
				int dfdx = (pixels[(x + 1)*width + y] - pixels[x*width + y + 1]);
				int dfdy = (pixels[x*width + y] - pixels[(x + 1)*width + y + 1]);
				//float magSqr=dfdx*dfdx+dfdy*dfdy;
				//float mag=labs(dfdx)>labs(dfdy)?labs(dfdx):labs(dfdy);
				float mag = labs(dfdx) + labs(dfdy);
				int angleInd = 0;

#if 1
				float angle = atan2((float)dfdy, (float)dfdx) / (2 * PI);//+1./(2*HISTO_SIZE);
				if (angle<0)
					angle += 1;
				else if (angle >= 1)
					angle -= 1;
				angleInd = (int)(HISTO_SIZE*angle);
#else
				int bit0 = (abs(dfdy)<abs(dfdx)) ? 1 : 0;
				int bit1 = (dfdx<0) ? 2 : 0;
				int bit2 = (dfdy<0) ? 4 : 0;
				angleInd = bit0 + bit1 + bit2;
#endif
				//mag+=0.1;
				float curIncrement = (mag >= maxMag) ? 1 : (mag / maxMag);
				histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + angleInd] += curIncrement;

#ifdef ORTHO_SERIES_CORRECT
				float angle = atan2((float)dfdy, (float)dfdx);
				//curIncrement=1;
				for (int projInd = 0; projInd<PROJECT_SIZE; ++projInd){
					a[projInd] += curIncrement*cos(angle*(projInd + 1))*(PROJECT_SIZE - projInd) / (mags.size()*(PROJECT_SIZE + 1));
					b[projInd] += curIncrement*sin(angle*(projInd + 1))*(PROJECT_SIZE - projInd) / (mags.size()*(PROJECT_SIZE + 1));
				}
#endif
			}
		}
#else
		for (di = 0; di<histoGranularityH; ++di){
			x = i*histoGranularityH + di;
			//x=i*histoGranularityH/2+di;
			if (x >= height){
				break;
			}
			for (dj = 0; dj<histoGranularityW; ++dj){
				y = j*histoGranularityW + dj;
				//y=j*histoGranularityW/2+dj;
				if (y >= width){
					break;
				}
				ind = x*width + y;
				int ind = HISTO_SIZE*pixels[x*width + y] / 256;
				++histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
			}
		}
#endif
		sum = 0;
		for (ind = 0; ind<HISTO_SIZE; ++ind){
			histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] += 0.1;
			sum += histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
		}
		histosSum[(k*pointsInH + i)*pointsInW + j] = sum;
		if (sum>0){
			float sum1 = 0;
			for (ind = 0; ind<HISTO_SIZE; ++ind){
				histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] /= sum;
			}
		}

#if defined(ORTHO_SERIES_HISTOS) && !defined(ORTHO_SERIES_CORRECT)
		for (int projInd = 0; projInd<PROJECT_SIZE; ++projInd){
			for (int ind1 = 0; ind1<HISTO_SIZE; ++ind1){
				double angle = 2 * PI*(ind1 + 1) / HISTO_SIZE;
				if (angle>PI)
					angle -= 2 * PI;
				a[projInd] += histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind1] * cos(angle*(projInd + 1));
				b[projInd] += histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind1] * sin(angle*(projInd + 1));
				//std::cout<<projInd<<' '<<angle<<' '<<a[projInd]<<' '<<b[projInd]<<'\n';
			}
			a[projInd] = a[projInd] * (PROJECT_SIZE - projInd) / ((PROJECT_SIZE + 1));
			b[projInd] = b[projInd] * (PROJECT_SIZE - projInd) / ((PROJECT_SIZE + 1));
			//std::cout<<"ab "<<a[projInd]<<' '<<b[projInd]<<'\n';
		}
#endif
		//exit(0);
		sum = 0;
		for (ind = 0; ind<HISTO_SIZE; ++ind){
			kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 0;
#ifdef ORTHO_SERIES_HISTOS
			double angle = 2 * PI*(ind + 1) / HISTO_SIZE;
			if (angle>PI)
				angle -= 2 * PI;
			for (int projInd = 0; projInd<PROJECT_SIZE; ++projInd){
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] += a[projInd] * cos(angle*(projInd + 1));
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] += b[projInd] * sin(angle*(projInd + 1));
			}
			//kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]/=PROJECT_SIZE+1;
			kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] += 0.5;
			if (kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]<0){
				cout << kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] << " error\n";
				for (int projInd = 0; projInd<PROJECT_SIZE; ++projInd){
					cout << a[projInd] << ' ' << b[projInd] << endl;
				}
				exit(0);
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 0.001;
			}
			//std::cout<<histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]<<' '<<kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]<<'\n';
#ifdef ORTHO_SERIES_CORRECT
			histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
#endif
#else
			for (int ind1 = 0; ind1<HISTO_SIZE; ++ind1)
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] +=
				kernel.kernel[ind][ind1] * histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind1];
#endif
			sum += kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
		}
#ifdef ORTHO_SERIES_CORRECT
		sum = 0;
		for (ind = 0; ind<HISTO_SIZE; ++ind){
			sum += histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
		}
		for (ind = 0; ind<HISTO_SIZE; ++ind){
			histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] /= sum;
		}
		sum = 0;
		for (ind = 0; ind<HISTO_SIZE; ++ind){
			kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 0;
			for (int ind1 = 0; ind1<HISTO_SIZE; ++ind1)
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] +=
				kernel.kernel[ind][ind1] * histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind1];
			sum += kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
		}

#endif
		if (sum>0){
			for (ind = 0; ind<HISTO_SIZE; ++ind){
				kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] /= sum;

				kernel_histos_function[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 0;
				if (kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]>0)
					kernel_histos_function[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] =
#if DISTANCE==HOMOGENEITY || DISTANCE==PNN
					log(kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]);
#elif DISTANCE == SIMPLIFIED_HOMOGENEITY
					1 / kernel_histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind];
#else
					0;
#endif
#if DISTANCE == KL
				kernel_histos_function[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind] = 
					log(histos[((k*pointsInH + i)*pointsInW + j)*HISTO_SIZE + ind]);
#endif
			}
		}
	}


#ifdef USE_EXTRA_FEATURES
	int lbp_feat_size = 0;
	float* lbp_features = 0;// extractLBP(lbp_feat_size, pixels);
	//std::cout << lbp_feat_size << '\n';
	//lbp_feat_size = 0;

#if 1
	int hog_feat_size = 0;
	float* hog_features =  extractHOG(hog_feat_size);
	//std::cout << hog_feat_size << '\n';
#else
	int hog_feat_size = COLORS_COUNT*POINTS_IN_W*POINTS_IN_H*HISTO_SIZE;
	float* hog_features = histos;
#endif
	//hog_feat_size = 0;
	int gabor_feat_size = 0;
	float* gabor_features = 0;// extractGabor(gabor_feat_size);

	featureVector.resize(lbp_feat_size + hog_feat_size+gabor_feat_size);
	for (int i = 0; i < lbp_feat_size; ++i)
		featureVector[i] = lbp_features[i];
	for (int i = 0; i < hog_feat_size; ++i)
		featureVector[i + lbp_feat_size] = hog_features[i];
	for (int i = 0; i < gabor_feat_size; ++i)
		featureVector[i + lbp_feat_size + hog_feat_size] = gabor_features[i];

	delete[] gabor_features;
	delete[] hog_features;
	delete[] lbp_features;
#endif

}


FaceImage::Kernel::Kernel()
{
	float expCoeffs[HISTO_SIZE];
#ifdef USE_GRADIENTS
	float alpha=-3;
#else
	float alpha=-0.5;
#endif
	for(int i=0;i<HISTO_SIZE;++i){
		expCoeffs[i]=exp(alpha*i);
	}
	for(int i=0;i<HISTO_SIZE;++i){
		float sum=0;
		for(int j=0;j<HISTO_SIZE;++j){
			int diff = abs(i - j);
#ifdef USE_GRADIENTS
			if (diff>HISTO_SIZE / 2)
				diff = (HISTO_SIZE - diff);
#endif
			kernel[i][j]=expCoeffs[diff];
			sum+=kernel[i][j];
		}
		for(int j=0;j<HISTO_SIZE;++j){
			kernel[i][j]/=sum;
		}
	}
}

//#define PARALLEL_DISTANCE
const int NUM_OF_ROWS=1;
const int thread_count=8;

inline float FaceImage::getRowDistance(const FaceImage* rhs,int k, int row) const
{
	float res=0;
	int jMin, jMax;
	int maxI=row+NUM_OF_ROWS;
	if(maxI>pointsInH)
		maxI=pointsInH;
	int delta = /*(pointsInW <= 5) ? 0 : */DELTA;
	for(int i=row;i<maxI;++i){
		int iMin = i >= delta ? i - delta : 0;
		int iMax = i + delta;
		if(iMax>=pointsInH)
			iMax=pointsInH-1;
		for (int j = 0; j<pointsInW; ++j){
			jMin = j >= delta ? j - delta : 0;
			jMax = j + delta;
			if (jMax >= pointsInW)
				jMax = pointsInW - 1;
			float minSum=1000000;
			int i1=i,j1=j;
			//for(i1=iMin;i1<=iMax;++i1)
			//    for(j1=jMin;j1<=jMax;++j1)
			for(int i2=iMin;i2<=iMax;++i2){
				for(int j2=jMin;j2<=jMax;++j2){
					float curSum=0;
#if DISTANCE==KOLMOGOROFF
					curSum=100000;
#endif
					float cdf1=0,cdf2=0;
					for(int ind=0;ind<HISTO_SIZE;++ind){
						float d1 = histos[((k*pointsInH + i1)*pointsInW + j1)*HISTO_SIZE + ind];
						float d2 = rhs->histos[((k*pointsInH + i2)*pointsInW + j2)*HISTO_SIZE + ind];
							
						float kd1 = kernel_histos[((k*pointsInH + i1)*pointsInW + j1)*HISTO_SIZE + ind];
						float kd2 = rhs->kernel_histos[((k*pointsInH + i2)*pointsInW + j2)*HISTO_SIZE + ind];

						float kd1_f=kernel_histos_function[((k*pointsInH + i1)*pointsInW + j1)*HISTO_SIZE + ind];
						float kd2_f=rhs->kernel_histos_function[((k*pointsInH + i2)*pointsInW + j2)*HISTO_SIZE + ind];
#if DISTANCE==MANHATTEN
						curSum+=(fabs(d1-d2));
						//curSum+=fast_sqrt(fabs(d1-d2));
#elif DISTANCE==EUC
						curSum+=(d1-d2)*(d1-d2);
#elif DISTANCE==INTERSECT
						curSum+=1-((d1<d2)?d1:d2);
#elif DISTANCE==SMIRNOFF
						cdf1+=d1;
						cdf2+=d2;
						curSum+=fabs(cdf1-cdf2);
						/*if(ind==HISTO_SIZE-1){
							if(cdf1<0.999 || cdf2<0.999)
								std::cerr<<cdf1<<' '<<cdf2<<'\n';
						}*/
#elif DISTANCE==KOLMOGOROFF
						cdf1+=d1;
						cdf2+=d2;
						float tmpDist=fabs(cdf1-cdf2);
						if(curSum>tmpDist)
							curSum=tmpDist;
#elif DISTANCE==LEMAN_ROSENBLATT
						cdf1+=d1;
						cdf2+=d2;
#if 0
						float summary=(d1*histosSum[(k*pointsInH + i1)*pointsInW + j1]+
							d2*rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2])/
							(histosSum[(k*pointsInH + i1)*pointsInW + j1]+rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2]);
#else
						float summary=(d1+d2)/2;
						//float summary=(kd1+kd2)/2;
#endif

						curSum+=summary*
							fabs(cdf1-cdf2);
							//(cdf1-cdf2)*(cdf1-cdf2);
#elif DISTANCE==CHI_SQUARE
						if(d1+d2>0)
							curSum+=(d1-d2)*(d1-d2)/(d1+d2);
#elif DISTANCE==KL
#if 0
						curSum+=d1*std::log(d1/d2);
#else
						curSum+=d1*(kd1_f-kd2_f);
#endif
#elif DISTANCE==SYM_KL
						curSum+=(d1-d2)*std::log(d1/d2);
#elif DISTANCE==JENSEN_SHANNON
						if(d1>0)
							curSum+=d1*log(2*d1/(d1+d2));
							//curSum+=(4*d1*d1-(d1+d2)*(d1+d2))/(d1+d2);
						if(d2>0)
							curSum+=d2*log(2*d2/(d1+d2));
							//curSum+=(4*d2*d2-(d1+d2)*(d1+d2))/(d1+d2);
						//curSum+=d1*log(2*kd1/(kd1+kd2))+d2*log(2*kd2/(kd1+kd2));
#elif DISTANCE==HOMOGENEITY ||  DISTANCE==SIMPLIFIED_HOMOGENEITY
#if 1
#else
						float kd1=d1;
						float kd2=d2;
#endif
#if 0
						float curProd=kd1*histosSum[(k*pointsInH + i1)*pointsInW + j1];
						float rhsCurProd = kd2*rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2];
						float summary = (curProd + rhsCurProd) / 
							(histosSum[(k*pointsInH + i1)*pointsInW + j1] + rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2]);
#else
						float summary=(d1+d2)/2.f;
						//float summary=(kd1+kd2)/2;
#endif
						if(summary>0){
							float inv_sum =
#if DISTANCE==HOMOGENEITY 
								log(summary);
#else
								2.f / summary;
#endif
							float tempSum=0;
							/*if (kd1>0){
								tempSum += kd1*(kd1*kd1 - summary*summary) / (2 * kd1*summary);
							}
							if (kd2>0){
								tempSum += kd2*(kd2*kd2 - summary*summary) / (2 * kd2*summary);
							}*/
							
#if DISTANCE==HOMOGENEITY 
							if (kd1>0)
							{
								tempSum += d1*(kd1_f - inv_sum);//*histosSum[(k*pointsInH + i1)*pointsInW + j1];
							}
#else
							tempSum += kd1_f*inv_sum*d1*(kd1*kd1 - summary*summary);
							//tempSum+=d1/2*(kd1/summary-summary/kd1);
#endif
#if DISTANCE==HOMOGENEITY 
							if (kd2>0)
							{
								tempSum += d2*(kd2_f - inv_sum);//*rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2];
							}
#else
							tempSum += kd2_f*inv_sum*d2*(kd2*kd2 - summary*summary);
							//tempSum+=d2/2*(kd2/summary-summary/kd2);
#endif
							//tempSum/=(histosSum[(k*pointsInH + i1)*pointsInW + j1]+rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2]);
							curSum+=tempSum;
						}
#elif  DISTANCE==PNN
#if 1
						float det=kd2;
#else
						float det=d2;
						//kd1=d1;
#endif
						if (d1 > 0 && det > 0){
							//curSum += d1*log(kd1 / det);
							curSum += d1*(kd1_f - kd2_f);
							//curSum+=d1*(kd1*kd1-det*det)/(2*kd1*det);
						}

#elif DISTANCE==MY_PNN
						float num1=0,num2=0,den=0;
						for(int ind1=0;ind1<HISTO_SIZE;++ind1){
							float curProd=histos[((k*pointsInH + i1)*pointsInW + j1)*HISTO_SIZE + ind1]*histosSum[(k*pointsInH + i1)*pointsInW + j1];
							float rhsCurProd=rhs->histos[((k*pointsInH + i2)*pointsInW + j2)*HISTO_SIZE + ind1]*rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2];
							float summary=(curProd+rhsCurProd)/
								(histosSum[(k*pointsInH + i1)*pointsInW + j1]+rhs->histosSum[(k*pointsInH + i2)*pointsInW + j2]);
								
							num1+=kernel.kernel[ind][ind1]*histos[((k*pointsInH + i1)*pointsInW + j1)*HISTO_SIZE + ind1];
							num2+=kernel.kernel[ind][ind1]*rhs->histos[((k*pointsInH + i2)*pointsInW + j2)*HISTO_SIZE + ind1];
							den+=kernel.kernel[ind][ind1]*summary;
						}
						if(den>0){
							if(num1>0)
								curSum+=d1*log(num1/den);
							if(num2>0)
								curSum+=d2*log(num2/den);
						}
#elif DISTANCE==BHATTACHARYYA
						curSum+=fast_sqrt(d1*d2);
#endif
					}
#if DISTANCE==BHATTACHARYYA
					if(curSum>0)
						curSum=-log(curSum);
#endif

					if(curSum<0)
						curSum=0;
					if(minSum>curSum){
						minSum=curSum;
					}
				}
			}
			minSum=fast_sqrt(minSum);
			res += weights[i*pointsInW + j] * (minSum);
		}
	}
	return res;
}


#ifndef PARALLEL_DISTANCE

/*
 * Euclidean distance functor
 */
template<class T>
struct CV_EXPORTS SqrtL2
{
    typedef T ValueType;
    typedef typename cv::Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        ResultType result = ResultType();
#if 1
		for(int i=0;i<16;++i){
	        ResultType cellResult = ResultType();
			for(int j=0;j<8;++j){
				ResultType diff = (ResultType)(a[i*8+j] - b[i*8+j]);
				cellResult += diff*diff;//abs(diff);
			}
			result+=(ResultType)fast_sqrt((float)cellResult);
		}
#else        
		for(int i1=0;i1<4;++i1){
			int iMin=i1>=DELTA?i1-DELTA:0;
			int iMax=i1+DELTA;
			if(iMax>=4)
				iMax=3;
			for(int j1=0;j1<4;++j1){
				int jMin=j1>=DELTA?j1-DELTA:0;
				int jMax=j1+DELTA;
				if(jMax>=4)
					jMax=3;
				float cellResult = FLT_MAX;

				for(int i2=iMin;i2<=iMax;++i2){
					for(int j2=jMin;j2<=jMax;++j2){
						float curSum=0.f;
						for(int k=0;k<8;++k){
							T cur_a=a[(i1*4+j1)*8+k];
							T cur_b=b[(i2*4+j2)*8+k];
							T diff = (cur_a - cur_b);
							//if(cur_a + cur_b>0)
								curSum += diff*diff;///(cur_a + cur_b);
						}
						if(curSum<0)
							curSum=0;
						if(cellResult>curSum){
							cellResult=curSum;
						}
					}
				}
				result+=
					(ResultType)cellResult;
					//(ResultType)fast_sqrt(cellResult);
			}
		}
#endif
        return result;
    }
};


float FaceImage::distance(const FaceImage* rhs, float* var){
    double res=0;
	double var_dist = 0;
#if 1 || defined(USE_DNN_FEATURES)
	const float* search_features = rhs->getFeatures();
    const float* features=getFeatures();
	double tmp;
	for (int k = 0; k<FEATURES_COUNT; ++k){
#if DISTANCE==MANHATTEN
		tmp=fabs(features[k] - search_features[k]);
#elif DISTANCE==CHI_SQUARE
		if ((features[k] + search_features[k])>0)
			tmp = (features[k] - search_features[k])*(features[k] - search_features[k])/(features[k] + search_features[k]);
		else
			tmp = 0;
#else // DISTANCE==EUC by default
		tmp = (features[k] - search_features[k])*(features[k] - search_features[k]);
#endif
		res += tmp;

	}
	res /= FEATURES_COUNT;
#if DISTANCE==EUC
	res=sqrt(res);
#endif
	return (float)res;
#else
#if DISTANCE==LOCAL_DESCRIPTORS

	std::vector<vector<cv::DMatch > > matches;
	std::vector<cv::DMatch > good_matches;
	//cv::FlannBasedMatcher matcher;
	cv::BruteForceMatcher<
		//SqrtL2
		cv::L2
		<float> > matcher;

	matcher.knnMatch(descriptors, rhs->descriptors, matches, 2);
	
    for(int i = 0; i < std::min(rhs->descriptors.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
    {
        if((matches[i][0].distance < 0.8*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
        {
            good_matches.push_back(matches[i][0]);
        }
    }
	res=-1.*good_matches.size();///rhs->keypoints.size();
	return (float)res;
#else

    for(int k=0;k<COLORS_COUNT;++k){
		for (int row = 0; row<pointsInH; row += NUM_OF_ROWS){
			res+=getRowDistance(rhs, k, row);
        }
    }
	return (float)(res / (COLORS_COUNT*pointsInH*pointsInW));
#endif
#endif
}
#else
#include "ThreadPool.h"
#include "PrivateThreadPool.h"
using namespace windowsthreadpool; 
namespace{
	CRITICAL_SECTION csCheckBest;
	PrivateThreadPool threadPool;

	class ParallelDistance{
	public:
		ParallelDistance();
		~ParallelDistance();
	};
	ParallelDistance::ParallelDistance(){
		InitializeCriticalSection(&csCheckBest);

		//threadPool.SetThreadpoolMin(thread_count);
		threadPool.SetThreadpoolMax(thread_count);
	}
	ParallelDistance::~ParallelDistance(){
		DeleteCriticalSection(&csCheckBest);
	}
	ParallelDistance parallelDistance;

	const FaceImage *lhsFaceImage, *rhsFaceImage;
	float res_distance;
	//volatile int tasksCount;
	DWORD WINAPI calculateRowDistance(PVOID param1, PVOID param2)
	{
		int k = (int)param1;
		int row = (int)param2;
		float tmpDist=lhsFaceImage->getRowDistance(rhsFaceImage,k,row);
		
		EnterCriticalSection(&csCheckBest);
		res_distance+=tmpDist;

		//--tasksCount;
		/*if(tasksCount==0)
			SetEvent(jobCompletedEvent);*/
		LeaveCriticalSection(&csCheckBest);
		//std::cerr<<tasksCount<<" hi\n";
		return 0;
	}
}

//int checks=10;
float FaceImage::distance(const FaceImage* rhs){
	res_distance=0;
	lhsFaceImage=this;
	rhsFaceImage=rhs;
	//tasksCount=COLORS_COUNT*pointsInH;
	//ResetEvent(jobCompletedEvent);
	//std::cerr<<CThreadPool::GetThreadPool().GetWorkingThreadCount()<<"\n";
	for(int k=0;k<COLORS_COUNT;++k)
	for(int row=0;row<pointsInH;row+=NUM_OF_ROWS)
			//CThreadPool::GetThreadPool().Run(calculateRowDistance,(LPVOID)k, (LPVOID)row);
			threadPool.QueueUserWorkItem(calculateRowDistance,(LPVOID)k, (LPVOID)row);
	/*
	//if(WaitForSingleObject(jobCompletedEvent,10000)==WAIT_TIMEOUT){
	do{
		if(!CThreadPool::GetThreadPool().WaitForAllTasksCompletion()){
			std::cout<<tasksCount<<" "<<CThreadPool::GetThreadPool().GetWorkingThreadCount()<<" timeout\n";
			for(int k=0;k<COLORS_COUNT;++k){
				for(int row=0;row<pointsInH;++row){
					res_distance+=getRowDistance(rhs, k, row);
				}
			}
			break;
		}

	}while(tasksCount!=0);
	*/
	threadPool.WaitForAll(); 
	return res_distance/(COLORS_COUNT*pointsInH*pointsInW);
}
#endif


