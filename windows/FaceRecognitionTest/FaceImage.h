#ifndef FACEIMAGE_H
#define FACEIMAGE_H


#include <Windows.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "distances.h"
#include "DnnFeatureExtractor.h"

//#define USE_PYRAMID

const int COLORS_COUNT=1;
const int DELTA=0;

#define USE_GRADIENTS

#ifdef USE_GRADIENTS

#ifdef USE_PYRAMID
const int POINTS_IN_W = 20;// (DELTA == 0) ? 5 : 10;
const int POINTS_IN_H = 20;//(DELTA==0)?5:10;
#else //USE_PYRAMID
const int POINTS_IN_W = 12;// 10;// (DELTA == 0) ? 5 : 10;
const int POINTS_IN_H = 12;// 10;//(DELTA==0)?5:10;
#endif
const int HISTO_SIZE=8; //18

#else //USE_GRADIENTS
const int POINTS_IN_W=5;//3;//5;
const int POINTS_IN_H=7;//4;//7;
const int HISTO_SIZE=32;
#endif
const int HISTO_COUNT=POINTS_IN_W*POINTS_IN_H;

//#define USE_EXTRA_FEATURES
#define USE_DNN_FEATURES

#ifdef USE_EXTRA_FEATURES
const int FEATURES_COUNT =
//3776;
//3776+1800;
1800;
#elif defined(USE_DNN_FEATURES)
const int FEATURES_COUNT =
#ifdef USE_RGB_DNN
//4096;
2048;
//512;
#else
						256;
#endif
#else
const int FEATURES_COUNT=COLORS_COUNT*POINTS_IN_W*POINTS_IN_H*HISTO_SIZE;
#endif

enum class ImageTransform{NONE, FLIP, NORMALIZE};

class FaceImage
{
public:
	FaceImage(const char* imageFileName, const std::string& personName = "", int pointsInW = POINTS_IN_W, int pointsInH = POINTS_IN_H, int rndImageRange = 0, ImageTransform img_transform = ImageTransform::NONE);
	FaceImage(cv::Mat& img, const std::string& personName = "", int pointsInW = POINTS_IN_W, int pointsInH = POINTS_IN_H);
#ifdef USE_DNN_FEATURES
	FaceImage(std::string fileName, std::string personName, std::vector<float>& features, bool normalize=true);
	static FaceImage* readFaceImage(std::ifstream& file);
	void writeFaceImage(std::ofstream& file);
#endif
	FaceImage(); //init noise image
	~FaceImage();

	float distance(const FaceImage* rhs, float* var=0);

	FaceImage* nextInPyramid(double scale = 2);

    std::string personName;
    std::string fileName;

    const float* getFeatures()const{
#if defined(USE_EXTRA_FEATURES) | defined(USE_DNN_FEATURES)
		return &featureVector[0];
#else
		return histos;// &histos[0][0][0][0];
#endif		
		
    }
	/*const*/ std::vector<float>& getFeatureVector(){
		return featureVector;
	}

	inline float getRowDistance(const FaceImage* rhs,int k, int row) const;

	int blockSize;
	friend class FacesDatabase;
	
	cv::Mat pixelMat;
	/*size_t ptr_histos_diff(){
		return (((char*)&kernel_histos[0][0][0][0]) - ((char*)&histos[0][0][0][0]));
	}*/
private:
	int width, height, pointsInW, pointsInH;
	std::vector<float> featureVector;
	float *histos; //[COLORS_COUNT][POINTS_IN_H][POINTS_IN_W][HISTO_SIZE];
	float *kernel_histos; //[COLORS_COUNT][POINTS_IN_H][POINTS_IN_W][HISTO_SIZE];
	float *kernel_histos_function; //[COLORS_COUNT][POINTS_IN_H][POINTS_IN_W][HISTO_SIZE];
	float *histosSum; //[COLORS_COUNT][POINTS_IN_H][POINTS_IN_W];
	int* pixels;
	std::vector<float> weights;// [POINTS_IN_H][POINTS_IN_W];

	void init(cv::Mat& img, int width, int height, int rndImageRange = 0, ImageTransform img_transform = ImageTransform::NONE);
	void initHistos(int* pixels);
	
	float *extractHOG(int& feat_size);
	template<typename T> float* extractLBP(int& feat_size, T* image);
	float* extractGabor(int& feat_size);

	FaceImage(int* pixels, int width, int height, const std::string& personName = "", int pointsInW = POINTS_IN_W, int pointsInH = POINTS_IN_H);

	class Kernel{
	public:
		Kernel();

		float kernel[HISTO_SIZE][HISTO_SIZE];
	} kernel;

#if DISTANCE==LOCAL_DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	static cv::DescriptorExtractor* extractor;
	static cv::FeatureDetector * detector;
#endif
 };


#if defined(_M_X64)
#define fast_sqrt(x) sqrt(x)
/*
inline float fast_sqrt1(float x)
{
unsigned int i = *(unsigned int*) &x;
// adjust bias
i  += 127 << 23;
// approximation of square root
i >>= 1;
return *(float*) &i;
} */
#else
float inline __declspec (naked) __fastcall fast_sqrt(float n)
{
	_asm fld dword ptr [esp+4]
	_asm fsqrt
	_asm ret 4
} 
#endif
#endif // FACEIMAGE_H
