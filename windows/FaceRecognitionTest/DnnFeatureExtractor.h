#ifndef __DNN_FEATURE_EXTRACTOR_H__
#define __DNN_FEATURE_EXTRACTOR_H__
#include <string>
#include <opencv2/core/core.hpp>

#define USE_RGB_DNN

struct DnnFeatureExtractorImpl;
class DnnFeatureExtractor
{
public:
	static DnnFeatureExtractor* GetInstance();
	void extractFeatures(std::string fileName, float* features);
	void extractFeatures(cv::Mat& grey_image, float* features);
private:
	DnnFeatureExtractor();
	~DnnFeatureExtractor();
	DnnFeatureExtractor(DnnFeatureExtractor&){}

	DnnFeatureExtractorImpl* impl;

};
#endif //__DNN_FEATURE_EXTRACTOR_H__

