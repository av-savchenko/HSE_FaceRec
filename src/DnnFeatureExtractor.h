#include <string>
#include <opencv2/core/core.hpp>
#if 1
//#define DETECT_AGE
struct DnnFeatureExtractorImpl;
class DnnFeatureExtractor
{
public:
	static DnnFeatureExtractor* GetInstance();
	void extractFeatures(std::string fileName, float* features);
    void extractFeatures(cv::Mat& grey_image, float* features, bool faces_features=true);
#ifdef DETECT_AGE
    const char* detect_age(cv::Mat& grey_image);
#endif
private:
	DnnFeatureExtractor();
	~DnnFeatureExtractor();
	DnnFeatureExtractor(DnnFeatureExtractor&){}

	DnnFeatureExtractorImpl* impl;
#ifdef DETECT_AGE
    DnnFeatureExtractorImpl* age_detect_impl;
#endif

};

#endif
