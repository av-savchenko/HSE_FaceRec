#include <string>
#include <opencv2/core/core.hpp>
#if 1
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

#endif
