#include "DnnFeatureExtractor.h"
#include <memory>
#include <vector>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef COMPILE_DNN
//#define CPU_ONLY
#define CMAKE_WINDOWS_BUILD
#define USE_OPENCV
#define USE_CUDNN
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
using namespace caffe;

#ifdef USE_RGB_DNN
#if 1
const char* FEATURE_LAYER ="fc7";
const char* PROTO_FILE = "D:\\src_code\\DNN_models\\vggface\\VGG_FACE_predict.prototxt";
const char* PROTO_MODEL = "D:\\src_code\\DNN_models\\vggface\\VGG_FACE.caffemodel";
#elif 0
const char* FEATURE_LAYER = "pool5/7x7_s1";
const char* PROTO_FILE = "D:\\src_code\\DNN_models\\resnet50_ft_caffe\\resnet50_predict.prototxt";
const char* PROTO_MODEL = "D:\\src_code\\DNN_models\\resnet50_ft_caffe\\resnet50_ft.caffemodel";
#elif 1
const char* FEATURE_LAYER = "pool5";
const char* PROTO_FILE = "D:\\src_code\\DNN_models\\resnet-101\\ResNet-101-deploy_augmentation.prototxt";
const char* PROTO_MODEL = "D:\\src_code\\DNN_models\\resnet-101\\snap_resnet__iter_120000.caffemodel";
#elif 1
const char* FEATURE_LAYER = "fc5";
const char* PROTO_FILE = "D:\\src_code\\DNN_models\\ydwen\\face_predict.prototxt";
const char* PROTO_MODEL = "D:\\src_code\\DNN_models\\ydwen\\face_model.caffemodel";
#endif
#else
const char* FEATURE_LAYER ="eltwise_fc1";
const char* PROTO_FILE = "D:\\src_code\\DNN_models\\lightCNN\\LCNN_prediction.prototxt";
const char* PROTO_MODEL = "D:\\src_code\\DNN_models\\lightCNN\\LCNN_C.caffemodel";
#endif

struct DnnFeatureExtractorImpl{
public:
	DnnFeatureExtractorImpl();
	std::shared_ptr<caffe::Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	shared_ptr<Blob<float> > feature_blob_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> input_layer_;
};
DnnFeatureExtractorImpl::DnnFeatureExtractorImpl(){
	FLAGS_minloglevel = google::ERROR;
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
	Caffe::set_solver_count(1);
	
	net_.reset(new Net<float>(PROTO_FILE, TEST));
	net_->CopyTrainedLayersFrom(PROTO_MODEL);

	input_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net_->layers()[0]);
	num_channels_ = input_layer_->channels();
	input_geometry_ = cv::Size(input_layer_->width(), input_layer_->height());
	feature_blob_ = net_->blob_by_name(FEATURE_LAYER);

	std::cout << "DNN params: "<<input_geometry_.width << ' ' << input_geometry_.height << ' ';
	if (feature_blob_)
		std::cout << feature_blob_->count() << std::endl;
	else
		std::cout << " layer " << FEATURE_LAYER << " not found\n";

}
#endif //COMPILE_DNN
DnnFeatureExtractor* DnnFeatureExtractor::GetInstance(){
	static DnnFeatureExtractor fd;
	return &fd;
}
DnnFeatureExtractor::DnnFeatureExtractor()
{
#ifdef COMPILE_DNN
	impl = new DnnFeatureExtractorImpl();
#endif
}
DnnFeatureExtractor::~DnnFeatureExtractor(){
#ifdef COMPILE_DNN
	delete impl;
#endif
}
void DnnFeatureExtractor::extractFeatures(std::string fileName, float* features)
{
	cv::Mat grey_image = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
	if (!grey_image.data) {
		std::cout << "Could not open or find input file.\n";
		return;
	}
	extractFeatures(grey_image, features);
}
void DnnFeatureExtractor::extractFeatures(cv::Mat& grey_image, float* features){
#ifdef COMPILE_DNN
	cv::Mat img_resized;
	cv::resize(grey_image, img_resized, impl->input_geometry_);

	float loss = 0.0;
	std::vector<cv::Mat> dv = { img_resized };
	std::vector<int> dvl = { 0 };
	impl->input_layer_->AddMatVector(dv, dvl);
	std::vector<Blob<float>*> results = impl->net_->ForwardPrefilled(&loss);

	//std::cout << impl->feature_blob_->count() << '\n';
	for (int i = 0; i < impl->feature_blob_->count(); ++i)
		features[i] = impl->feature_blob_->cpu_data()[i];
#endif //COMPILE_DNN
}

