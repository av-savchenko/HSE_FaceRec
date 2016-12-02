#include "DnnFeatureExtractor.h"
#if 1
#include <memory>
#include <vector>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CPU_ONLY
#define USE_OPENCV

//#define H5_SIZEOF_SSIZE_T 1
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
using namespace caffe;

#include <QtCore>

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
    FLAGS_minloglevel = 3;
	Caffe::set_mode(Caffe::CPU);

    QFile f_proto(":/caffe_models/LCNN_prediction.prototxt");
    QTemporaryFile* tmp_proto=QTemporaryFile::createNativeFile(f_proto);
    tmp_proto->setAutoRemove(true);
    net_.reset(new Net<float>(tmp_proto->fileName().toStdString().c_str(), TEST));
    delete tmp_proto;

    QFile f_model(":/caffe_models/LCNN.caffemodel");
    QTemporaryFile* tmp_model=QTemporaryFile::createNativeFile(f_model);
    tmp_model->setAutoRemove(true);
    net_->CopyTrainedLayersFrom(tmp_model->fileName().toStdString().c_str());
    delete tmp_model;
    const string feature_blob_name="eltwise_fc1";

	input_layer_ = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net_->layers()[0]);
	num_channels_ = input_layer_->channels();
	input_geometry_ = cv::Size(input_layer_->width(), input_layer_->height());
    feature_blob_ = net_->blob_by_name(feature_blob_name);
    //cout << input_geometry_.width << ' ' << input_geometry_.height << ' ' << feature_blob->count() << " hi!!\n";
}
DnnFeatureExtractor* DnnFeatureExtractor::GetInstance(){
	static DnnFeatureExtractor fd;
	return &fd;
}
DnnFeatureExtractor::DnnFeatureExtractor()
{
	impl = new DnnFeatureExtractorImpl();
}
DnnFeatureExtractor::~DnnFeatureExtractor(){
	delete impl;
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
	cv::Mat img_resized;
	cv::resize(grey_image, img_resized, impl->input_geometry_);
#if 0
	Blob<float>* input_layer = impl->net_->input_blobs()[0];
	caffe::Datum* datum = new caffe::Datum();
	CVMatToDatum(img_resized, datum);
	caffe::Blob<float>* input_blob = new caffe::Blob<float>(1, datum->channels(), datum->height(), datum->width());
	//get the blobproto
	caffe::BlobProto blob_proto;
	blob_proto.set_num(1);
	blob_proto.set_channels(datum->channels());
	blob_proto.set_height(datum->height());
	blob_proto.set_width(datum->width());

	const string& data = datum->data();
	for (uint32_t i = 0; i < data.length(); ++i) {
		blob_proto.add_data((uint8_t)data[i]);
	}

	//set data into blob
	input_blob->FromProto(blob_proto);

	std::vector<caffe::Blob<float>*> input_cnn;
	input_cnn.push_back(input_blob);

	std::cout << "hi " << datum->channels() << ' ' << data.length() << "\n";
	impl->net_->Forward(input_cnn);
#else
	float loss = 0.0;
	std::vector<cv::Mat> dv = { img_resized };
	std::vector<int> dvl = { 0 };
	impl->input_layer_->AddMatVector(dv, dvl);
	std::vector<Blob<float>*> results = impl->net_->ForwardPrefilled(&loss);
#endif


	for (int i = 0; i < impl->feature_blob_->count(); ++i)
		features[i] = impl->feature_blob_->cpu_data()[i];

}

#if 0
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	// Get convolution layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetConvolutionLayer(
		const LayerParameter& param) {
		ConvolutionParameter_Engine engine = param.convolution_param().engine();
		if (engine == ConvolutionParameter_Engine_DEFAULT) {
			engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = ConvolutionParameter_Engine_CUDNN;
#endif
		}
		if (engine == ConvolutionParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == ConvolutionParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

	// Get pooling layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
		PoolingParameter_Engine engine = param.pooling_param().engine();
		if (engine == PoolingParameter_Engine_DEFAULT) {
			engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = PoolingParameter_Engine_CUDNN;
#endif
		}
		if (engine == PoolingParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == PoolingParameter_Engine_CUDNN) {
			PoolingParameter p_param = param.pooling_param();
			if (p_param.pad() || p_param.pad_h() || p_param.pad_w() ||
				param.top_size() > 1) {
				LOG(INFO) << "CUDNN does not support padding or multiple tops. "
					<< "Using Caffe's own pooling layer.";
				return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
			}
			return shared_ptr<Layer<Dtype> >(new CuDNNPoolingLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

	// Get relu layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
		ReLUParameter_Engine engine = param.relu_param().engine();
		if (engine == ReLUParameter_Engine_DEFAULT) {
			engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = ReLUParameter_Engine_CUDNN;
#endif
		}
		if (engine == ReLUParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == ReLUParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

	// Get sigmoid layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetSigmoidLayer(const LayerParameter& param) {
		SigmoidParameter_Engine engine = param.sigmoid_param().engine();
		if (engine == SigmoidParameter_Engine_DEFAULT) {
			engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = SigmoidParameter_Engine_CUDNN;
#endif
		}
		if (engine == SigmoidParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == SigmoidParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNSigmoidLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

	// Get softmax layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
		SoftmaxParameter_Engine engine = param.softmax_param().engine();
		if (engine == SoftmaxParameter_Engine_DEFAULT) {
			engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = SoftmaxParameter_Engine_CUDNN;
#endif
		}
		if (engine == SoftmaxParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == SoftmaxParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

	// Get tanh layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetTanHLayer(const LayerParameter& param) {
		TanHParameter_Engine engine = param.tanh_param().engine();
		if (engine == TanHParameter_Engine_DEFAULT) {
			engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = TanHParameter_Engine_CUDNN;
#endif
		}
		if (engine == TanHParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == TanHParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNTanHLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

	// Layers that use their constructor as their default creator.

	REGISTER_LAYER_CLASS(Accuracy);
	REGISTER_LAYER_CLASS(AbsVal);
	REGISTER_LAYER_CLASS(ArgMax);
	REGISTER_LAYER_CLASS(BN);
	REGISTER_LAYER_CLASS(BNLL);
	REGISTER_LAYER_CLASS(Concat);
	REGISTER_LAYER_CLASS(ContrastiveLoss);
	REGISTER_LAYER_CLASS(Covariance);
	REGISTER_LAYER_CLASS(Data);
	REGISTER_LAYER_CLASS(Deconvolution);
	REGISTER_LAYER_CLASS(Dropout);
	REGISTER_LAYER_CLASS(DummyData);
	REGISTER_LAYER_CLASS(Eltwise);
	REGISTER_LAYER_CLASS(Embed);
	REGISTER_LAYER_CLASS(EuclideanLoss);
	REGISTER_LAYER_CLASS(Exp);
	REGISTER_LAYER_CLASS(Filter);
	REGISTER_LAYER_CLASS(Flatten);
	REGISTER_LAYER_CLASS(HDF5Data);
	REGISTER_LAYER_CLASS(HDF5Output);
	REGISTER_LAYER_CLASS(HingeLoss);
	REGISTER_LAYER_CLASS(Im2col);
	REGISTER_LAYER_CLASS(ImageData);
	REGISTER_LAYER_CLASS(InfogainLoss);
	REGISTER_LAYER_CLASS(InnerProduct);
	REGISTER_LAYER_CLASS(Insanity);
	REGISTER_LAYER_CLASS(Local);
	REGISTER_LAYER_CLASS(Log);
	REGISTER_LAYER_CLASS(LRN);
	REGISTER_LAYER_CLASS(MemoryData);
	REGISTER_LAYER_CLASS(MultinomialLogisticLoss);
	REGISTER_LAYER_CLASS(MVN);
	REGISTER_LAYER_CLASS(Normalize);
	REGISTER_LAYER_CLASS(Power);
	REGISTER_LAYER_CLASS(PReLU);
	REGISTER_LAYER_CLASS(Reduction);
	REGISTER_LAYER_CLASS(Reshape);
	REGISTER_LAYER_CLASS(ROIPooling);
	REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);
	REGISTER_LAYER_CLASS(Silence);
	REGISTER_LAYER_CLASS(Slice);
	REGISTER_LAYER_CLASS(SmoothL1Loss);
	REGISTER_LAYER_CLASS(SoftmaxWithLoss);
	REGISTER_LAYER_CLASS(Split);
	REGISTER_LAYER_CLASS(SPP);
	REGISTER_LAYER_CLASS(Threshold);
	REGISTER_LAYER_CLASS(Tile);
	REGISTER_LAYER_CLASS(Transformer);
	REGISTER_LAYER_CLASS(TripletLoss);
	REGISTER_LAYER_CLASS(WindowData);


}  // namespace caffe
#endif
#endif
