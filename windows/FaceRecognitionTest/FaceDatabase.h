#ifndef __FACE_DATABASE_H__
#define __FACE_DATABASE_H__


//#define USE_SVM
//#define USE_EIGENVALUES

#ifdef USE_SVM
#include "svm.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "PrivateThreadPool.h"
#include "FaceImage.h"


class FacesDatabase{
public:
	FacesDatabase(std::vector<FaceImage*>& faces);
	~FacesDatabase();

	//FaceImage* getClosestImage(int pyramidLevel);
	int getClosestIndex(int pyramidLevel, float bestDist = 100000);
	int getDistanceMap(FaceImage* testImg, std::map<std::string, float>& class2DistanceMap);

	void setTestImage(FaceImage* testImg);
	int getDistanceMap(std::map<std::string, float>& class2DistanceMap);
private:
	//methods
	void calculateDistance(int dbImageIndex);
	static DWORD WINAPI calculateDistanceTask(PVOID ptr1, PVOID ptr2);

	//fields
	std::vector<std::vector<FaceImage*>> pyramidFaces;
	std::vector<FaceImage*>  pyramidTestImages;
	std::vector<std::map<std::string, std::pair<int, float> > > pyramidDistanceMap;
	std::vector<std::map<std::string, std::pair<int, float> > > initialPyramidDistanceMap;
	int terminateSearch;
	static std::vector<float>  pyramidThresholds;

	windowsthreadpool::PrivateThreadPool threadPool;
	float invProbabThreshold;
	int currentPyramidLevel;

	//synchronization
	class Synchronizer{
	public:
		CRITICAL_SECTION csCheckBestDistance;

		Synchronizer()
		{
			InitializeCriticalSection(&csCheckBestDistance);
		}
		~Synchronizer()
		{
			DeleteCriticalSection(&csCheckBestDistance);
		}
	} synchronizer;

#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	std::vector<std::string> class_labels;
#endif

#ifdef USE_SVM

	//SVM
#define NO_ALIGN
#ifdef NO_ALIGN
	static const int MODELS_COUNT = 1;
	static const int SVM_FEATURES_COUNT = ::FEATURES_COUNT;

	//CvSVM SVM;
#else
	static const int MODELS_COUNT = COLORS_COUNT*POINTS_IN_W*POINTS_IN_H;
	static const int SVM_FEATURES_COUNT = HISTO_SIZE;
#endif
	svm_parameter param;
	svm_model *models[MODELS_COUNT];
	svm_node query_object[FEATURES_COUNT + 1];
	float *prob_estimates, *sum_of_probs;
	int classify_by_SVM(FaceImage* testImg);
#endif

#ifdef USE_EIGENVALUES
	cv::Ptr<cv::FaceRecognizer> model;
	int classify_by_EigenValues(FaceImage* testImg);
#endif
};
#endif // __FACE_DATABASE_H__