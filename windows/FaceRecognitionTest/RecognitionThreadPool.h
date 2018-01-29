#ifndef __RECOGNITION_THREAD_POOL__
#define __RECOGNITION_THREAD_POOL__

#include <windows.h>
#include <vector>
#include <opencv/cv.h>

#include "FaceImage.h"
#include "FaceDatabase.h"
#include "ThreadPool.h"
#include "DirectedEnumeration.h"
#include "SmallWorld.h"
#include "HistoDistances.h"

#include "PrivateThreadPool.h"
using namespace windowsthreadpool; 

#define NEED_DEM
//#define NEED_SMALL_WORLD
//#define NEED_KD_TREE
#define NEED_FLANN
//#define NEED_NON_METRIC_SPACE_LIB

#ifdef NEED_NON_METRIC_SPACE_LIB
#include "object.h"
#include "space/space_vector_gen.h"
#include "init.h"
#include "index.h"
#include "params.h"
#include "rangequery.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "ztimer.h"
#endif

enum NNMethod{
	Simple
	, DEM
	, KD_TREE
	, SVM_CLASSIFIER
	,FLANN
	,NON_METRIC_SPACE_LIB,
	SMALL_WORLD
};


#define NUM_OF_THREADS 1

class FacesData;

class RecognitionThreadPool{
public:
	RecognitionThreadPool(std::vector<FaceImage*>& dbImages, float falseAcceptRate);
	~RecognitionThreadPool();

	void init();
	float getAverageCheckedPercent();

	FaceImage* recognizeTestImage(FaceImage* test);
	void setImageCountToCheck(int imageCountToCheck);

	static NNMethod nnMethod;

	static void divideDatabasesIntoClasses(int numOfClasses, int dbSize, float* distMatrix, vector<int>* indices);
private:
	PrivateThreadPool threadPool;

	std::vector<FacesData*> facesData;

	HANDLE jobCompletedEvent;
	CRITICAL_SECTION csTasksCount;
	//volatile int tasksCount;

	FaceImage* closestImage;
	float closestDistance;

	void taskCompleted(FacesData* faces);
	static DWORD WINAPI processTask(LPVOID param1, LPVOID param2);

	ImageDist* image_dists;
};

struct UserData{
	LPVOID pData;
	LPVOID pDataAux;
	UserData(LPVOID pData, LPVOID pDataAux){
		this->pData=pData;
		this->pDataAux=pDataAux;
	}
};

class FacesData{
	friend class RecognitionThreadPool;
public:
	FacesData(std::vector<FaceImage*>& faceImages, float threshold, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices);
	FacesData(std::vector<FaceImage*>& faceImages, float threshold, ImageDist* image_dists, const int image_dists_matrix_size);
	~FacesData();

	void setTestImage(FaceImage* test)
	{
		closestImage=0;
		closestDistance=1000000;
		testImage=test;
	}
	void operator() ();

	void setImageCountToCheck(int imageCountToCheck);

private:
	void init(){
		avgCheckedPercent=0;
	}
	int avgCheckedPercent;

	void initMatchers();
	
	std::vector<FaceImage*> faceImages;
	FaceImage* testImage;
	FaceImage* closestImage;
	float closestDistance;

	float* dictionary_features;
	int imageCountToCheck;
#ifdef NEED_KD_TREE
	CvMat dictionary_features_mat;
	CvFeatureTree *dictionary_tree;

	/*CvSVM svm;
	std::vector<std::string> class_labels;*/

#endif
#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	FacesDatabase fdb;
#endif
#ifdef NEED_DEM
	DirectedEnumeration<FaceImage*> directedEnumeration;
#endif
#ifdef NEED_SMALL_WORLD
	SmallWorld<FaceImage*> smallWorld;
#endif
	
#ifdef NEED_FLANN
	cv::FlannBasedMatcher FLANNMatcher;
	cvflann::Index<CURRENT_DISTANCE<float> > *flann_index;
#endif
#ifdef NEED_NON_METRIC_SPACE_LIB
	//MetrizedDirectedEnumeration msw;
	similarity::VectorSpaceGen<float, CURRENT_DISTANCE<float> >  customSpace;
	similarity::ObjectVector    dataSet;
	similarity::Index<float>*   index;
	std::vector<float> queryData;
#endif
};


#endif //__RECOGNITION_THREAD_POOL__
