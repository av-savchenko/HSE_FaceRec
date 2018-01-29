#include "stdafx.h"

#include "RecognitionThreadPool.h"

#include<set>
#include<string>
using namespace std;

#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
using namespace cv;

#ifdef NEED_NON_METRIC_SPACE_LIB
using namespace similarity;
#endif

#define CURRENT_MATCHER KDTreeIndexParams()
//#define CURRENT_MATCHER CompositeIndexParams()
//#define CURRENT_MATCHER FLANNIndexParams(8, 30, 2)
#define INIT_MATCHER ,FLANNMatcher (new flann::CURRENT_MATCHER ,new flann::SearchParams(125*8/NUM_OF_THREADS))

FacesData::FacesData(vector<FaceImage*>& faces, float threshold, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices):
		faceImages(faces)
#ifdef NEED_DEM
		, directedEnumeration(faces, threshold, dist_matrix, dist_matrix_size, indices)
#endif
#ifdef NEED_SMALL_WORLD
		, smallWorld(faces, threshold, dist_matrix, dist_matrix_size, indices)
#endif
#ifdef NEED_FLANN
	INIT_MATCHER
#endif
#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	, fdb(faces)
#endif
{
	initMatchers();
}
FacesData::FacesData(vector<FaceImage*>& faces, float threshold, ImageDist* image_dists, const int image_dists_matrix_size):
		faceImages(faces)
#ifdef NEED_DEM
		, directedEnumeration(faces, threshold, image_dists, image_dists_matrix_size)
#endif
#ifdef NEED_FLANN
	INIT_MATCHER
#endif
#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	, fdb(faces)
#endif
{
	initMatchers();
}

FacesData::~FacesData()
{
#ifdef NEED_KD_TREE
	cvReleaseFeatureTree(dictionary_tree);
#endif
#ifdef NEED_FLANN
	delete flann_index;
#endif
    delete[] dictionary_features;
}
void FacesData::setImageCountToCheck(int imageCountToCheck)
{
	this->imageCountToCheck=imageCountToCheck;
#ifdef NEED_DEM
	directedEnumeration.setImageCountToCheck(imageCountToCheck);
#endif
#ifdef NEED_SMALL_WORLD
	smallWorld.setImageCountToCheck(imageCountToCheck);
#endif
#ifdef NEED_NON_METRIC_SPACE_LIB
	double ratio = 1.0;
	if (imageCountToCheck > 0){
		ratio = ((double)imageCountToCheck) / faceImages.size();
		char params[16];
		sprintf(params, "dbScanFrac=%.2f", ratio);
		index->SetQueryTimeParams(AnyParams({ params }));
	}
#endif
}
void FacesData::operator() ()
{
	switch(RecognitionThreadPool::nnMethod){
	case NNMethod::Simple:
		float tmpDist;

		for(int j=0;j<faceImages.size();++j){
			//tmpDist=faceImages[j]->distance(test);
			tmpDist=testImage->distance(faceImages[j]);
			/*for(int k=0;k<FEATURES_COUNT;++k){
				tmpDist+=fabs(dictionary_features[j*FEATURES_COUNT+k]-search_features[k]);
				//tmpDist+=(dictionary_features[j*FEATURES_COUNT+k]*search_features[k]);
			}
			tmpDist/=FEATURES_COUNT;*/
			if(tmpDist<closestDistance){
				closestDistance=tmpDist;
				closestImage=faceImages[j];
			}
		}
		break;
#ifdef NEED_DEM
	case NNMethod::DEM:
		//if(ind>=0)
		{
			closestImage = directedEnumeration.recognize(testImage);
			avgCheckedPercent+=directedEnumeration.getCheckedPercent();
			//std::cout<<directedEnumeration.getCheckedPercent()<<"\n";
			//closestImage=faceImages[ind];
			closestDistance=directedEnumeration.bestDistance;
			if(directedEnumeration.isFoundLessThreshold)
				DirectedEnumeration<FaceImage*>::terminateSearch = 1;
		}
		break;
#endif
#ifdef NEED_SMALL_WORLD
	case NNMethod::SMALL_WORLD:
		//if(ind>=0)
		{
			closestImage = smallWorld.recognize(testImage);
			avgCheckedPercent += smallWorld.getCheckedPercent();
			//std::cout<<smallWorld.getCheckedPercent()<<"\n";
			//closestImage=faceImages[ind];
			closestDistance = smallWorld.bestDistance;
			if (smallWorld.isFoundLessThreshold)
				SmallWorld<FaceImage*>::terminateSearch = 1;
		}
		break;
#endif

#ifdef NEED_KD_TREE
	case NNMethod::KD_TREE:
		{
			const int NUM_OF_NEIGHBORS=1;
			const int searchSize = 1;
			int matches_data[NUM_OF_NEIGHBORS];
			double distance_data[NUM_OF_NEIGHBORS];
			CvMat matches = cvMat(searchSize, NUM_OF_NEIGHBORS, CV_32SC1, matches_data);
			CvMat distance = cvMat(searchSize, NUM_OF_NEIGHBORS, CV_64FC1, distance_data);

			const float* search_features=testImage->getFeatures();
			CvMat search_features_mat = cvMat(1, FEATURES_COUNT, CV_32FC1, (void*)search_features);
			//std::cerr<<imageCountToCheck<<" hi\n";
			cvFindFeatures(dictionary_tree, &search_features_mat, &matches, &distance, NUM_OF_NEIGHBORS, imageCountToCheck);
			//std::cerr<<matches_data[0]<<"\n";
			closestImage=faceImages[matches_data[0]];
			closestDistance=distance_data[0];

				/*
		int i=0;
		for(int j=0;j<NUM_OF_NEIGHBORS;++j)
		//printf("PAIRS : %i vs %i\n", searchSize, (int)(dictionary_features->size()));
		   std::cout<<"#: KD dists: "<<distance_data[j]<<" diff="<<difference<<
				   "  , match: "<<CV_MAT_ELEM(matches,int,i,j)<<", className: "<<faceImages[matches_data[i*NUM_OF_NEIGHBORS+j]]->personName<<std::endl;

			//if (CV_MAT_ELEM(distance,float,i,1) < sqrt(0.8)*CV_MAT_ELEM(distance,float,i,0)) {
				   */
		}
		break;
#endif
#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	case SVM_CLASSIFIER:
	{
		/*vector<float> search_features (FEATURES_COUNT);
		for(int i=0;i<FEATURES_COUNT;++i)
			search_features[i]=testImage->getFeatures()[i];
		cv::Mat search_features_mat (1, FEATURES_COUNT, CV_32FC1, (void*)testImage->getFeatures());
		int predict_label = (int)svm.predict(search_features_mat);
		for(vector<FaceImage*>::iterator iter=faceImages.begin();iter!=faceImages.end();++iter)
		if((*iter)->personName==class_labels[predict_label]){
			closestImage=*iter;
			closestDistance=testImage->distance(closestImage);
			break;
		}*/
		map<string, float> classDistances;
		int bestInd = fdb.getDistanceMap(testImage, classDistances);
		closestImage = faceImages[bestInd];
		closestDistance = testImage->distance(closestImage);
		//if(closestImage==NULL)
		//std::cout<<predict_label<<" "<<class_labels[predict_label]<<" "<<testImage->personName<<"\n";
	}
	break;

#endif
#ifdef NEED_FLANN
	case NNMethod::FLANN:
		{
#if 0
			vector<vector<DMatch> > matches;
			Mat search_features_mat =Mat(1, FEATURES_COUNT, CV_32FC1, (void*)testImage->getFeatures());
			FLANNMatcher.knnMatch(search_features_mat,matches,1);
			//std::cerr<<matches[0][0].distance<<' '<<matches[0][0].imgIdx<<"\n";
			closestImage=faceImages[matches[0][0].imgIdx];
			closestDistance=matches[0][0].distance;
#else
			int bestInd;
			cvflann::Matrix<int> indices(&bestInd, 1, 1);
			cvflann::Matrix<float> dists(&closestDistance, 1, 1);
			//flann::Matrix<int> indices(new int[FEATURES_COUNT], 1, 1);
			//flann::Matrix<float> dists(&closestDistance, 1, 1);
			cvflann::Matrix<float> query((float*)testImage->getFeatures(), 1, 3 * FEATURES_COUNT);
			flann_index->knnSearch(query, indices, dists, 1, 
				cvflann::SearchParams(imageCountToCheck==0?faceImages.size():imageCountToCheck));
			//std::cerr<<bestInd<<" "<<closestDistance<<"\n";
			closestImage=faceImages[bestInd];
#endif
				/*
		int i=0;
		for(int j=0;j<NUM_OF_NEIGHBORS;++j)
		//printf("PAIRS : %i vs %i\n", searchSize, (int)(dictionary_features->size()));
		   std::cout<<"#: KD dists: "<<distance_data[j]<<" diff="<<difference<<
				   "  , match: "<<CV_MAT_ELEM(matches,int,i,j)<<", className: "<<faceImages[matches_data[i*NUM_OF_NEIGHBORS+j]]->personName<<std::endl;

			//if (CV_MAT_ELEM(distance,float,i,1) < sqrt(0.8)*CV_MAT_ELEM(distance,float,i,0)) {
				   */
		}
		break;
#endif

#ifdef NEED_NON_METRIC_SPACE_LIB
	case NON_METRIC_SPACE_LIB:
		
		/*MetricElement* me=msw.nnSearch(testImage);
		closestImage=me->face;
		closestDistance=me->currentDistance;*/
		unsigned K = 1; // 1-NN query
		int featuresCount = FEATURES_COUNT;
#ifndef USE_DNN_FEATURES
		featuresCount*=3;
#endif
		for (int j = 0; j < featuresCount; ++j)
			queryData[j] = testImage->getFeatures()[j];
		const Object*   queryObj = customSpace.CreateObjFromVect(-1, queryData);
		KNNQuery<float>   knnQ(&customSpace, queryObj, K);
		index->Search(&knnQ);
		KNNQueue<float>* res = knnQ.Result()->Clone();
		if (!res->Empty()){
			closestDistance = res->TopDistance();
			closestImage = faceImages[res->Pop()->id()];
		}
		else{
			closestImage = 0;
			closestDistance = FLT_MAX;
		}
		avgCheckedPercent += 100.*knnQ.DistanceComputations() / faceImages.size();
		knnQ.Reset();
		delete queryObj;
		break;
#endif
	}
}


void FacesData::initMatchers(){

	dictionary_features=NULL;
	int featuresCount = FEATURES_COUNT;
#ifndef USE_DNN_FEATURES
	featuresCount*=3;
#endif

#if defined(NEED_KD_TREE) || defined (NEED_FLANN)
	dictionary_features = new float[faceImages.size()*featuresCount];
    for(int j=0;j<faceImages.size();++j){
		for (int k = 0; k<featuresCount; ++k)
			dictionary_features[j*featuresCount + k] = faceImages[j]->getFeatures()[k];
    }
#endif
#ifdef NEED_KD_TREE
	// Build the k-d tree
    dictionary_features_mat = cvMat(faceImages.size(), FEATURES_COUNT, CV_32FC1, dictionary_features);
    dictionary_tree =
            cvCreateKDTree
            //cvCreateSpillTree
                                     (&dictionary_features_mat);

	int cur_class_ind=-1;
	set<string> classes;
	vector<float> dictionary_features_vec;
	vector<float> labels;
    for(int j=0;j<faceImages.size();++j){
		if(classes.find(faceImages[j]->personName)==classes.end()){
			++cur_class_ind;
			classes.insert(faceImages[j]->personName);
			class_labels.push_back(faceImages[j]->personName);
		}
		labels.push_back(cur_class_ind);
		for(int k=0;k<FEATURES_COUNT;++k)
            dictionary_features_vec.push_back(faceImages[j]->getFeatures()[k]);
    }
    cv::Mat trainingDataMat(faceImages.size(), FEATURES_COUNT, CV_32FC1, &dictionary_features_vec[0]);
	cv::Mat labelsMat(faceImages.size(), 1, CV_32FC1, &labels[0]);

	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma=1.0/FEATURES_COUNT;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-3);
	svm.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

	/*int errors=0;
	for ( int i = 0; i < trainingDataMat . rows ; i ++) {
		cv :: Mat sample = trainingDataMat .row(i);
		float result = svm.predict ( sample );
		if((int)result!=(int)labels[i])
			++errors;
	}
	std::cout<<"errors="<<errors<<"\n";*/
#endif
	
#ifdef NEED_FLANN
#if 0
	vector<Mat> descriptors (faceImages.size());
    for(int j=0;j<faceImages.size();++j){
		descriptors[j]=Mat(1,FEATURES_COUNT, CV_32FC1, &dictionary_features[j*FEATURES_COUNT]);
    }
	FLANNMatcher.add(descriptors);
	FLANNMatcher.train();
#else
	cvflann::Matrix<float> samplesMatrix((float*)dictionary_features, faceImages.size(), featuresCount);
    //Index<cvflann::ChiSquareDistance<float>> flann_index(samplesMatrix, cvflann::LinearIndexParams());
    flann_index=new cvflann::Index<CURRENT_DISTANCE<float>>(samplesMatrix, cvflann::KDTreeIndexParams(4));
    flann_index->buildIndex();
#endif
#endif


#ifdef NEED_NON_METRIC_SPACE_LIB
	//msw.addAll(faceImages);
	queryData.resize(featuresCount);

	vector<vector<float>> rawData(faceImages.size());
	for (int i = 0; i < faceImages.size(); ++i){
		rawData[i].resize(featuresCount);
		for (int j = 0; j < featuresCount; ++j)
			rawData[i][j] = faceImages[i]->getFeatures()[j];
	}
	customSpace.CreateDataset(dataSet, rawData);
	//customSpace.WriteDataset(dataSet, "training_dataset.txt");
	index =
		MethodFactoryRegistry<float>::Instance().
		CreateMethod(false /* print progress */,
#if 0
		"small_world_rand",
		"custom", &customSpace,
		dataSet,
		AnyParams(
			{
				"NN=25",//11 //15
				"initIndexAttempts=5",
				"initSearchAttempts=1",
				"indexThreadQty=4", /* 4 indexing threads */
			}
		)
#elif 1
		"perm_incsort",
		"custom", &customSpace,
		dataSet, 
		AnyParams(
		{
			"dbScanFrac=0.075",//0.7 for 8 threads, 0.4 for 1 thread // A fraction of the data set to scan
			"numPivot=32",   // 16 Number of pivots (should be < the # of objects)
		}
		)
#else
		"vptree",
		"custom", &customSpace,
		dataSet,
		AnyParams(
			{
				"alphaLeft=4.0",
				"alphaRight=17.0",
				"bucketSize=20"
			}
		)
#endif
		);
#endif
	}