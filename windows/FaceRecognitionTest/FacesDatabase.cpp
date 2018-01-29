#include "stdafx.h"

#include "FaceDatabase.h"

#include <set>
#include <algorithm>
#include <chrono>

using namespace std;

#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
using namespace cv;


//#define USE_SVM
#if defined(USE_SVM) && defined(NO_ALIGN)
//#define USE_OPENCV_SVM
#endif
namespace{
	void print_null(const char *s) {}
}


const int BEST_CLASSES = 5;
static inline void sortDistances(vector<pair<int, float> >& distanceToClass, const std::map<std::string, std::pair<int, float> >& classDistances){
	transform(classDistances.begin(), classDistances.end(), distanceToClass.begin(),
		[](const pair<std::string, std::pair<int, float> >& p){
		return make_pair(p.second.first, p.second.second);
	}
	);

	int bestClasses = distanceToClass.size();
	if (bestClasses>BEST_CLASSES)
		bestClasses = BEST_CLASSES;
	partial_sort(distanceToClass.begin(), distanceToClass.begin() + bestClasses, distanceToClass.end(),
		[](const pair<int, float>& lhs, const pair<int, float>& rhs) {
		return lhs.second < rhs.second; }
	);
	//cout << distanceToClass[0].second << ' ' << distanceToClass[distanceToClass.size() - 1].second << '\n';
}

static float getInvProbab(const vector<pair<int, float>>& distanceToClass, double scale=1){
	//scale = 1;
	int bestClasses = distanceToClass.size();
	if (bestClasses>BEST_CLASSES)
		bestClasses = BEST_CLASSES;
	float probab = 1.f;
	for (int i = 1; i<bestClasses; ++i){
		probab +=
			//distanceToClass[i].second / distanceToClass[0].second;
			exp(-100.f * (distanceToClass[i].second - distanceToClass[0].second));
	}
	//probab = 1 / probab;
	return probab;
}
static float getInvProbab(const std::map<std::string, std::pair<int, float> >& classDistances, 
		std::string bestClass, float minDist, double scale = 1){
	scale = 1;
	double probab = 1;
	for (auto class_dist_pair : classDistances)
		if (bestClass!=class_dist_pair.first)
			probab +=
			//distanceToClass[i].second / distanceToClass[0].second;
			exp(-100 * scale*(class_dist_pair.second.second - minDist));
	
	//probab = 1 / probab;
	return probab;
}
static void getProbabMap(map<string, float>& probabMap, const std::map<std::string, std::pair<int, float> >& classDistances){
	const float scale = 100;
	int bestClasses = classDistances.size();

	double sum = 0;
	for (auto classDistance : classDistances){
		sum += exp(-scale * classDistance.second.second);
	}
	sum = log(sum);
	for (auto classDistance : classDistances){
		double probab = -scale * classDistance.second.second - sum;
		if (probabMap.find(classDistance.first) == probabMap.end())
			probabMap[classDistance.first] = probab;
		else
			probabMap[classDistance.first] += probab;
	}
}
static float estimateThreshold(const std::vector<FaceImage*>& faces, int pyramidLevel){
#if 0
	vector<float> otherClassesDists(faces.size());
	for (int i = 0; i<faces.size(); ++i){
		otherClassesDists[i] = FLT_MAX;
		for (int j = 0; j<faces.size(); ++j){
			if (faces[i]->personName != faces[j]->personName){
				float tmp_dist = faces[j]->distance(faces[i]);
				if (tmp_dist<otherClassesDists[i])
					otherClassesDists[i] = tmp_dist;
			}
		}
	}
	float falseAcceptRate = 0.01f;
	int ind = (int)(otherClassesDists.size()*falseAcceptRate);
	std::nth_element(otherClassesDists.begin(), otherClassesDists.begin() + ind, otherClassesDists.end());
	std::cout << otherClassesDists[ind] << ' ' << ind << '\n';
	return otherClassesDists[ind];
#else
	//return pyramidLevel == 0 ? 0.435f : 0.393f;
	//return pyramidLevel == 0 ? 0.417f : 0.393f;
	//return pyramidLevel == 0 ? 0.463f : 0.734f;
	return -1;
#endif
}

vector<float>  FacesDatabase::pyramidThresholds;

const double pyramid_scale = //1.5;
							2;

FacesDatabase::FacesDatabase(vector<FaceImage*>& faces)
{
	threadPool.SetThreadpoolMax(8);
	const int PYRAMID_LEVELS =
#ifdef USE_PYRAMID
		(pyramid_scale==1.5)?3:2;
#else
		1;
#endif
	int dbSize = faces.size();

	pyramidFaces.resize(PYRAMID_LEVELS);
	pyramidTestImages.resize(PYRAMID_LEVELS);
	pyramidDistanceMap.resize(PYRAMID_LEVELS);
	initialPyramidDistanceMap.resize(PYRAMID_LEVELS);
	
	pyramidFaces[pyramidFaces.size()-1] = faces;
	//pyramidThresholds[pyramidFaces.size() - 1] = estimateThreshold(pyramidFaces[pyramidFaces.size() - 1], pyramidFaces.size() - 1);
	for (int i = pyramidFaces.size() - 2; i >= 0; --i){
		pyramidFaces[i].reserve(dbSize);
		vector<FaceImage*>& prevLevel = pyramidFaces[i+1];
		for (auto face : prevLevel){
			pyramidFaces[i].push_back(face->nextInPyramid(pyramid_scale));
		}
	}

	for (int i = 0; i < pyramidFaces.size(); ++i){
		for (int j = 0; j < dbSize; ++j)
			initialPyramidDistanceMap[i][pyramidFaces[i][j]->personName] =
				std::make_pair(j, 10000.f);
	}

	if (pyramidThresholds.empty()){
		pyramidThresholds.resize(PYRAMID_LEVELS);
		for (int i = 0; i < pyramidFaces.size(); ++i){
			pyramidThresholds[i] = estimateThreshold(pyramidFaces[i], i);
		}
	}
	//invProbabThreshold = 1;
	invProbabThreshold = 1.1f;// 175f;

#if defined(USE_SVM) || defined(USE_EIGENVALUES)
	//fill classes
	set<string> classes;
	vector<FaceImage*>::iterator iter;
	for (iter = pyramidFaces[0].begin(); iter != pyramidFaces[0].end(); ++iter)
		if(classes.find((*iter)->personName)==classes.end()){
			class_labels.push_back((*iter)->personName);
			classes.insert((*iter)->personName);
		}
#endif

#ifdef USE_SVM
	//prepare param
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;//RBF;
	param.degree = 3;
	param.gamma = 1.0/SVM_FEATURES_COUNT;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1; //0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	svm_set_print_string_function(print_null);

	prob_estimates = new float[class_labels.size()];
	sum_of_probs = new float[class_labels.size()];
	
	//fill models
#ifdef USE_OPENCV_SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma = 1.0 / SVM_FEATURES_COUNT;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-3);

	int cur_class_ind = -1;
	classes.clear();
	vector<float> dictionary_features;
	vector<float> labels;
	for (int j = 0; j<pyramidFaces[0].size(); ++j){
		if (classes.find(pyramidFaces[0][j]->personName) == classes.end()){
			++cur_class_ind;
			classes.insert(pyramidFaces[0][j]->personName);
		}
		labels.push_back(cur_class_ind);
		for(int k=0;k<SVM_FEATURES_COUNT;++k)
			dictionary_features.push_back(pyramidFaces[0][j]->getFeatures()[k]);
    }
	cv::Mat trainingDataMat(pyramidFaces[0].size(), SVM_FEATURES_COUNT, CV_32FC1, &dictionary_features[0]);
	cv::Mat labelsMat(pyramidFaces[0].size(), 1, CV_32FC1, &labels[0]);
	SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);
#else

	int modelInd=0;
	float avg_self_error_rate=0;
#ifdef NO_ALIGN
	{
		int num_of_train_data=1;
#else
	for(int k=0;k<COLORS_COUNT;++k)
        for(int i=0;i<POINTS_IN_H;++i)
            for(int j=0;j<POINTS_IN_W;++j)
			{
				int num_of_train_data=0;
				for(int di=-DELTA;di<=DELTA;++di){
					if(((i+di)<0) || ((i+di)>=POINTS_IN_H))
						continue;
					for(int dj=-DELTA;dj<=DELTA;++dj){
						if(((j+dj)<0) || ((j+dj)>=POINTS_IN_W))
							continue;
						++num_of_train_data;
					}
				}
#endif
				const int num_of_copies=1;
				num_of_train_data *= pyramidFaces[0].size()*num_of_copies;
				svm_problem prob;
				prob.l=num_of_train_data;
				svm_node* x_space = new svm_node[num_of_train_data*(SVM_FEATURES_COUNT+1)];
				prob.y = new double[prob.l];
				prob.x = new svm_node* [prob.l];

				int x_ind=0;
				int train_index=0;
				for(int cp=0;cp<num_of_copies;++cp){
					int cur_class_ind=-1;
					classes.clear();
					for (iter = pyramidFaces[0].begin(); iter != pyramidFaces[0].end(); ++iter){
						if(classes.find((*iter)->personName)==classes.end()){
							++cur_class_ind;
							classes.insert((*iter)->personName);
						}
#ifdef NO_ALIGN
						{
							{

#else
						for(int di=-DELTA;di<=DELTA;++di){
							if(((i+di)<0) || ((i+di)>=POINTS_IN_H))
								continue;
							for(int dj=-DELTA;dj<=DELTA;++dj){
								if(((j+dj)<0) || ((j+dj)>=POINTS_IN_W))
									continue;
#endif
								prob.x[train_index] = &x_space[x_ind];
								prob.y[train_index] = cur_class_ind;
								for(int featureInd=0;featureInd<SVM_FEATURES_COUNT;++featureInd){
									prob.x[train_index][featureInd].index=featureInd+1;
#ifdef NO_ALIGN
									prob.x[train_index][featureInd].value=(*iter)->getFeatures()[featureInd];
#else
									prob.x[train_index][featureInd].value=(*iter)->histos[k][i+di][j+dj][featureInd];
#endif
									++x_ind;
								}
								x_space[x_ind++].index = -1;
								++train_index;
							}
						}
					}
				}
				const char* error_msg = svm_check_parameter(&prob,&param);
				if(error_msg)
				{
					std::cout<<"ERROR: "<<error_msg<<"\n";
					return;
				}
				else
					;//std::cout<<"end train num_of_train_data="<<num_of_train_data<<"\n";
				const auto start = std::chrono::system_clock::now();
				models[modelInd] = svm_train(&prob,&param);
				const auto stop = std::chrono::system_clock::now();
				const auto d_actual = std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
				cout << "SVM train time=" << d_actual << endl;

				float self_error_rate=0;
				for(train_index=0;train_index<prob.l;++train_index){
					if(svm_predict(models[modelInd],prob.x[train_index] )!=prob.y[train_index] )
						++self_error_rate;
				}
				avg_self_error_rate+=self_error_rate/prob.l;
				++modelInd;

				//delete[] x_space;
				delete[] prob.x;
				delete[] prob.y;
			}
	
	avg_self_error_rate/=MODELS_COUNT;
	//std::cout<<"self error="<<100*avg_self_error_rate<<"%\n";
	//prepare query object
	for(int featureInd=0;featureInd<SVM_FEATURES_COUNT;++featureInd){
		query_object[featureInd].index=featureInd+1;
	}
	query_object[SVM_FEATURES_COUNT].index=-1;
#endif
#endif
#ifdef USE_EIGENVALUES
	int facesCount = pyramidFaces[0].size();
	vector<Mat> images(facesCount);
    vector<int> labels(facesCount);
	for(int i=0;i<facesCount;++i){
		images[i] = pyramidFaces[0][i]->pixelMat;
		labels[i] = find(class_labels.begin(), class_labels.end(), pyramidFaces[0][i]->personName) - class_labels.begin();
	}

	model = createEigenFaceRecognizer();
	//model = createFisherFaceRecognizer();
	//model = createLBPHFaceRecognizer();
	model->train(images, labels);
#endif
}

FacesDatabase::~FacesDatabase()
{
#ifdef USE_SVM
	delete[] sum_of_probs;
	delete[] prob_estimates;
#ifndef USE_OPENCV_SVM
	for(int modelInd=0;modelInd<MODELS_COUNT;++modelInd)
		svm_free_and_destroy_model(&models[modelInd]);
#endif

	svm_destroy_param(&param);
#endif
	for (int i = 0; i < pyramidFaces.size()-1; ++i)
		for (auto pyramid : pyramidFaces[i])
			delete pyramid;
}

#ifdef USE_SVM
int FacesDatabase::classify_by_SVM(FaceImage* testImg)
{
	int res = 1;
#ifdef USE_SVM
	int num_of_classes=class_labels.size();
	for(int class_ind=0;class_ind<num_of_classes;++class_ind)
		sum_of_probs[class_ind]=0;
	int modelInd=0;
#ifndef NO_ALIGN
	for(int k=0;k<COLORS_COUNT;++k)
        for(int i=0;i<POINTS_IN_H;++i)
            for(int j=0;j<POINTS_IN_W;++j)
#endif
#ifdef USE_OPENCV_SVM
	vector<float> search_features (SVM_FEATURES_COUNT);
	for(int i=0;i<SVM_FEATURES_COUNT;++i)
		search_features[i]=testImg->getFeatures()[i];
	cv::Mat search_features_mat (1, SVM_FEATURES_COUNT, CV_32FC1, (void*)&search_features[0]);
	int predict_label = (int)SVM.predict(search_features_mat);
	//std::cout<<"pred="<<predict_label<<"\n";
	++sum_of_probs[predict_label];
#else
			{
				for(int featureInd=0;featureInd<SVM_FEATURES_COUNT;++featureInd)
#ifdef NO_ALIGN
					query_object[featureInd].value=testImg->getFeatures()[featureInd];
#else
					query_object[featureInd].value=testImg->histos[k][i][j][featureInd];
#endif
#if 1
				int predict_label = (int)svm_predict(models[modelInd],query_object);
				++sum_of_probs[predict_label];
#else
				std::cerr<<models[modelInd]->SV[0][0].index<<"\n";
				float predict_label = svm_predict_probability(models[modelInd],query_object,prob_estimates);
				for(int class_ind=0;class_ind<num_of_classes;++class_ind)
					sum_of_probs[class_ind]+=prob_estimates[class_ind];
#endif
				++modelInd;
			}
	#endif

	int best_class_ind=-1;
	float best_probab=-1;
	for(int class_ind=0;class_ind<num_of_classes;++class_ind){
		sum_of_probs[class_ind]/=MODELS_COUNT;
		//std::cout<<"class="<<class_labels[class_ind]<<" probab="<<sum_of_probs[class_ind]<<"\n";
		if(best_probab<sum_of_probs[class_ind]){
			best_probab=sum_of_probs[class_ind];
			best_class_ind=class_ind;
		}
	}
	//std::cout<<"best_class_ind="<<best_class_ind<<" class="<<class_labels[best_class_ind]<<" test class="<<testImg->personName<<"\n";
	if (best_class_ind != -1){
		for (int i = 0; i < pyramidFaces[0].size(); ++i){
			if (pyramidFaces[0][i]->personName == class_labels[best_class_ind]){
				res = i;
				break;
			}
		}
	}
#endif
	return res;
}
#endif

#ifdef USE_EIGENVALUES
int FacesDatabase::classify_by_EigenValues(FaceImage* testImg){
	int best_class_ind = model->predict(testImg->pixelMat);
    int best_ind = -1;
	if (best_class_ind != -1){
		for (int i = 0; i < pyramidFaces[0].size(); ++i){
			if (pyramidFaces[0][i]->personName == class_labels[best_class_ind]){
				best_ind = i;
				break;
			}
		}
	}
	return best_ind;
}
#endif
DWORD WINAPI FacesDatabase::calculateDistanceTask(PVOID ptr1, PVOID ptr2){
	FacesDatabase* db = (FacesDatabase*)ptr1;
	int dbImageIndex = (int)ptr2;
	db->calculateDistance(dbImageIndex);
	return 0;
}
#define FROM_LOW_TO_HIGH
//volatile int num_of_terminations;
int FacesDatabase::getClosestIndex(int pyramidLevel, float bestDist){
#ifdef USE_SVM
	return classify_by_SVM(pyramidTestImages[0]);
#elif defined(USE_EIGENVALUES)
	return classify_by_EigenValues(pyramidTestImages[0]);
#else

	//testImage = pyramidTestImages[pyramidLevel];
	int numberOfModels = pyramidFaces[pyramidLevel].size();
	int closestDbImageIndex = -1;
	currentPyramidLevel = pyramidLevel;
	terminateSearch = 0;
	
	pyramidDistanceMap[pyramidLevel] = initialPyramidDistanceMap[pyramidLevel];
			
#if 1
	for (int i = 0; i<numberOfModels; ++i)
		threadPool.QueueUserWorkItem(calculateDistanceTask, (PVOID)this, (PVOID)i);
	threadPool.WaitForAll();
#else
	for (int i = 0; i < numberOfModels; ++i){
		calculateDistance(i);
		if (terminateSearch)
			return i;
	}
#endif

	
	vector<pair<int, float>> distanceToClass(pyramidDistanceMap[pyramidLevel].size());
	sortDistances(distanceToClass, pyramidDistanceMap[pyramidLevel]);
	float invProbab = getInvProbab(distanceToClass);
	if (bestDist > invProbab){
		bestDist = invProbab;
		closestDbImageIndex = distanceToClass[0].first;
	}
	
#ifdef USE_PYRAMID
	if (//false && 
#ifdef FROM_LOW_TO_HIGH
		pyramidLevel<(pyramidTestImages.size() - 1)
#else
		pyramidLevel>0
#endif
		&& invProbab >= invProbabThreshold)
	{
		int otherBestIndex = getClosestIndex(
#ifdef FROM_LOW_TO_HIGH
			pyramidLevel + 1,
#else
			pyramidLevel - 1,
#endif
			bestDist);
		if (otherBestIndex != -1)
			closestDbImageIndex = otherBestIndex;
	}
#endif
	if (closestDbImageIndex == -1 && pyramidLevel==0)
		closestDbImageIndex = distanceToClass[0].first;
	return closestDbImageIndex;
#endif
}

void FacesDatabase::setTestImage(FaceImage* testImg){
	int lastLevel = pyramidFaces.size() - 1;
	pyramidTestImages[lastLevel] = testImg;
	for (int i = lastLevel - 1; i >= 0; --i){
		if (pyramidTestImages[i])
			delete pyramidTestImages[i];
		pyramidTestImages[i] = pyramidTestImages[i + 1]->nextInPyramid(pyramid_scale);
		pyramidDistanceMap[i].clear();
	}
}
int FacesDatabase::getDistanceMap(FaceImage* testImg, std::map<std::string, float>& class2DistanceMap)
{
	setTestImage(testImg);
	return getDistanceMap(class2DistanceMap);
}
int FacesDatabase::getDistanceMap(std::map<std::string, float>& class2DistanceMap)
{
	int closestDbImageIndex = getClosestIndex(
#ifdef FROM_LOW_TO_HIGH
		0
#else
		pyramidTestImages.size() - 1
#endif
		);
	for (auto classDistance : pyramidDistanceMap[0]){
		class2DistanceMap[classDistance.first] = classDistance.second.second;
	}
	return closestDbImageIndex;
}
void FacesDatabase::calculateDistance(int dbImageIndex)
{
	FaceImage* modelImage = pyramidFaces[currentPyramidLevel][dbImageIndex];
	float tmpDist = 1000.f;
	EnterCriticalSection(&synchronizer.csCheckBestDistance);
	if (!terminateSearch){
		LeaveCriticalSection(&synchronizer.csCheckBestDistance);
		tmpDist = pyramidTestImages[currentPyramidLevel]->distance(modelImage);
		
		EnterCriticalSection(&synchronizer.csCheckBestDistance);
		
		std::map<std::string, std::pair<int, float> >& classDistances = pyramidDistanceMap[currentPyramidLevel];
		if (classDistances[modelImage->personName].second > tmpDist){
			classDistances[modelImage->personName] = std::make_pair(dbImageIndex, tmpDist);
		}
		if (tmpDist < pyramidThresholds[currentPyramidLevel])
		{
			terminateSearch = 1;
		}
	}

	LeaveCriticalSection(&synchronizer.csCheckBestDistance);
}