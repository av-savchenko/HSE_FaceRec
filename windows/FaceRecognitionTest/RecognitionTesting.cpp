#include "stdafx.h"

#include <windows.h>
#include <strsafe.h>

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>

#include "RecognitionTesting.h"
#include "FaceDatabase.h"
#include "db.h"
#include "PrivateThreadPool.h"

using namespace std;

const int rndImageRange = 0;

//#define USE_REJECT_OPTION

#ifdef USE_REJECT_OPTION
#include "DirectedEnumeration.h"
enum RejectType { MinDistance, DiffDistance, DistanceRatio, PosteriorProbab };
const RejectType rejectType = RejectType::PosteriorProbab;

//probab parameters
const float probabMultiplier = 20;
const float probabModelsCount = 20;
								//40;

//#define USE_ALL_DISTANCES

enum RO_STATISTICS{ FalseAcceptRate=0, RejectRate, FalseRejectRate, EndRoIndex};

const float fixedFAR =
#ifdef USE_ALL_DISTANCES
0.01;
#else
0.05;
#endif

#include <map>
template<typename A, typename B>
std::pair<B, A> flip_pair(const std::pair<A, B> &p)
{
	return std::pair<B, A>(p.second, p.first);
}

template<typename A, typename B>
std::multimap<B, A> flip_map(const std::map<A, B> &src)
{
	std::multimap<B, A> dst;
	std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
		flip_pair<A, B>);
	return dst;
}
#endif



void processTestSet(FacesDatabase& db, vector<FaceImage*>& faceImages, vector<FaceImage*>& testImages,
	float& total, int& errorsCount, long& testsCount, double& total_sens, double& total_spec, double& total_precision
#ifdef USE_REJECT_OPTION
	, float& threshold, int ro_statistics[RO_STATISTICS::EndRoIndex]
#endif
	){
	int bestInd = -1;
	float bestDist = 100000, tmpDist;
	map<string, map<string, int> > current_conf_matrix;

	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);

	map<string, float> classDistances;

	for (std::vector<FaceImage*>::iterator iter = testImages.begin(); iter != testImages.end(); ++iter, ++testsCount){
#if 1
		FaceImage* test = *iter;
#else
		FaceImage testFace((*iter)->fileName.c_str(),(*iter)->personName, POINTS_IN_W, POINTS_IN_H, rndImageRange);
		FaceImage* test=&testFace;
#endif
		FaceImage* closestDbImage = 0;
		db.setTestImage(test);
		QueryPerformanceCounter(&start);

#if 1
		//closestDbImage = db.getClosestImage(test);
		bestInd = db.getDistanceMap(classDistances);
		closestDbImage = faceImages[bestInd];
		bestDist = test->distance(closestDbImage);
#else
		bestInd = -1;
		bestDist = 100000;
		//const float* search_features=test->getFeatures();
		for (int j = 0; j<faceImages.size(); ++j){
			//tmpDist=faceImages[j]->distance(test);
			tmpDist = test->distance(faceImages[j]);
			bool classNameExists = classDistances.find(faceImages[j]->personName) != classDistances.end();
			if (!classNameExists || (classNameExists && classDistances[faceImages[j]->personName] > tmpDist))
				classDistances[faceImages[j]->personName] = tmpDist;
			if (tmpDist<bestDist){
				bestDist = tmpDist;
				bestInd = j;
			}
		}
		closestDbImage = faceImages[bestInd];
#endif
		QueryPerformanceCounter(&end);
		total += (float)(end.QuadPart - start.QuadPart) / (/*faceImages.size()*/freq.QuadPart)*1000000.0;

		if (closestDbImage == 0 || test->personName != closestDbImage->personName){
			++errorsCount;
		}
		++current_conf_matrix[test->personName][closestDbImage->personName];
#ifdef USE_REJECT_OPTION
		//if (closestDbImage != 0)
		{
			float roFeature;
			if (rejectType == RejectType::MinDistance){
				roFeature = bestDist;
			}
			else if (rejectType == RejectType::DiffDistance || rejectType == RejectType::DistanceRatio){
				if (bestDist>0){
					float bestOtherDist = FLT_MAX;
					for (const auto &classDistance : classDistances)
					{
						if (classDistance.first != closestDbImage->personName && bestOtherDist>classDistance.second){
							bestOtherDist = classDistance.second;
						}
					}

					if (rejectType == RejectType::DiffDistance)
						roFeature = bestDist - bestOtherDist;
					else
						roFeature = bestDist / bestOtherDist;
				}
			}
			else if (rejectType == RejectType::PosteriorProbab){
				roFeature = 0;
				multimap<float, string> distanceClasses = flip_map(classDistances);
#if 1
				int probabCounter = 0;
				for (const auto &distanceClass : distanceClasses){
					if (++probabCounter>probabModelsCount)
						break;
					roFeature += exp(-probabMultiplier*(distanceClass.first - bestDist));
				}
#else
				for (const auto &classDistance : classDistances)
				{
					roFeature += exp(-probabMultiplier*(classDistance.second - bestDist));
				}
#endif
				//roFeature = 1 / roFeature;
			}

			bool reject = roFeature >= threshold;// (rejectType == RejectType::MinDistance) ? roFeature >= threshold : roFeature < threshold;

			if (!reject && test->personName != closestDbImage->personName)
				++ro_statistics[FalseAcceptRate];
			if (reject){
				++ro_statistics[RejectRate];
				if (test->personName == closestDbImage->personName)
					++ro_statistics[FalseRejectRate];
			}
		}
#endif
	}

	double sensitivity = 0, precision = 0;
	for (auto conf_iter : current_conf_matrix){
		string c = conf_iter.first;
		map<string, int>& row = conf_iter.second;
		double total_c = 0;
		for (auto iter : row)
			total_c += iter.second;
		double tp_c = (row.find(c) != row.end()) ? row[c] : 0.0;
		double sens = tp_c / total_c;
		sensitivity += sens / current_conf_matrix.size();

		double fp_c = 0;
		for (auto iter : current_conf_matrix){
			if (iter.first != conf_iter.first && iter.second.find(c)!=iter.second.end()){
				fp_c += iter.second[c];
			}
		}
		double spec = (testImages.size() - total_c-fp_c) / (testImages.size() - total_c);
		total_spec += spec / current_conf_matrix.size();

		double prec = ((fp_c + tp_c)>0)?tp_c / (fp_c + tp_c):0;
		precision += prec / current_conf_matrix.size();
	}

	total_sens += sensitivity;
	total_precision += precision;
	std::cout << "sensitivity=" << sensitivity << " precision=" << precision << endl;
}

void testRecognition(){
	MapOfFaces totalImages;
	loadFaces(totalImages);
	//calculateProbab(totalImages);

	vector<FaceImage*> faceImages;
    vector<FaceImage*> testImages;

	int bestInd=-1;
    float bestDist=100000,tmpDist;
	int errorsCount;
	//srand ( unsigned ( time (NULL) ) );
	
	std::ofstream res("errorRates.txt");
	const int TESTS=10;
	float totalTestsErrorRate=0, errorRateVar=0;
	float minError=100,maxError=0;
#ifdef USE_REJECT_OPTION
	float falseAcceptRate = 0, falseRejectRate = 0, rejectRate = 0;
	float other_db_falseAcceptRate = 0, other_db_falseRejectRate = 0, other_db_rejectRate = 0;
	
	float roFeature = 0;
	int probabCounter;

	MapOfFaces otherDbImages;
	loadImages("C:\\Users\\Andrey\\Documents\\images\\students\\db", "C:\\Users\\Andrey\\Documents\\images\\students\\test", 
		otherDbImages);
	vector<FaceImage*> otherTestImages;
	for (MapOfFaces::iterator iter = otherDbImages.begin(); iter != otherDbImages.end(); ++iter){
		int currentFaceCount = iter->second.size();
		copy(iter->second.begin(), iter->second.end(), back_inserter(otherTestImages));
	}
	cout << "other db size=" << otherTestImages.size() << endl;
#endif

	float total=0; 
	long testsCount=0;
	double total_sens = 0, total_spec = 0, total_precision=0;

	vector<float> errorRates(TESTS);
	int testSetSize=0;
	for (int testCount = 0; testCount < TESTS; ++testCount){
		errorsCount = 0;
		getTrainingAndTestImages(totalImages, faceImages, testImages);


		FacesDatabase db(faceImages);
		//vector<float> distances(faceImages.size());
		map<string, float> classDistances;
#ifdef USE_REJECT_OPTION
		int ro_statistics[RO_STATISTICS::EndRoIndex] = { 0, 0, 0 };
		int other_db_ro_statistics[RO_STATISTICS::EndRoIndex] = { 0, 0, 0 };

		vector<float> otherClassesRoFeatures;
		for (int i = 0; i < faceImages.size(); ++i){
			classDistances.clear();
			for (int j = 0; j < faceImages.size(); ++j){
				tmpDist = faceImages[i]->distance(faceImages[j]);
				bool classNameExists=classDistances.find(faceImages[j]->personName) != classDistances.end();
				if (faceImages[i]->personName == faceImages[j]->personName){
					if (tmpDist>=0.001 && !classNameExists || (classNameExists && classDistances[faceImages[j]->personName] > tmpDist))
						classDistances[faceImages[j]->personName] = tmpDist;
				}
				else
					if (!classNameExists || (classNameExists && classDistances[faceImages[j]->personName] > tmpDist))
						classDistances[faceImages[j]->personName] = tmpDist;
			}
#ifdef USE_ALL_DISTANCES
			multimap<float, string> distanceClasses = flip_map(classDistances);
			if (rejectType != RejectType::PosteriorProbab){
				float prevDist = -1;
				for (const auto &distanceClass : distanceClasses)
				{
					if (distanceClass.second != faceImages[i]->personName){
						roFeature = -1;
						if (rejectType == RejectType::MinDistance)
							roFeature = distanceClass.first;
						else if (prevDist >= 0){
							if (rejectType == RejectType::DiffDistance)
								roFeature = prevDist - distanceClass.first;
							else
								roFeature = prevDist / distanceClass.first;
						}
						if (roFeature != -1)
							otherClassesRoFeatures.push_back(roFeature);
						
						prevDist = distanceClass.first;
					}
				}
			}
			else{
				float sum = 0;
#if 1
				probabCounter = 0;
				for (const auto &distanceClass : distanceClasses){
					if (++probabCounter>probabModelsCount)
						break;
					//if (distanceClass.second != faceImages[i]->personName)
					sum += exp(-probabMultiplier*distanceClass.first);
				}
				probabCounter = 0;
				for (const auto &distanceClass : distanceClasses){
					if (++probabCounter>probabModelsCount)
						break;
					if (distanceClass.second != faceImages[i]->personName){
						roFeature = sum / exp(-probabMultiplier*distanceClass.first);
						otherClassesRoFeatures.push_back(roFeature);
					}
				}
#else
				for (const auto &classDistance : classDistances)
				{
					if (classDistance.first != faceImages[i]->personName)
						sum += exp(-probabMultiplier*classDistance.second);
				}
				for (const auto &classDistance : classDistances)
				{
					if (classDistance.first != faceImages[i]->personName){
						roFeature= sum/exp(-probabMultiplier*classDistance.second);
						otherClassesRoFeatures.push_back(roFeature);
					}
				}
#endif
			}
#else
			bestDist = FLT_MAX;
			string bestClass = "";
			for (const auto &classDistance : classDistances)
			{
				if (classDistance.first != faceImages[i]->personName && bestDist > classDistance.second){
					bestDist = classDistance.second;
					bestClass = classDistance.first;
				}
			}
			if (rejectType == RejectType::MinDistance){
				roFeature = bestDist;
			}
			else if (rejectType == RejectType::DiffDistance || rejectType == RejectType::DistanceRatio){
				float bestOtherDist = FLT_MAX;
				for (const auto &classDistance : classDistances)
				{
					if (classDistance.first != faceImages[i]->personName && classDistance.first != bestClass && bestOtherDist > classDistance.second)
						bestOtherDist = classDistance.second;
				}
				if (rejectType == RejectType::DiffDistance){
					roFeature = bestDist-bestOtherDist;
				}
				else
					roFeature = bestDist/bestOtherDist;
			}
			else if (rejectType == RejectType::PosteriorProbab){
				roFeature = 0;
#if 1
				multimap<float, string> distanceClasses = flip_map(classDistances);
				probabCounter = 0;
				for (const auto &distanceClass : distanceClasses){
					if (++probabCounter>probabModelsCount)
						break;
					if (distanceClass.second != faceImages[i]->personName){
						roFeature += exp(-probabMultiplier*(distanceClass.first - bestDist));
					}
				}
#else
				for (const auto &classDistance : classDistances)
				{
					if (classDistance.first != faceImages[i]->personName)
						roFeature += exp(-probabMultiplier*(classDistance.second - bestDist));
				}
#endif
				//roFeature = 1 / roFeature;
			}
			otherClassesRoFeatures.push_back(roFeature);
#endif
		}
#if 1
		float rate = fixedFAR;// (rejectType == RejectType::MinDistance) ? fixedFAR : 1 - fixedFAR;
		int ind = (int)(otherClassesRoFeatures.size()*rate);
		if (ind == 0 && otherClassesRoFeatures.size() > 1)
			ind = 1;
		std::nth_element(otherClassesRoFeatures.begin(), otherClassesRoFeatures.begin() + ind, otherClassesRoFeatures.end());
		float threshold = otherClassesRoFeatures[ind];
		cout<<"threshold="<<threshold<<endl;
#else
		float threshold = DirectedEnumeration::getThreshold(otherClassesRoFeatures, fixedFAR);
		//threshold = 0.8;
#endif
#endif

		testSetSize=testImages.size();
		processTestSet(db, faceImages, testImages, total, errorsCount, testsCount, total_sens, total_spec, total_precision
#ifdef USE_REJECT_OPTION
			,threshold, ro_statistics
#endif
			);
		errorRates[testCount]=100.*errorsCount/testImages.size();
		std::cout<<"test="<<testCount<<" error="<<errorRates[testCount]<<" dbSize="<<faceImages.size()<<
			" testSize=" << testImages.size()<<std::endl;
		res<<errorRates[testCount]<<std::endl;
		totalTestsErrorRate+=errorRates[testCount];
		errorRateVar+=errorRates[testCount]*errorRates[testCount];
		if(minError>errorRates[testCount])
			minError=errorRates[testCount];
		if(maxError<errorRates[testCount])
			maxError=errorRates[testCount];

#ifdef USE_REJECT_OPTION
		falseAcceptRate += 100.*ro_statistics[FalseAcceptRate] / testImages.size();
		falseRejectRate += 100.*ro_statistics[FalseRejectRate] / testImages.size();
		rejectRate += 100.*ro_statistics[RejectRate] / testImages.size();

		float total1 = 0;
		int errorsCount1 = 0;
		long testsCount1 = 0;
		processTestSet(db, faceImages, otherTestImages, total1, errorsCount1, testsCount1, 
			total_sens, total_spec, total_precision, threshold, other_db_ro_statistics);
		other_db_falseAcceptRate += 100.*other_db_ro_statistics[FalseAcceptRate] / otherTestImages.size();
		other_db_falseRejectRate += 100.*other_db_ro_statistics[FalseRejectRate] / otherTestImages.size();
		other_db_rejectRate += 100.*other_db_ro_statistics[RejectRate] / otherTestImages.size();

#endif
	}
	
	//QueryPerformanceCounter(&end); 
	float delta_microseconds = total / testsCount; 

	totalTestsErrorRate/=TESTS;
	errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));

	double range=1.96*sqrt(totalTestsErrorRate*(100-totalTestsErrorRate)/testSetSize); //5%
	double low=totalTestsErrorRate-range,high=totalTestsErrorRate+range;
	int outOfRange=0;
	for(int testCount=0;testCount<TESTS;++testCount){
		if(errorRates[testCount]<low || errorRates[testCount]>high)
			++outOfRange;
	}

	std::cout<<"Avg error="<<totalTestsErrorRate<<" Sigma="<<errorRateVar<<" delta="<<delta_microseconds<<
		" Min error=" << minError << " Max error=" << maxError<<
		" Range="<<range<<" Low="<<low<<" High="<<high<<" outOfRange="<<outOfRange<<std::endl;
	res<<"Avg error="<<totalTestsErrorRate<<" Sigma="<<errorRateVar<<" delta="<<delta_microseconds<<
		" Min error="<<minError<< " Max error="<<maxError<<" Range="<<range<<" Low="<<low<<" High="<<high<<" outOfRange="<<outOfRange<<std::endl;
	total_spec /= TESTS;
	total_sens /= TESTS;
	total_precision /= TESTS;
	std::cout << "total sens=" << 100 * total_sens << " total spec=" << 100 * total_spec << " total precision=" << 100 * total_precision<<'\n';

	res.close();

#ifdef USE_REJECT_OPTION
	falseAcceptRate /= TESTS;
	falseRejectRate /= TESTS;
	rejectRate /= TESTS;
	other_db_falseAcceptRate /= TESTS;
	other_db_falseRejectRate /= TESTS;
	other_db_rejectRate /= TESTS;
	std::cout << "FAR=" << falseAcceptRate << " FRR=" << falseRejectRate << " rejectRate=" << rejectRate << std::endl;
	std::cout << "other db FAR=" << other_db_falseAcceptRate << " FRR=" << other_db_falseRejectRate << " rejectRate=" << other_db_rejectRate << std::endl;
#endif

}
