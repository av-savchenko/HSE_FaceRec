#include "stdafx.h"

#include "SmallWorld.h"
#include "DirectedEnumeration.h"
#include "distance_pack.h"

#include <algorithm>
#include <queue>
#include <vector>
#include <cmath>
#include <ctime>
#include <climits>
#include <cfloat>
#include <iostream>
#include <chrono>
#include <set>
using namespace std;

template<typename T> volatile int SmallWorld<T>::terminateSearch=0;

template<typename T> SmallWorld<T>::SmallWorld(){
}
template<typename T> SmallWorld<T>::SmallWorld(vector<T>& faceImages, float thresh, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices) :
dbImages(faceImages), threshold(thresh), dbSize(faceImages.size()), image_dists_matrix_size(faceImages.size()), distances(faceImages.size()), neighbours(faceImages.size())
{
	setImageCountToCheck(0);
#if 1
	vector<ImageDist> dists(dbSize);
	for (int i = 0; i < dbSize; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		for (int j = 0; j < dbSize; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			dists[j] = ImageDist(unpack_var(dist_matrix[iInd*dist_matrix_size + jInd]), j);
		}
		std::sort(dists.begin(),dists.end());
		neighbours[i].resize(NUM_OF_NEIGHBOURS);
		for (int j = 0; j < NUM_OF_NEIGHBOURS; ++j)
			neighbours[i][j] = dists[j+1].imageNum;
	}
#endif
}


template<typename T> SmallWorld<T>::~SmallWorld(){
}
template<typename T> void SmallWorld<T>::setImageCountToCheck(int imageCountToCheck){
	this->imageCountToCheck = (imageCountToCheck>0 && imageCountToCheck<dbSize) ? imageCountToCheck : dbSize;
}
template<typename T> bool SmallWorld<T>::areSameClasses(int i, int j){
	return i == j;
}
template<typename T> float SmallWorld<T>::distance(T& testImage, int modelInd, bool updateCounters)
{
	if (distances[modelInd] > -0.001)
		return distances[modelInd];

	float res = testImage->distance(dbImages[modelInd]);
	if (updateCounters){
		++distanceCalcCount;
		distances[modelInd]=res;
	}
	return res;
}

#define CHECK_FOR_BEST_DIST                            \
		tmpDist = distance(testImage, imageNum);        \
        if(tmpDist<bestDistance){                       \
			bestItemUpdated=1;		                    \
            bestDistance=tmpDist;                       \
            bestIndex=imageNum;                         \
            if(bestDistance<threshold){                 \
                isFoundLessThreshold=true;              \
                goto end;                               \
	        }                                           \
        }                                               

template<typename T> T SmallWorld<T>::recognize(T& testImage)
{

	int imageNum, bestImageNum;
	int bestIndex = -1;
	int bestItemUpdated = 0;
	float tmpDist = 0;

	isFoundLessThreshold = false;
	bestDistance = FLT_MAX;
	distanceCalcCount = 0;
	firstIndexToSeek = 0;

	for (int i = 0; i<dbSize; ++i){
		distances[i] = -1;
	}
	imageNum = 0;
	bestDistance = distance(testImage, imageNum);
	bestIndex = imageNum;
	if (bestDistance<threshold){
		goto end;
	}
#if 1
	int n = 0;
	float curDist = bestDistance, curBestDist;
	int bestInd = bestIndex,curBestInd;
	while (!terminateSearch && (distanceCalcCount<imageCountToCheck /*|| countOfLoopsWithNoUpdate<5*/)){
		++n;
		curBestDist = curDist;
		curBestInd=-1;
		for (int j = 0; j < NUM_OF_NEIGHBOURS; ++j){
			imageNum = neighbours[bestInd][j];
			if (distances[imageNum] < 0){
				tmpDist = distance(testImage, imageNum);
			}
			if (distances[imageNum]<curBestDist){
				curBestDist = distances[imageNum];
				curBestInd = imageNum;

				if (curBestDist < bestDistance){
					bestDistance = curBestDist;
					bestIndex = imageNum;
					if (bestDistance < threshold){
						goto end;
					}
				}
			}
		}
		if (curBestInd==-1){
			for (imageNum = 0; imageNum < dbSize && distances[imageNum] > -0.001; ++imageNum);
			curDist = distance(testImage, imageNum);
			bestInd = imageNum;
			if (curDist<bestDistance){
				bestDistance = curDist;
				bestIndex = imageNum;
				if (bestDistance<threshold){
					goto end;
				}
			}

		}
		else{
			bestInd = curBestInd;
			curDist=curBestDist;
		}
    }
#endif
end:
	if (bestDistance<threshold){
		isFoundLessThreshold = true;
	}
    return bestIndex!=-1?dbImages[bestIndex]:NULL;
}

//faces
#include "FaceImage.h"
template<> bool SmallWorld<FaceImage*>::areSameClasses(int i, int j){
	return dbImages[i]->personName == dbImages[j]->personName;
}
template class SmallWorld<FaceImage*>;
