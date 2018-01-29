#include "stdafx.h"

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

#define OPTIMAL

#ifdef OPTIMAL
#define EQUAL_SIGMA
#define PIVOT
#endif

template<typename T> volatile int DirectedEnumeration<T>::terminateSearch=0;

template<typename T> float DirectedEnumeration<T>::getThreshold(vector<float>& otherClassesDists, float falseAcceptRate)
{
	int ind=(int)(otherClassesDists.size()*falseAcceptRate);
    std::nth_element(otherClassesDists.begin(),otherClassesDists.begin()+ind,otherClassesDists.end());
    float threshold=otherClassesDists[ind];
    std::cout<<threshold<<" "<<*std::min_element(otherClassesDists.begin(),otherClassesDists.end())<<" "<<*std::max_element(otherClassesDists.begin(),otherClassesDists.end())<<std::endl;

	return threshold;
}
template<typename T> DirectedEnumeration<T>::DirectedEnumeration(vector<T>& faceImages, float falseAcceptRate/*=0.05*/, float threshold/*=0*/, int imageCountToCheck/*=0*/) :
dbImages(faceImages), dbSize(faceImages.size()), isP_matrix_allocated(true), pivots_only_used(false)
{
	init(imageCountToCheck);

	image_dists_matrix_size = dbSize;
    P_matrix=new ImageDist[dbSize*dbSize];

	vector<float> otherClassesDists;

    for(int i=0;i<dbSize;++i){
        for(int j=0;j<dbSize;++j){
			float dist = distance(dbImages[j], i, false);
            P_matrix[i*dbSize+j]=ImageDist(dist,j);
        }
        std::sort(&P_matrix[i*dbSize],&P_matrix[(i+1)*dbSize]);
        for(int idInd=0;idInd<dbSize;++idInd){
            ImageDist& id=P_matrix[i*dbSize+idInd];
			if (!areSameClasses(i, id.imageNum)){
                otherClassesDists.push_back(id.dist);
                break;
            }
        }
#ifdef OPTIMAL
	    std::sort(&P_matrix[i*dbSize],&P_matrix[(i+1)*dbSize],ImageDist::ComparerByNumber);
#endif
    }

    if(threshold>0){
        this->threshold=threshold;
    }
    else{
		this->threshold=getThreshold(otherClassesDists, falseAcceptRate);
    }
}
template<typename Cont, typename It>
auto SaveIndices(Cont &cont, It beg, It end) -> decltype(std::end(cont))
{
	int helpIndx(0);
	return std::remove_if(std::begin(cont), std::end(cont),
		[&](typename Cont::value_type const& val) -> bool {
		return std::find(beg, end, helpIndx++) == end;
	});
}

//unsupervised clastering (UC)
//#define CLUSTERING
static float variance_same_dist, variance_other_dist;
template<typename T> DirectedEnumeration<T>::DirectedEnumeration(vector<T>& faceImages, float thresh, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices) :
dbImages(faceImages), threshold(thresh), dbSize(faceImages.size()), isP_matrix_allocated(true), pivots_only_used(true), image_dists_matrix_size(faceImages.size())
{
#ifdef CLUSTERING
	float min_sum = numeric_limits<float>::max();
	int best_index = -1;
	int ind_count = dbSize;
	for (int i = 0; i < ind_count; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		float sum = 0;
		for (int j = 0; j < ind_count; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			sum += dist_matrix[iInd*dist_matrix_size + jInd];
		}
		if (sum < min_sum){
			min_sum = sum;
			best_index = i;
		}
	}
	set<int> indices_to_leave;
	indices_to_leave.insert(best_index);
	while (true){
		float worst_distance=-1;
		int worst_index = -1;
		for (int i = 0; i < ind_count; ++i){
			int iInd = indices != NULL ? (*indices)[i] : i;
			float best_distance = numeric_limits<float>::max();
			for (int j :indices_to_leave){
				int jInd = indices != NULL ? (*indices)[j] : j;
				if (best_distance> dist_matrix[iInd*dist_matrix_size + jInd]){
					best_distance = dist_matrix[iInd*dist_matrix_size + jInd];
				}
			}
			if (worst_distance < best_distance){
				worst_distance = best_distance;
				worst_index = i;
			}
		}
		if (worst_distance <= threshold)
			break;
		indices_to_leave.insert(worst_index);
	}
	std::cout << "in DEM indices_to_leave=" << indices_to_leave.size() << " dbSize=" << dbSize << '\n';
	other_objects.resize(indices_to_leave.size());
	for (int i = 0; i < ind_count; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		float best_distance = numeric_limits<float>::max();
		int best_ind = -1;
		int ind = 0;
		for (int j : indices_to_leave){
			int jInd = indices != NULL ? (*indices)[j] : j;
			if (best_distance> dist_matrix[iInd*dist_matrix_size + jInd]){
				best_distance = dist_matrix[iInd*dist_matrix_size + jInd];
				best_ind = ind;
			}
			++ind;
		}
		if (!areSameClasses(i, best_ind)){
			other_objects[best_ind].push_back(dbImages[i]);
		}
	}
	//convert indices
	vector<int> real_indices;
	int ind = 0;
	for (int j : indices_to_leave){
		real_indices.push_back(indices != NULL ? (*indices)[j] : j);
	}
	indices = &real_indices;

	db_size_ratio = 1.0f*indices_to_leave.size() / dbSize;
	dbImages.erase(SaveIndices(dbImages, std::begin(indices_to_leave), std::end(indices_to_leave)), dbImages.end());
	image_dists_matrix_size = dbSize = dbImages.size();
	std::cout << "dbSize=" << dbSize << '\n';

#endif

	auto t1 = std::chrono::high_resolution_clock::now();
	init(0);

#ifndef PIVOT
	pivots_only_used = false;
	P_matrix = new ImageDist[dbSize*dbSize];
	for (int i = 0; i < dbSize; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		for (int j = 0; j < dbSize; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			P_matrix[i*dbSize + j] = ImageDist(dist_matrix[iInd*dist_matrix_size + jInd], j);
		}
		
#ifndef OPTIMAL
		std::sort(&P_matrix[i*dbSize], &P_matrix[(i + 1)*dbSize]);
#endif

	}
#else //PIVOT
	P_matrix = new ImageDist[startIndices.size()*dbSize];
	dist_vars = new float[startIndices.size()*dbSize];
#if 1
	float min_other_dist = numeric_limits<float>::max();
	for (int ii = 0; ii < startIndices.size(); ++ii){
		int i = startIndices[ii];
		int iInd = indices != NULL ? (*indices)[i] : i;
		int mostFarModel = -1;
		double maxFarDist = 0;
		for (int j = 0; j < dbSize; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			P_matrix[ii*dbSize + j] = ImageDist(
				unpack_var(dist_matrix[iInd*dist_matrix_size + jInd])
				//distance(dbImages[j],i,false)
				,j);
			dist_vars[ii*dbSize + j] = unpack_dist(dist_matrix[iInd*dist_matrix_size + jInd]);
			if (!areSameClasses(i,j) && P_matrix[ii*dbSize + j].dist<min_other_dist){
				min_other_dist = P_matrix[ii*dbSize + j].dist;
			}
			double currentFarDist = 0;
			for (int ind = 0; ind <= ii; ++ind){
				if (startIndices[ind] == j)
					currentFarDist = -1000000;
				else
					currentFarDist += P_matrix[ind * dbSize + j].dist;
			}
			if (currentFarDist > maxFarDist){
				maxFarDist = currentFarDist;
				mostFarModel = j;
			}
		}
		if (ii < startIndices.size() - 1)
			//;
			 startIndices[ii + 1] = mostFarModel;
#ifndef OPTIMAL
		std::sort(&P_matrix[i*dbSize], &P_matrix[(i + 1)*dbSize]);
#endif
	}
#else //medoids
	vector<int> clusters(dbSize);
	std::fill(clusters.begin(), clusters.end(), -1);
	bool modified = true;
	while (modified){
		//maximization
		modified = false;
		for (int j = 0; j < dbSize; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			float min_dist = 10000000;
			int best_centroid = -1;
			for (int ii = 0; ii < startIndices.size(); ++ii){
				int i = startIndices[ii];
				int iInd = indices != NULL ? (*indices)[i] : i;
				if (dist_matrix[iInd*dist_matrix_size + jInd] < min_dist){
					min_dist = dist_matrix[iInd*dist_matrix_size + jInd];
					best_centroid = ii;
				}
			}
			if (clusters[j]!=best_centroid){
				modified = true;
				clusters[j] = best_centroid;
			}
		}

		//expectation
		for (int c = 0; c < startIndices.size(); ++c){
			int best_ind = -1;
			float min_sum = 100000;
			for (int j = 0; j < dbSize; ++j){
				if (clusters[j] == c){
					int jInd = indices != NULL ? (*indices)[j] : j;
					float sum = 0;
					int num_of_elements = 0;
					for (int j1 = 0; j1 < dbSize; ++j1){
						if (clusters[j1] == c){
							int j1Ind = indices != NULL ? (*indices)[j1] : j1;
							sum += dist_matrix[jInd*dist_matrix_size + j1Ind];
							++num_of_elements;
						}
					}
					if (num_of_elements>0)
						sum /= num_of_elements;
					if (sum < min_sum){
						min_sum = sum;
						best_ind = j;
					}
				}
			}
			startIndices[c]=best_ind;
		}
	}
	for (int ii = 0; ii < startIndices.size(); ++ii){
		int i = startIndices[ii];
		int iInd = indices != NULL ? (*indices)[i] : i;
		for (int j = 0; j < dbSize; ++j){
			int jInd = indices != NULL ? (*indices)[j] : j;
			P_matrix[ii*dbSize + j] = ImageDist(dist_matrix[iInd*dist_matrix_size + jInd], j);
		}
	}
#endif //medoids
	for (int c = 0; c < startIndices.size(); ++c)
		std::cout << startIndices[c] << ' ';
	std::cout << std::endl;

#if 0
	float avg_same_dist = 0, avg_other_dist = 0, var_same_dist = 0, var_other_dist = 0;
	int num_same = 0, num_other = 0;
	for (int i = 0; i < dbSize; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		for (int j = 0; j < dbSize; ++j){
			if (i == j)
				continue;
			int jInd = indices != NULL ? (*indices)[j] : j;
			if (areSameClasses(i, j)){
				avg_same_dist += dist_matrix[iInd*dist_matrix_size + jInd];
				var_same_dist += dist_matrix[iInd*dist_matrix_size + jInd] * dist_matrix[iInd*dist_matrix_size + jInd];
				++num_same;
			}
			else{
				avg_other_dist += dist_matrix[iInd*dist_matrix_size + jInd];
				var_other_dist += dist_matrix[iInd*dist_matrix_size + jInd] * dist_matrix[iInd*dist_matrix_size + jInd];
				++num_other;
			}
		}
	}
	std::wcout << "num_same=" << num_same<< " avg=" << avg_same_dist << " same_var=" << var_same_dist <<
		" num_other="<<num_other<<" avg=" << avg_other_dist << " other_var = " << var_other_dist << '\n';
	avg_same_dist /= num_same;
	variance_same_dist = (var_same_dist - avg_same_dist * avg_same_dist * num_same) / (num_same - 1);

	avg_other_dist /= num_other;
	variance_other_dist = (var_other_dist - avg_other_dist * avg_other_dist * num_other) / (num_other - 1);
	std::wcout << "avg=" << avg_same_dist << " same_var=" << variance_same_dist <<
		" avg=" << avg_other_dist << " other_var = " << variance_other_dist << '\n';

	variance_same_dist = variance_other_dist=0;
	for (int i = 0; i < dbSize; ++i){
		int iInd = indices != NULL ? (*indices)[i] : i;
		for (int j = 0; j < dbSize; ++j){
			if (i == j)
				continue;
			int jInd = indices != NULL ? (*indices)[j] : j;
			if (areSameClasses(i, j)){
				variance_same_dist += (dist_matrix[iInd*dist_matrix_size + jInd] - avg_same_dist) * 
					(dist_matrix[iInd*dist_matrix_size + jInd] - avg_same_dist);
			}
			else{
				variance_other_dist += (dist_matrix[iInd*dist_matrix_size + jInd] - avg_same_dist) *
					(dist_matrix[iInd*dist_matrix_size + jInd] - avg_same_dist);
			}
		}
	}
	variance_same_dist /= (num_same - 1);
	variance_other_dist /= (num_other - 1);
	std::wcout << "avg=" << avg_same_dist << " same_var=" << variance_same_dist <<
		" avg=" << avg_other_dist << " other_var = " << variance_other_dist << '\n';

	variance_same_dist /= variance_other_dist;
	variance_other_dist = 1;
	std::cout << "variance_same_dist=" << variance_same_dist << '\n';
#endif
#endif //PIVOT
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "init took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";
}
template<typename T> DirectedEnumeration<T>::DirectedEnumeration(vector<T>& faceImages, float thresh, ImageDist* image_dists, const int img_dists_matrix_size) :
dbImages(faceImages), threshold(thresh), dbSize(faceImages.size()), P_matrix(image_dists), isP_matrix_allocated(false), image_dists_matrix_size(img_dists_matrix_size), pivots_only_used(false)
{
	init(0);

#ifndef OPTIMAL
	for(int i=0;i<dbSize;++i){
	    std::sort(&P_matrix[i*image_dists_matrix_size],&P_matrix[(i+1)*image_dists_matrix_size]);
    }
#endif
}

template<typename T> DirectedEnumeration<T>::~DirectedEnumeration(){
	if(isP_matrix_allocated)
		delete[] P_matrix;
	delete[] dist_vars;
	
	delete[] likelihoods;
	delete[] likelihood_indices;
}
template<typename T> void DirectedEnumeration<T>::init(int imageCountToCheck){
	likelihoods=NULL;
	likelihood_indices = NULL;
#ifdef OPTIMAL
	likelihoods = new DISTANCE_RESULT_TYPE[dbSize];
	likelihood_indices = new int[dbSize];
#endif;
	states.resize(dbSize);

	setImageCountToCheck(imageCountToCheck);

	vector<int> indices(dbSize);
	for (int i = 0; i < dbSize; ++i)
		indices[i] = i;
	random_shuffle(indices.begin(), indices.end());
	//srand(time(0));
	int N =
#ifdef PIVOT
		(int)(dbSize*0.0175);//0.03 32;
	if(N<5)
		N=5;
	else if (N>32)
		N=32;
	//N=50;
#else
		1;
#endif
	startIndices.resize(N);
    for(int i=0;i<N;++i)
		startIndices[i] = indices[i];
}
template<typename T> void DirectedEnumeration<T>::setImageCountToCheck(int imageCountToCheck){
#ifdef CLUSTERING
	imageCountToCheck = (int)(db_size_ratio*imageCountToCheck);
#endif
	this->imageCountToCheck = (imageCountToCheck>0 && imageCountToCheck<dbSize) ? imageCountToCheck : dbSize;
}
template<typename T> bool DirectedEnumeration<T>::areSameClasses(int i, int j){
	return i == j;
}
template<typename T> float DirectedEnumeration<T>::distance(T& testImage, int modelInd, bool updateCounters)
{
	float res = testImage->distance(dbImages[modelInd]);
	if (updateCounters){
		++distanceCalcCount;
		distanceToTestCalculated(modelInd);
	}
	return res;
}
//static int neighborsAllocated=0;

class Neighbours {
public:
    Neighbours(int imgNum,float sourceDist):
            imageNum(imgNum),
			distance(sourceDist)
    {
    }

	Neighbours():
            imageNum(-1),
            distance(-1.0f)
			{}
    inline bool operator<(const Neighbours& rhs) const{
        return distance>rhs.distance;
    }

public:
    int imageNum;
    float distance;
};

enum DIR{LEFT,NOTHING,RIGHT};

const float EPS=0.0001f;

template<typename T> int DirectedEnumeration<T>::getImageListToCheck(int sourceImageNum, float distance)
{
    int imageNum=0,l,r;
    int i,j;
    int di,prevDi;
    DIR dir=NOTHING;
    int fdSize=dbSize;
    ImageDist* dists=&P_matrix[sourceImageNum*image_dists_matrix_size];
    l=0;
    r=fdSize;
    int m;
    while(l<r){
        m=(l+r)/2;
        if(distance<(dists[m].dist +EPS)){
            r=m;
        }else{
            l=m+1;
        }
    }
    i=r;
    if(i>=fdSize)
        i=fdSize-1;
    int neighborsCount=0;


    l=i-TRIAL_COUNT/2;
    r=i+TRIAL_COUNT/2;
    if(l<0){
        r-=l;
        if(r>=fdSize)
            r=fdSize-1;
        l=0;
    }
    else if(r>=fdSize){
        l-=(r-fdSize+1);
        if(l<0)
            l=0;
        r=fdSize-1;
    }
    int trialCount=r-l;
    prevDi=di=0;


    for(j=0;j<trialCount;++j){
        imageNum=dists[i+di].imageNum;
        //if(!dbImages[imageNum]->areNeighboursInvestigated())
        {
            neighboursHolder[neighborsCount++]=imageNum;
        }
        if(dir==LEFT)
            --di;
        else if (dir==RIGHT)
            ++di;
        else{
            prevDi=di;
            if((j&1)!=0)
                di=-((j+2)>>1);
            else
                di=(j+2)>>1;
            if((i+di)>r){
                di=prevDi-1;
                dir=LEFT;
            }
            else if((i+di)<l){
                di=prevDi+1;
                dir=RIGHT;
            }
        }
    }
    return neighborsCount;
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

#define CHECK_FOR_BEST_DIST_UPDATE_QUEUE                \
        CHECK_FOR_BEST_DIST	                            \
        neighbourSet.push(Neighbours(imageNum,tmpDist));

         /*
        im->neighboursInvestigated();           \
         neighbourSet.push(Neighbours(          \
            imageNum,getImageListToCheck(       \
            imageNum, tmpDist),tmpDist));
        */

//#include <deque>

template<typename T> T DirectedEnumeration<T>::recognize(T& testImage)
{
#ifdef OPTIMAL
	return optimalRecognize(testImage);
#else
    int i;
    int imageNum;
    int bestIndex=-1;
	int bestItemUpdated=1;
	int countOfLoopsWithNoUpdate=0;
    float tmpDist=0;
    //std::priority_queue<Neighbours, std::deque<Neighbours>> neighbourSet;
	std::vector <Neighbours> pq_vec;
	pq_vec.reserve(100);
	std::priority_queue<Neighbours> neighbourSet(std::less<Neighbours>(), pq_vec);
	
	Neighbours neighbours;
	int nCount;

	isFoundLessThreshold=false;
    bestDistance=10000000;
	distanceCalcCount=0;
    firstIndexToSeek=0;

	resetStates();

    for(i=0;i<startIndices.size();++i){
        imageNum=startIndices[i];
        CHECK_FOR_BEST_DIST_UPDATE_QUEUE
    }
	while(!terminateSearch && (distanceCalcCount<imageCountToCheck /*|| countOfLoopsWithNoUpdate<5*/)){
		bestItemUpdated=0;
        if(!neighbourSet.empty()){
            neighbours=neighbourSet.top();
            neighbourSet.pop();
			nCount=getImageListToCheck(neighbours.imageNum, neighbours.distance);
            for(int j=0;j<nCount;++j){
                imageNum=neighboursHolder[j];
                if(!isDistToTestCalculated(imageNum)){
                    CHECK_FOR_BEST_DIST_UPDATE_QUEUE
                }
            }

        }else{
			//generateRndNotcalced
			imageNum=0;
			int i,r;
			//r=qrand()%(dbSize-distanceCalcCount+1);
			for(i=firstIndexToSeek;i<dbSize;++i){
				if(!isDistToTestCalculated(i))// && --r==0)
				{
					firstIndexToSeek=i+1;
					imageNum= i;
					break;
				}
			}
            CHECK_FOR_BEST_DIST_UPDATE_QUEUE
        }
		if(bestItemUpdated)
			countOfLoopsWithNoUpdate=0;
		else
			++countOfLoopsWithNoUpdate;
    }

    end:
    //startIndices[0]=bestIndex;
    return bestIndex!=-1?dbImages[bestIndex]:NULL;
#endif
}

inline int isGreater(float* f1, float* f2)
{
  int i1, i2, t1, t2;

  i1 = *(int*)f1;
  i2 = *(int*)f2;

  t1 = i1 >> 31;
  i1 = (i1 ^ t1) + (t1 & 0x80000001);

  t2 = i2 >> 31;
  i2 = (i2 ^ t2) + (t2 & 0x80000001);

  return i1 > i2;
}
inline int FloatToIntToCompare(float* f){
  int t, i= *(int*)f;
  t = i >> 31;
  i = (i ^ t) + (t & 0x80000001);
  return i;
}
inline int isGeqZero(float* f1)
{
  int i1, i2, t1, t2;

  i1 = *(int*)f1;
  i2 = 0;

  t1 = i1 >> 31;
  i1 = (i1 ^ t1) + (t1 & 0x80000001);

  return i1 >= i2;
}

class LikelihoodsComparator:std::binary_function<int,int, bool>{
public:
	LikelihoodsComparator(DISTANCE_RESULT_TYPE* l) :
		likelihoods(l)
	{
	}
	bool operator() (int lhsIndex, int rhsIndex){
		return likelihoods[lhsIndex]<likelihoods[rhsIndex];
	}

private:
	DISTANCE_RESULT_TYPE* likelihoods;
};
template<typename T> T DirectedEnumeration<T>::optimalRecognize(T& testImage)
{
	int i,nu,k;
    int imageNum,bestImageNum;
    int bestIndex=-1;
	int bestItemUpdated=1;
	int countOfLoopsWithNoUpdate=0;
	float tmpDist = 0;
	DISTANCE_RESULT_TYPE modelsDist, tmp, tmp_dist, dist_var;
	DISTANCE_RESULT_TYPE bestLikelihood;
	ImageDist* P_row;
	float* dist_var_row;
	
	Neighbours neighbours;
	int nCount;
	//threshold = 0;

	isFoundLessThreshold=false;
    bestDistance=FLT_MAX;
	distanceCalcCount=0;
    firstIndexToSeek=0;

	resetStates();
    for(i=0;i<dbSize;++i){
		likelihoods[i]=0;
		likelihood_indices[i] = i;
    }
	int start_index = 0;
	LikelihoodsComparator likelihoodsComparator(likelihoods);
	int tmp_ind;

	bestImageNum=-1;
	bestLikelihood=FLT_MAX;
	for (i = 0; i<startIndices.size(); ++i){
        imageNum=startIndices[i];
		CHECK_FOR_BEST_DIST
			tmp_dist = TO_DISTANCE_RESULT_TYPE(tmpDist);
		likelihood_indices[imageNum] = likelihood_indices[start_index];
		likelihood_indices[start_index++] = imageNum;

		int i_ind = (pivots_only_used ? i : imageNum);
		P_row = P_matrix + image_dists_matrix_size*i_ind;
		dist_var_row = dist_vars + image_dists_matrix_size*i_ind;
		for (int ii = start_index; ii<dbSize; ++ii){
			nu = likelihood_indices[ii];
			modelsDist=P_row[nu].dist;
			if (modelsDist >= 0){
				tmp = tmp_dist - modelsDist;
#ifdef EQUAL_SIGMA
				likelihoods[nu] += tmp*tmp;
#else
#if 1
				dist_var = modelsDist;
#elif 1
				dist_var = areSameClasses(i_ind, nu) ? variance_same_dist : 1;
#else
				dist_var = dist_var_row[nu];
#endif
				likelihoods[nu] += tmp*tmp/ dist_var;//+log(dist_var)/2;
#endif
			}
		}
    }
	{
		int TRIALS =
#ifndef PIVOT
			2;
#else
			dbSize - start_index;
		std::sort(likelihood_indices + start_index, likelihood_indices + dbSize, likelihoodsComparator);
#endif
		while (!terminateSearch && (distanceCalcCount < imageCountToCheck)/*|| countOfLoopsWithNoUpdate<100) && distanceCalcCount <dbSize*/){
			bestItemUpdated = 0;
			if ((start_index + TRIALS) >= dbSize){
				imageNum = likelihood_indices[start_index++];
				CHECK_FOR_BEST_DIST
			}
			else{
				std::partial_sort(likelihood_indices + start_index, likelihood_indices + start_index + TRIALS,
					likelihood_indices + dbSize, likelihoodsComparator);
				for (int i = 0; i < TRIALS; ++i){
					bestImageNum = imageNum = likelihood_indices[start_index++];
					CHECK_FOR_BEST_DIST
					tmp_dist = TO_DISTANCE_RESULT_TYPE(tmpDist);
					if (bestItemUpdated)
						countOfLoopsWithNoUpdate = 0;
					else
						++countOfLoopsWithNoUpdate;

					P_row = P_matrix + imageNum*image_dists_matrix_size;
					for (int ii = start_index; ii < dbSize; ++ii){
						nu = likelihood_indices[ii];
						modelsDist = P_row[nu].dist;
						if (modelsDist >= 0){
							tmp = tmp_dist - modelsDist;
#ifdef EQUAL_SIGMA
							likelihoods[nu] += tmp*tmp;
#else
							likelihoods[nu] += tmp*tmp / modelsDist;//+log(modelsDist)/2;
#endif
						}
					}
				}
			}
		}
	}
    end:
	//startIndices[0]=bestIndex;
	T res = dbImages[bestIndex];
#ifdef CLUSTERING
	for (T& other_object : other_objects[bestIndex]){
		tmpDist = testImage->distance(other_object);
		if (tmpDist < bestDistance){
			bestDistance = tmpDist;
			//bestIndex = other_ind;
			res = other_object;
		}
	}
#endif
    return res;
}


//faces
#include "FaceImage.h"
template<> bool DirectedEnumeration<FaceImage*>::areSameClasses(int i, int j){
	return dbImages[i]->personName == dbImages[j]->personName;
}
template class DirectedEnumeration<FaceImage*>;
