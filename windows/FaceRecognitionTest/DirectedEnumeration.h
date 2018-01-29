#ifndef DirectedEnumeration_H
#define DirectedEnumeration_H

#include <vector>
#include <list>

#if 0
#define TO_DISTANCE_RESULT_TYPE(x) ((int)((x)*(1<<16)))
typedef int DISTANCE_RESULT_TYPE;
#else
#define TO_DISTANCE_RESULT_TYPE(x) ((x))
typedef float DISTANCE_RESULT_TYPE;
#endif

class ImageDist{
public:
	DISTANCE_RESULT_TYPE dist;
	int imageNum;

	ImageDist(float d = -1, int iNum = 0) :
		dist(TO_DISTANCE_RESULT_TYPE(d)), imageNum(iNum)
	{}

	bool operator<(const ImageDist& rhs) const{
		return dist<rhs.dist;
	}

	static inline bool ComparerByNumber(const ImageDist& lhs, const ImageDist& rhs){
		return lhs.imageNum<rhs.imageNum;
	}
};
template <typename T> class DirectedEnumeration
{

public:
    DirectedEnumeration(std::vector<T>& faceImages,float falseAcceptRate=0.05f,float threshold=0, int imageCountToCheck=0);
	DirectedEnumeration(std::vector<T>& faceImages, float threshold, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices);
	DirectedEnumeration(std::vector<T>& faceImages, float threshold, ImageDist* image_dists, const int image_dists_matrix_size);
    ~DirectedEnumeration();

    static const int TRIAL_COUNT=1+
		16;
		//64;
		//80;
	
    static const int MAX_ROW_SIZE=2;

	static float getThreshold(std::vector<float>& otherClassesDists, float falseAcceptRate);

    T recognize(T& testImage);

    float getCheckedPercent(){
        return 100.*distanceCalcCount/dbSize;
    }
	void setImageCountToCheck(int imageCountToCheck);
	
	bool isFoundLessThreshold;
	float bestDistance;

	static volatile int terminateSearch;
    
	
private:
	void init(int imageCountToCheck);

	std::vector<T> dbImages;
	std::vector<std::list<T> > other_objects;
	float db_size_ratio;

    int dbSize;
    float threshold;
    int firstIndexToSeek;
	int imageCountToCheck;

    ImageDist* P_matrix;
	float* dist_vars;
	bool isP_matrix_allocated;
	bool pivots_only_used;
	int image_dists_matrix_size;

    std::vector<int> startIndices;
    int neighboursHolder[TRIAL_COUNT+1];

    int distanceCalcCount;

	float distance(T& testImage, int modelInd, bool updateCounters = true);
	bool areSameClasses(int i, int j);
    int getImageListToCheck(int sourceImageNum,float distance);

	T optimalRecognize(T& testImage);
	DISTANCE_RESULT_TYPE* likelihoods;
	int* likelihood_indices;


	//states
	std::vector<int> states;
	static const int DIST_TO_TEST_CALCULATED = 1;
	static const int NEIGHBOURS_INVESTIGATED = 2;
	void resetStates(){
		std::fill(states.begin(), states.end(), 0);
	}
	bool isDistToTestCalculated(int ind){
		return (states[ind]&DIST_TO_TEST_CALCULATED) != 0;
	}
	void distanceToTestCalculated(int ind){
		states[ind] |= DIST_TO_TEST_CALCULATED;
	}
	bool areNeighboursInvestigated(int ind){
		return (states[ind] & NEIGHBOURS_INVESTIGATED) != 0;
	}
	void neighboursInvestigated(int ind){
		states[ind] |= NEIGHBOURS_INVESTIGATED;
	}

};

#endif // DirectedEnumeration_H
