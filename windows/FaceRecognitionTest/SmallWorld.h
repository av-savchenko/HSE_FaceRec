#ifndef __SMALL_WORLD_H__
#define __SMALL_WORLD_H__

#include <vector>
#include <list>

template <typename T> class SmallWorld
{
public:
	SmallWorld();
	//SmallWorld(std::vector<T>& faceImages, float falseAcceptRate = 0.05f, float threshold = 0, int imageCountToCheck = 0);
	SmallWorld(std::vector<T>& faceImages, float threshold, const float* dist_matrix, const int dist_matrix_size, std::vector<int>* indices);
	~SmallWorld();

    static const int NUM_OF_NEIGHBOURS=128;

    T recognize(T& testImage);

    float getCheckedPercent(){
        return 100.*distanceCalcCount/dbSize;
    }
	void setImageCountToCheck(int imageCountToCheck);
	
	bool isFoundLessThreshold;
	float bestDistance;

	static volatile int terminateSearch;    
	
private:
	std::vector<T> dbImages;
	std::vector<float> distances;
	std::vector<std::vector<int> > neighbours;
	float db_size_ratio;

    int dbSize;
    float threshold;
    int firstIndexToSeek;
	int imageCountToCheck;
	int image_dists_matrix_size;

    int distanceCalcCount;

	float distance(T& testImage, int modelInd, bool updateCounters = true);
	bool areSameClasses(int i, int j);
};


#endif // __SMALL_WORLD_H__
