#ifndef __FACE_DATA__
#define __FACE_DATA__

#include <windows.h>
#include <vector>

#include <opencv/cv.h>

#include "FaceImage.h"



//#define NEED_DEM
#define NEED_KD_TREE
#define NEED_FLANN


enum NNMethod{
        Simple
#ifdef NEED_DEM
        , DEM
#endif
#ifdef NEED_KD_TREE
        , KD_TREE
#endif
#ifdef NEED_FLANN
        ,FLANN
#endif
};

#define NUM_OF_THREADS 1

class FacesData{
public:

        static NNMethod nnMethod;

        FacesData(std::vector<FaceImage*>& faceImages);
        ~FacesData();

        void setTestImage(FaceImage* test)
        {
                closestImage=0;
                closestDistance=1000000;
                testImage=test;
        }
        void operator() ();

        FaceImage* recognize(FaceImage* test);
        void setImageCountToCheck(int imageCountToCheck);


        void init(){
                avgCheckedPercent=0;
        }

private:
        FaceImage* closestImage;
        int avgCheckedPercent;

        void initMatchers();

        std::vector<FaceImage*> faceImages;
        FaceImage* testImage;
        float closestDistance;

        float* dictionary_features;
        int imageCountToCheck;
#ifdef NEED_KD_TREE
        CvMat dictionary_features_mat;
        CvFeatureTree *dictionary_tree;

#endif
#ifdef NEED_DEM
        DirectedEnumeration directedEnumeration;
#endif

#ifdef NEED_FLANN
        cv::FlannBasedMatcher FLANNMatcher;
        cvflann::Index<cvflann::L1<float> > *flann_index;
#endif

};


#endif //__FACE_DATA__
