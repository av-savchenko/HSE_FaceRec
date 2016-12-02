#include "FacesData.h"


#include<set>
#include<string>
using namespace std;

#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/flann/matrix.h>
using namespace cv;

#define CURRENT_MATCHER KDTreeIndexParams()
//#define CURRENT_MATCHER CompositeIndexParams()
//#define CURRENT_MATCHER FLANNIndexParams(8, 30, 2)
#define INIT_MATCHER ,FLANNMatcher (new flann::CURRENT_MATCHER ,new flann::SearchParams(125*8/NUM_OF_THREADS))


NNMethod FacesData::nnMethod=Simple;

FacesData::FacesData(vector<FaceImage*>& faces):
                faceImages(faces)
#ifdef NEED_DEM
                , directedEnumeration(faces, threshold, dist_matrix, dist_matrix_size, indices)
#endif
#ifdef NEED_FLANN
        INIT_MATCHER
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
}

FaceImage* FacesData::recognize(FaceImage* test)
{
    setTestImage(test);
    closestImage=0;
    closestDistance=1000000;
    operator()();
    return closestImage;
}

void FacesData::operator() ()
{
        switch(FacesData::nnMethod){
        case Simple:
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
        case DEM:
                //if(ind>=0)
                {
                        int ind=directedEnumeration.recognize(testImage);
                        avgCheckedPercent+=directedEnumeration.getCheckedPercent();
                        //std::cout<<directedEnumeration.getCheckedPercent()<<"\n";
                        closestImage=faceImages[ind];
                        closestDistance=directedEnumeration.bestDistance;
                        if(directedEnumeration.isFoundLessThreshold)
                                DirectedEnumeration::terminateSearch=1;
                }
                break;
#endif
#ifdef NEED_KD_TREE
        case KD_TREE:
                {
                        const int NUM_OF_NEIGHBORS=1;
                        const int searchSize = 1;
                        int matches_data[NUM_OF_NEIGHBORS];
                        double distance_data[NUM_OF_NEIGHBORS];
                        CvMat matches = cvMat(searchSize, NUM_OF_NEIGHBORS, CV_32SC1, matches_data);
                        CvMat distance = cvMat(searchSize, NUM_OF_NEIGHBORS, CV_64FC1, distance_data);

                        const float* search_features=testImage->getFeatures();
                        CvMat search_features_mat = cvMat(1, FaceImage::FEATURES_COUNT, CV_32FC1, (void*)search_features);
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
#ifdef NEED_FLANN
        case FLANN:
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
                        cvflann::Matrix<float> query((float*)testImage->getFeatures(), 1,FaceImage::FEATURES_COUNT);
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
        }
}


void FacesData::initMatchers(){

        dictionary_features=NULL;
#if defined(NEED_KD_TREE) || defined (NEED_FLANN)
    dictionary_features=new float[faceImages.size()*FaceImage::FEATURES_COUNT];
    for(int j=0;j<faceImages.size();++j){
        for(int k=0;k<FaceImage::FEATURES_COUNT;++k)
            dictionary_features[j*FaceImage::FEATURES_COUNT+k]=faceImages[j]->getFeatures()[k];
    }
#endif
#ifdef NEED_KD_TREE
        // Build the k-d tree
    dictionary_features_mat = cvMat(faceImages.size(), FaceImage::FEATURES_COUNT, CV_32FC1, dictionary_features);
    dictionary_tree =
            cvCreateKDTree
            //cvCreateSpillTree
                                     (&dictionary_features_mat);
#endif

#ifdef NEED_FLANN
#if 0
        vector<Mat> descriptors (faceImages.size());
    for(int j=0;j<faceImages.size();++j){
                descriptors[j]=Mat(1,FaceImage::FEATURES_COUNT, CV_32FC1, &dictionary_features[j*FaceImage::FEATURES_COUNT]);
    }
        FLANNMatcher.add(descriptors);
        FLANNMatcher.train();
#else
        cvflann::Matrix<float> samplesMatrix((float*)dictionary_features, faceImages.size(), FaceImage::FEATURES_COUNT);
    //Index<cvflann::ChiSquareDistance<float>> flann_index(samplesMatrix, cvflann::LinearIndexParams());
    flann_index=new cvflann::Index<cvflann::L1<float> >(samplesMatrix, cvflann::KDTreeIndexParams(4));
    flann_index->buildIndex();
#endif
#endif
        }
