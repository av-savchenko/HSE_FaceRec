#ifndef FACEIMAGE_H
#define FACEIMAGE_H

class QImage;
#include <QString>
#include <QVector>

#include <opencv2/core.hpp>

#define COLORS_COUNT 1

#define USE_DNN_FEATURES

#ifndef USE_DNN_FEATURES
//#define USE_KL
#define USE_GRADIENT_ANGLE
#endif

class FaceImage
{
public:
    FaceImage(const QImage* image,const QString& personName="",const QString& fileName="",bool isUnrecAdded=false);
    FaceImage(cv::Mat& image,const QString& personName="");

    float distance(const FaceImage* rhs);

    static const QImage* get_center(const QVector<const QImage*>& images);

    QString personName;
    QString fileName;
    bool isUnrec;

    static const int POINTS_IN_W=10;//5;
    static const int POINTS_IN_H=10;//7;
    static const int HISTO_COUNT=POINTS_IN_W*POINTS_IN_H;

#ifdef USE_DNN_FEATURES
    static const int FEATURES_COUNT=256;
    std::vector<float> featureVector;
    const float* getFeatures()const{
        return &featureVector[0];
    }

#else
#if defined (USE_GRADIENT_ANGLE)
    static const int HISTO_SIZE=8;
    static const int FEATURES_COUNT=COLORS_COUNT*POINTS_IN_W*POINTS_IN_H*HISTO_SIZE;
#else
    static const int NUM_OF_PIXELS_IN_ONE_SLOT=8;
    static const int HISTO_SIZE=256/NUM_OF_PIXELS_IN_ONE_SLOT;
    static const int FEATURES_COUNT=COLORS_COUNT*HISTO_COUNT*HISTO_SIZE;
#endif
    const float* getFeatures()const{
        return &histos[0][0][0][0];
    }

#endif

private:

    void init(cv::Mat& image ,int *pixels, int width, int height);

#ifndef USE_DNN_FEATURES
#ifdef USE_GRADIENT_ANGLE
    float histos[COLORS_COUNT][POINTS_IN_H][POINTS_IN_W][HISTO_SIZE];
#else
    float histos[COLORS_COUNT][HISTO_COUNT][HISTO_SIZE];
#endif
    void shift(float* histo, float* resHisto, int shiftInd);
#endif

 };

#endif // FACEIMAGE_H
