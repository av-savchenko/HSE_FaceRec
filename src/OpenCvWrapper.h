#ifndef OPENCVWRAPPER_H
#define OPENCVWRAPPER_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <QImage>
#include <QRect>
#include <QVector>

class OpenCvWrapper
{
private:
    cv::CascadeClassifier  cascade;
    //CvMemStorage* storage;

    OpenCvWrapper();
    OpenCvWrapper(const OpenCvWrapper&);
    OpenCvWrapper& operator=(const OpenCvWrapper&);

public:
    static OpenCvWrapper& Instance();
    QVector<QRect> detectFaceRects(const cv::Mat&/*IplImage**/, bool oneFace=false);
    QVector<QRect> detectFaceRects(const QImage*, bool oneFace=false);

    QVector<std::pair<QRect,QImage> > detectFaces(const QVector<QRect>& faceRects, const QImage& image);

    enum FilterType{
        None, Noise, Smooth, Erode, Dilate, Morphology, Threshold, Sharpness, Brightness, Blackout, Sobel, Laplace
    };

    QImage putFilter(QImage* srcImage, FilterType filterType, int level=1);
};

#endif // OPENCVWRAPPER_H
