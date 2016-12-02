#ifndef CAMERAGRABBER_H
#define CAMERAGRABBER_H

#include "Grabable.h"

#include <memory>
#include <opencv2/opencv.hpp>

class CvCapture;

class CameraGrabber : public Grabable
{
public:
    CameraGrabber(int cameraInd=0);
    void grab(QOpenCVWidget*);

    template <int cameraInd> static Grabable* Create(){
        return new CameraGrabber(cameraInd);
    }

private:

    class AbstractOpenCvCamera
    {
    public:
        virtual cv::VideoCapture& getCamera()=0;
        virtual ~AbstractOpenCvCamera(){}
    };

    template <int cameraInd> class OpenCvCamera: public AbstractOpenCvCamera
    {
    private:
        cv::VideoCapture camera;
    public:
        OpenCvCamera();
        ~OpenCvCamera();
        cv::VideoCapture& getCamera(){
            return camera;
        }
    };

    static std::auto_ptr<AbstractOpenCvCamera> cameras[];

    int currentCamera;
};

#endif // CAMERAGRABBER_H
