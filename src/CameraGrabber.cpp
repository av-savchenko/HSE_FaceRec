#include "CameraGrabber.h"
#include "QOpenCVWidget.h"
#include "GrabableFactory.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QtCore>

CameraGrabber::CameraGrabber(int cameraInd):currentCamera(cameraInd)
{
}
void CameraGrabber::grab(QOpenCVWidget* capturedVideo){
    cv::Mat image;
    cameras[currentCamera]->getCamera()>>image;
    if(image.cols>0 && image.rows>0)
        capturedVideo->putImage(image);
}

#define INIT_CAMERA(n) std::auto_ptr<CameraGrabber::AbstractOpenCvCamera>(new CameraGrabber::OpenCvCamera<(n)>())
std::auto_ptr<CameraGrabber::AbstractOpenCvCamera> CameraGrabber::cameras[]={
    INIT_CAMERA(0),
    INIT_CAMERA(1),
    INIT_CAMERA(2)
};
template <int cameraInd>
        CameraGrabber::OpenCvCamera<cameraInd>::OpenCvCamera():camera(cameraInd){
    if(!camera.isOpened()){  // check if we succeeded
        qDebug()<<"no camera: "<<cameraInd;
        return;
    }
    GrabableFactory::Instance().registerCreator(QString(QObject::tr("Camera %1")).arg(cameraInd),CameraGrabber::Create<cameraInd>);
}
template <int cameraInd>
        CameraGrabber::OpenCvCamera<cameraInd>::~OpenCvCamera(){
}
