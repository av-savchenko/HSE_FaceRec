#ifndef SINGLEIMAGEGRABBER_H
#define SINGLEIMAGEGRABBER_H

#include <QImage>

#include "Grabable.h"
#include "QOpenCVWidget.h"

class SingleImageGrabber: public Grabable
{
private:
    QImage image;
    void processImage();
public:
    SingleImageGrabber();
    void grab(QOpenCVWidget*);
};

#endif // SINGLEIMAGEGRABBER_H
