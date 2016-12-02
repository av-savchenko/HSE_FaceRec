#include "QOpenCVWidget.h"
#include "OpenCvWrapper.h"

#include <QtCore>
#include <QPainter>
#include <QMessageBox>

// Constructor
QOpenCVWidget::QOpenCVWidget(QWidget *parent) : QWidget(parent) {
    layout = new QVBoxLayout;
    imagelabel = new QLabel;
    QImage dummy(100,100,QImage::Format_RGB32);
    image = dummy;
    layout->addWidget(imagelabel);

    setLayout(layout);
}

QOpenCVWidget::~QOpenCVWidget(void) {
    
}

void QOpenCVWidget::putImage(const QImage& srcImage, bool useAllImageIfNoFaceDetected/*=false*/) {
    image=srcImage;
    faceRects=OpenCvWrapper::Instance().detectFaceRects(&srcImage);
    if(faceRects.isEmpty() && useAllImageIfNoFaceDetected){
        faceRects.push_back(QRect(0,0,srcImage.width(),srcImage.height()));
    }
    faces=OpenCvWrapper::Instance().detectFaces(faceRects,image);
}
void QOpenCVWidget::putImage(cv::Mat&/*IplImage * */cvimage) {
    /*int cvIndex, cvLineStart;
    // switch between bit depths
    switch (cvimage->depth) {
        case IPL_DEPTH_8U:
            switch (cvimage->nChannels) {
                case 3:
                    if ( (cvimage->width != image.width()) || (cvimage->height != image.height()) ) {
                        QImage temp(cvimage->width, cvimage->height, QImage::Format_RGB32);
                        image = temp;
                    }
                    cvIndex = 0; cvLineStart = 0;
                    for (int y = 0; y < cvimage->height; y++) {
                        unsigned char red,green,blue;
                        cvIndex = cvLineStart;
                        for (int x = 0; x < cvimage->width; x++) {
                            // DO it
                            red = cvimage->imageData[cvIndex+2];
                            green = cvimage->imageData[cvIndex+1];
                            blue = cvimage->imageData[cvIndex+0];
                            
                            image.setPixel(x,y,qRgb(red, green, blue));
                            cvIndex += 3;
                        }
                        cvLineStart += cvimage->widthStep;                        
                    }
                    break;
                default:
                    QMessageBox::warning(0,tr("Warning"),tr("This number of channels is not supported"));
                    break;
            }
            break;
        default:
            QMessageBox::warning(0,tr("Warning"),tr("This type of IplImage is not implemented in QOpenCVWidget"));
            break;
    }
*/
    cv::Mat temp; // make the same cv::Mat
    cvtColor(cvimage, temp,CV_BGR2RGB); // cvtColor Makes a copt, that what i need
    QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    image=dest.copy();
    faceRects=OpenCvWrapper::Instance().detectFaceRects(cvimage);
    faces=OpenCvWrapper::Instance().detectFaces(faceRects,image);
}
void QOpenCVWidget::paintFaces() {
    QPainter painter;
    painter.begin(&image);
    painter.setPen(QPen(Qt::green,8));
    for(QVector<QRect>::const_iterator iter=faceRects.begin();iter!=faceRects.end();++iter){
        painter.drawEllipse((*iter).x(),(*iter).y(),(*iter).width(),(*iter).height());
    }
    painter.end();

    int w = imagelabel->width();
    int h = imagelabel->height();
    //qDebug()<<image.width()<<' '<<image.height();
    imagelabel->setPixmap(QPixmap::fromImage(image).scaled(w,h));
}

