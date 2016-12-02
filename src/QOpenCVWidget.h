
#ifndef QOPENCVWIDGET_H
#define QOPENCVWIDGET_H

#include <opencv2/core.hpp>

#include <QPixmap>
#include <QLabel>
#include <QWidget>
#include <QVBoxLayout>
#include <QImage>
#include <QVector>


class QOpenCVWidget : public QWidget {
    Q_OBJECT
    private:
        QLabel *imagelabel;
        QVBoxLayout *layout;
        
        QImage image;
        QVector<std::pair<QRect,QImage> > faces;
        QVector<QRect> faceRects;

    public:
        QOpenCVWidget(QWidget *parent = 0);
        ~QOpenCVWidget(void);

        QVector<std::pair<QRect,QImage> > getFaces(){
            return faces;
        }

        void putImage(const QImage&, bool useAllImageIfNoFaceDetected=false);
        void putImage(cv::Mat&/*IplImage* */);
        void paintFaces();
};

#endif
