/* ======================================================================
**  MediaPlayer.cpp
** ======================================================================
**
** ======================================================================
**  Copyright (c) 2009 by Max Schlee
** ======================================================================
*/

#include "MediaPlayer.h"

#include <QtGui>
#include <QtWidgets>
#include <qvideosurfaceformat.h>
#include <qvideowidget.h>
#include <QAbstractVideoSurface>
#include <QList>
#include <QImage>

class FrameGrabberVideoSurface : public QAbstractVideoSurface
{
public:
    QList<QVideoFrame::PixelFormat> supportedPixelFormats(
            QAbstractVideoBuffer::HandleType handleType = QAbstractVideoBuffer::NoHandle) const;

    bool present(const QVideoFrame &frame);

    QImage currentFrame;
};

QList<QVideoFrame::PixelFormat> FrameGrabberVideoSurface::supportedPixelFormats(
        QAbstractVideoBuffer::HandleType handleType) const
{
    if (handleType == QAbstractVideoBuffer::NoHandle) {
         return QList<QVideoFrame::PixelFormat>()
                 << QVideoFrame::Format_RGB32
                 << QVideoFrame::Format_ARGB32
                 << QVideoFrame::Format_ARGB32_Premultiplied
                 << QVideoFrame::Format_RGB565
                 << QVideoFrame::Format_RGB555;
     } else {
         return QList<QVideoFrame::PixelFormat>();
     }
}

bool FrameGrabberVideoSurface::present(const QVideoFrame &frame)
{
    if (frame.isValid()) {
        QVideoFrame cloneFrame(frame);
        cloneFrame.map(QAbstractVideoBuffer::ReadOnly);
        const QImage image(cloneFrame.bits(),
                           cloneFrame.width(),
                           cloneFrame.height(),
                           QVideoFrame::imageFormatFromPixelFormat(cloneFrame.pixelFormat()));
        currentFrame=image.copy();
        cloneFrame.unmap();
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------
MediaPlayer::MediaPlayer(QWidget* pwgt/*=0*/) : QWidget(pwgt)
{
    mediaPlayer=new QMediaPlayer(0, QMediaPlayer::VideoSurface);

    frameGrabber=new FrameGrabberVideoSurface();
    mediaPlayer->setVideoOutput(frameGrabber);
    slotLoad();
    return;

    QPushButton*          pcmdPlay    = new QPushButton("&Play");
    QPushButton*          pcmdStop    = new QPushButton("&Stop");
    QPushButton*          pcmdPause   = new QPushButton("P&ause");
    QPushButton*          pcmdLoad    = new QPushButton("&Load");
    psldSeek    = new QSlider(Qt::Horizontal);
    psldSeek->setRange(0, 0);
    connect(psldSeek, SIGNAL(sliderMoved(int)),
                this, SLOT(slotSeek(int)));

    connect(pcmdPlay, SIGNAL(clicked()), mediaPlayer, SLOT(play()));
    connect(pcmdStop, SIGNAL(clicked()), mediaPlayer, SLOT(stop()));
    connect(pcmdPause, SIGNAL(clicked()), mediaPlayer, SLOT(pause()));
    connect(pcmdLoad, SIGNAL(clicked()), SLOT(slotLoad()));
    //connect(pcmdPause, SIGNAL(clicked()), SLOT(slotCapture()));

    //Layout setup
    QHBoxLayout* phbxLayout = new QHBoxLayout;    
    phbxLayout->addWidget(pcmdPlay);
    phbxLayout->addWidget(pcmdPause);
    phbxLayout->addWidget(pcmdStop);
    phbxLayout->addWidget(psldSeek);
    phbxLayout->addWidget(pcmdLoad);

    QVBoxLayout* pvbxLayout = new QVBoxLayout;

    videoWidget = new QVideoWidget;
    //videoWidget->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    //videoWidget->setAttribute(Qt::WA_OpaquePaintEvent);
    //videoWidget->setAspectRatioMode(Qt::KeepAspectRatio);
    pvbxLayout->addWidget(videoWidget);
    mediaPlayer->setVideoOutput(videoWidget);

    pvbxLayout->addLayout(phbxLayout);

    setLayout(pvbxLayout);


    //connect(mediaPlayer, SIGNAL(stateChanged(QMediaPlayer::State)),this, SLOT(mediaStateChanged(QMediaPlayer::State)));
    connect(mediaPlayer, SIGNAL(positionChanged(qint64)), this, SLOT(positionChanged(qint64)));
    connect(mediaPlayer, SIGNAL(durationChanged(qint64)), this, SLOT(durationChanged(qint64)));
    connect(mediaPlayer, SIGNAL(error(QMediaPlayer::Error)), this, SLOT(handleError()));

    slotLoad();
    resize(500, 500);
    show();
}

MediaPlayer::~MediaPlayer(){
    mediaPlayer->stop();
}

// ----------------------------------------------------------------------
void MediaPlayer::slotLoad()
{
    QString str = QFileDialog::getOpenFileName(0, "Load", "", "*.*");
    if (!str.isEmpty()) {
        mediaPlayer->setMedia(QUrl::fromLocalFile(str));
        //mediaPlayer->setPlaybackRate(0.33);
        mediaPlayer->play();
        //videoWidget->resize();
        /*capturedVideo->resize(pvw->size());
        capturedVideo->raise();*/
    }
}

void MediaPlayer::slotSeek(int position){
    mediaPlayer->setPosition(position);
}
void MediaPlayer::positionChanged(qint64 position)
{
    psldSeek->setValue(position);
}

void MediaPlayer::durationChanged(qint64 duration)
{
    psldSeek->setRange(0, duration);
}

void MediaPlayer::handleError()
{
    //errorLabel->setText("Error: " + mediaPlayer.errorString());
}
void MediaPlayer::slotCapture()
{
    //QPixmap image = QPixmap::grabWindow(QApplication::desktop()->winId());
    QPixmap image= //QPixmap::grabWindow(videoWidget->winId());
            QGuiApplication::primaryScreen()->grabWindow(videoWidget->winId());
    image.save("printScreen.png");
}

void MediaPlayer::grab(QOpenCVWidget* capturedVideo){
#if 1
    if(!frameGrabber->currentFrame.isNull()){
        //frameGrabber->currentFrame.save("printScreen.png");
        capturedVideo->putImage(frameGrabber->currentFrame);
    }
#else
    QPixmap image=
            //QPixmap::grabWindow(videoWidget->winId());
            //QGuiApplication::primaryScreen()->grabWindow(videoWidget->winId());
            videoWidget->grab();
    capturedVideo->putImage(image.toImage());
#endif
}
