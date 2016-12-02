/* ======================================================================
**  MediaPlayer.h
** ======================================================================
**
** ======================================================================
**  Copyright (c) 2009 by Max Schlee
** ======================================================================
*/

#ifndef _MediaPlayer_h_
#define _MediaPlayer_h_

#include <QWidget>

#include "Grabable.h"
#include "QOpenCVWidget.h"

#include <qmediaplayer.h>

class QLabel;
class QMediaPlayer;
class QVideoWidget;
class QSlider;
class FrameGrabberVideoSurface;

// ======================================================================
class MediaPlayer : public QWidget, public Grabable {
    Q_OBJECT
private:
    QMediaPlayer* mediaPlayer;
    QVideoWidget *videoWidget;
    QSlider*   psldSeek;

    FrameGrabberVideoSurface* frameGrabber;

public:
    MediaPlayer(QWidget* pwgt = 0);
    ~MediaPlayer();
    void grab(QOpenCVWidget*);

public slots:
    void slotLoad();
    void slotCapture();
    void slotSeek(int position);
    //void mediaStateChanged(QMediaPlayer::State state);
    void positionChanged(qint64 position);
    void durationChanged(qint64 duration);
    void handleError();
};
#endif  //_MediaPlayer_h_

