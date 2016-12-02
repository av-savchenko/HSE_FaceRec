#ifndef IPCAMERAGRABBER_H
#define IPCAMERAGRABBER_H

#include <QPixmap>

#include "Grabable.h"
#include "QOpenCVWidget.h"


class QNetworkReply;
class QAuthenticator;
class QNetworkAccessManager;

class IpCameraGrabber: public QObject, public Grabable
{
    Q_OBJECT
private:
    QNetworkAccessManager* m_netwManager;
    QNetworkReply *m_reply;
    QByteArray m_imgData;
    QPixmap pixmap;

    int m_len;

    static void requestCredentials(QString& url, QString& username,QString& password);

public:
    IpCameraGrabber();
    void grab(QOpenCVWidget*);

public slots:
    void getData();

};

#endif // IPCAMERAGRABBER_H
