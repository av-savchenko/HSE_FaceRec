#include "IpCameraGrabber.h"

#include <QtNetwork>
#include <QtGui>
#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>

IpCameraGrabber::IpCameraGrabber():QObject()
{
    QString url="http://192.168.0.20/video/mjpg.cgi?profileid=3";
    QString username="admin";
    QString password="";
    requestCredentials(url,username,password);

    m_netwManager = new QNetworkAccessManager(this);
    //connect(m_netwManager,SIGNAL(authenticationRequired(QNetworkReply*,QAuthenticator*)),this,SLOT(slot_netwManagerFinished(QNetworkReply*,QAuthenticator*)));
    QUrl cameraUrl(url);
    QNetworkRequest request(cameraUrl);
    QString concatenated = username+":"+password;
    QByteArray data = concatenated.toLocal8Bit().toBase64();
    QString headerData = "Basic " + data;
    request.setRawHeader("Authorization", headerData.toLocal8Bit());

    m_reply=m_netwManager->get(request);
    connect(m_reply,SIGNAL(readyRead()),this,SLOT(getData()));
}
void IpCameraGrabber::requestCredentials(QString& url, QString& username,QString& password){
    QDialog* requestCredentialsDialog=new QDialog(0);
    requestCredentialsDialog->setWindowTitle(tr("IP camera"));
    QGridLayout* pTopLayout=new QGridLayout();
    QLabel* urlLab=new QLabel(tr("&URL"));
    QLineEdit* urlEdit=new QLineEdit(url);
    urlLab->setBuddy(urlEdit);
    pTopLayout->addWidget(urlLab,0,0);
    pTopLayout->addWidget(urlEdit,0,1,1,2);
    QLabel* userLab=new QLabel(tr("User&name"));
    QLineEdit* userEdit=new QLineEdit(username);
    userLab->setBuddy(userEdit);
    pTopLayout->addWidget(userLab,1,0);
    pTopLayout->addWidget(userEdit,1,1,1,2);
    QLabel* passwordLab=new QLabel(tr("&Password"));
    QLineEdit* passwordEdit=new QLineEdit(password);
    passwordEdit->setEchoMode(QLineEdit::Password);
    passwordLab->setBuddy(passwordEdit);
    pTopLayout->addWidget(passwordLab,2,0);
    pTopLayout->addWidget(passwordEdit,2,1,1,2);
    QPushButton* okBtn=new QPushButton(tr("OK"));
    requestCredentialsDialog->connect(okBtn,SIGNAL(clicked()),SLOT(accept()));
    pTopLayout->addWidget(okBtn,3,2);

    requestCredentialsDialog->setLayout(pTopLayout);
    if(requestCredentialsDialog->exec()==QDialog::Accepted){
        url=urlEdit->text();
        username=userEdit->text();
        password=passwordEdit->text();
    }
    delete requestCredentialsDialog;
}
/*void IpCameraGrabber::slot_netwManagerFinished(QNetworkReply*,QAuthenticator* auth)
{
    qDebug()<<"auth";
    auth->setUser("admin");
    auth->setPassword("");
}*/
void IpCameraGrabber::getData(){
    if (m_reply->error() != QNetworkReply::NoError) {
        m_reply->deleteLater();
        return;
    }
    QByteArray content;
    content = m_reply->readAll();
    QByteArray flag("\r\n\r\n");
    int iEnd = content.indexOf(flag);
    QByteArray b("content-length: ");
    int pos = content.toLower().indexOf(b);

    if(pos>0)//case 1,content contains the header;
    {
        QString length;
        QByteArray temp = content.mid(pos+16);
        length.append(temp.left(temp.indexOf("\r\n")));
        int len = length.toInt();

        pixmap.loadFromData(m_imgData);
        m_imgData.resize(len);//of new size.
        m_len = len;
        m_imgData.clear();//clear the content of the variable--imgData.
        m_imgData.append(content.mid(iEnd+4));//add the data backward the header to imgData.
    }
    else if( pos < 0&& iEnd <0)//case 2,content dose not contain the header;
    {
        m_imgData.append(content);//add the data of the content to imgData.
    }
    else
    {
        pixmap.loadFromData(m_imgData);
        m_imgData.clear();
        m_imgData.append(content.mid(iEnd+4));
    }
}


void IpCameraGrabber::grab(QOpenCVWidget* capturedVideo){
    if(!pixmap.isNull())
        capturedVideo->putImage(pixmap.toImage());
}
