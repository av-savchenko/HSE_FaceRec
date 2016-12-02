#include "DisplayImageDialog.h"
#include <QtGui>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QPushButton>

DisplayImageDialog::DisplayImageDialog(const QString& date, const QStringList& imageFiles, QWidget* pwgt/*=0*/):QDialog(pwgt)
{
    setWindowTitle(date);
    QGridLayout* pTopLayout=new QGridLayout();
    QHBoxLayout* imagesLayout=new QHBoxLayout();

    foreach(QString file, imageFiles){
        QLabel* imageLabel=new QLabel();
        QPixmap pixmap(file);
        //qDebug()<<pixmap.width()<<' '<<pixmap.height()<<' '<<file;
        imageLabel->setFixedSize(140,160);
        //imageLabel->setScaledContents(true);
        imageLabel->setPixmap(pixmap.scaled(imageLabel->size()));
        imageLabel->setFrameStyle(QFrame::Box|QFrame::Plain);
        imagesLayout->addWidget(imageLabel);
    }
    QWidget* imagesWidget=new QWidget;
    imagesWidget->setLayout(imagesLayout);
    QScrollArea* scrollImagesWidget=new QScrollArea;
    scrollImagesWidget->setWidget(imagesWidget);
    scrollImagesWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollImagesWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    pTopLayout->addWidget(scrollImagesWidget,0,0,3,3);

    QPushButton* closeBtn=new QPushButton(tr("Close"),this);
    connect(closeBtn,SIGNAL(clicked()),SLOT(accept()));
    pTopLayout->addWidget(closeBtn,3,2);

    setLayout(pTopLayout);
}
