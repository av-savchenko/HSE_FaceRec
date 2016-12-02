#include "SingleImageGrabber.h"
#include "OpenCvWrapper.h"

#include <QtNetwork>
#include <QFileDialog>

#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>

void SingleImageGrabber::processImage(){
    QDialog* requestFilterDialog=new QDialog(0);
    requestFilterDialog->setWindowTitle("Request filter");
    QGridLayout* pTopLayout=new QGridLayout();
    QLabel* filterLab=new QLabel("&Filter");
    QComboBox* filterComboBox=new QComboBox();
    filterComboBox->addItem("");
    filterComboBox->addItem("Noise");
    filterComboBox->addItem("Smooth");
    filterComboBox->addItem("Erode");
    filterComboBox->addItem("Dilate");
    filterComboBox->addItem("Morphology");
    filterComboBox->addItem("Threshold");
    filterComboBox->addItem("Sharpness");
    filterComboBox->addItem("Brightness");
    filterComboBox->addItem("Blackout");
    filterComboBox->addItem("Sobel");
    filterComboBox->addItem("Laplace");
    filterLab->setBuddy(filterComboBox);
    pTopLayout->addWidget(filterLab,0,0);
    pTopLayout->addWidget(filterComboBox,0,1,1,2);
    QLabel* paramLab=new QLabel("&Param");
    QLineEdit* paramEdit=new QLineEdit("1");
    paramLab->setBuddy(paramEdit);
    pTopLayout->addWidget(paramLab,1,0);
    pTopLayout->addWidget(paramEdit,1,1,1,2);
    QPushButton* okBtn=new QPushButton(QPushButton::tr("OK"));
    requestFilterDialog->connect(okBtn,SIGNAL(clicked()),SLOT(accept()));
    pTopLayout->addWidget(okBtn,3,2);

    requestFilterDialog->setLayout(pTopLayout);
    if(requestFilterDialog->exec()==QDialog::Accepted){
        bool ok;
        int param=paramEdit->text().toInt(&ok);
        if(ok){
            qDebug()<<filterComboBox->currentIndex()<<' '<<param;
            OpenCvWrapper::Instance().putFilter(&image, OpenCvWrapper::FilterType(filterComboBox->currentIndex()), param);
        }
    }
    delete requestFilterDialog;
}
SingleImageGrabber::SingleImageGrabber():image(QFileDialog::getOpenFileName(0, "Load", "", "*.*"))
{
    processImage();
}

void SingleImageGrabber::grab(QOpenCVWidget* capturedVideo){
    if(!image.isNull()){
        capturedVideo->putImage(image,true);
    }
}
