#include "AppSettings.h"

#include <QtGui>
#include <QDialog>
#include <QGridLayout>
#include <QCheckBox>
#include <QPushButton>

AppSettings::AppSettings():settings("asav","FaceRecognition")
{
}
bool AppSettings::storeUnrecognized(){
    return settings.value("/Settings/storeUnrecognized",false).toBool();
}
void AppSettings::setStoreUnrecognized(bool storeUnrecognized){
    settings.setValue("/Settings/storeUnrecognized",storeUnrecognized);
}
bool AppSettings::addUnrecognizedToDatabase(){
    return settings.value("/Settings/addUnrecognizedToDatabase",false).toBool();
}
void AppSettings::setAddUnrecognizedToDatabase(bool addUnrecognizedToDatabase){
    settings.setValue("/Settings/addUnrecognizedToDatabase",addUnrecognizedToDatabase);
}

void AppSettings::showDialog(){
    QDialog* saveDialog=new QDialog(0);
    saveDialog->setWindowTitle(tr("Options"));
    QGridLayout* pTopLayout=new QGridLayout();
    QCheckBox* storeUnrecCheck=new QCheckBox(tr("store unrecognized images"));
    storeUnrecCheck->setChecked(storeUnrecognized());
    pTopLayout->addWidget(storeUnrecCheck,0,0,1,3);
    QPushButton* okBtn=new QPushButton(tr("OK"));
    saveDialog->connect(okBtn,SIGNAL(clicked()),SLOT(accept()));
    pTopLayout->addWidget(okBtn,1,1);
    QPushButton* cancelBtn=new QPushButton(tr("Cancel"));
    saveDialog->connect(cancelBtn,SIGNAL(clicked()),SLOT(reject()));
    pTopLayout->addWidget(cancelBtn,1,2);

    saveDialog->setLayout(pTopLayout);
    if(saveDialog->exec()==QDialog::Accepted){
        setStoreUnrecognized(storeUnrecCheck->isChecked());
    }
    delete saveDialog;
}
