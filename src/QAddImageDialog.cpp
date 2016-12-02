#include "QAddImageDialog.h"
#include <QtGui>
#include <QMessageBox>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>

QAddImageDialog::QAddImageDialog(const QImage& img, QString defaultPersonName/*="person"*/, QWidget* pwgt/*=0*/):QDialog(pwgt),image(img)
{
    setWindowTitle(tr("add image"));
    QGridLayout* pTopLayout=new QGridLayout();
    QLabel* imgLabel=new QLabel(this);
    imgLabel->setPixmap(QPixmap::fromImage(image));
    pTopLayout->addWidget(imgLabel,0,0,3,3);
    QLabel* labName=new QLabel(tr("Name"),this);
    pTopLayout->addWidget(labName,0,4,1,2);
    editClassName=new QLineEdit(defaultPersonName,this);
    editClassName->setMaxLength(32);
    pTopLayout->addWidget(editClassName,1,4,1,2);

    QPushButton* okBtn=new QPushButton(tr("OK"),this);
    connect(okBtn,SIGNAL(clicked()),SLOT(okClicked()));
    pTopLayout->addWidget(okBtn,0,6);
    QPushButton* cancelBtn=new QPushButton(tr("Cancel"),this);
    connect(cancelBtn,SIGNAL(clicked()),SLOT(reject()));
    pTopLayout->addWidget(cancelBtn,1,6);
    setLayout(pTopLayout);
}
void QAddImageDialog::okClicked(){
    QString personName=editClassName->text();
    if(personName.isEmpty()){
        QMessageBox::information(0,tr("info"),tr("Enter not empty person name"));
    }
    else
        accept();
}
