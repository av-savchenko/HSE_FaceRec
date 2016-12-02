#include "DbReader.h"
#include "OpenCvWrapper.h"

#ifdef USE_ORA_DATABASE

#include <QtSql>
#include <QtGui>
#include <QtCore>
#include <QDialog>
#include <QProgressDialog>
#include <QGridLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

/*
  CREATE TABLE persons (id int PRIMARY KEY,name varchar(32));
  CREATE TABLE photos (personId int,photo blob, FOREIGN KEY(personId) REFERENCES persons(id));
*/
DbReader::DbReader():
        initialized(false),
        db(QSqlDatabase::addDatabase("QOCI"))
{
    QString host("localhost");
    QString dbName("XE");
    QString username("SYSDBA");
    QString password;
    if(!requestCredentials(host,dbName,username,password)){
        qDebug()<<"Database not selected. Return";
        return;
    }

    db.setHostName(host);
    db.setDatabaseName(dbName);
    db.setUserName(username);
    db.setPassword(password);
    if (!db.open()) {
        qDebug() << "Cannot open database:" << db.lastError();
        return;
    }
    qDebug() << "opened "<<db.connectionName();
    initialized=true;
}
DbReader::~DbReader(){
    qDebug()<<"closed";
    db.close();
    QSqlDatabase::removeDatabase("QOCI");
}
bool DbReader::requestCredentials(QString& host, QString& dbName, QString& username,QString& password){
    QDialog* requestCredentialsDialog=new QDialog(0);
    requestCredentialsDialog->setWindowTitle(QObject::tr("Connect to database"));
    QGridLayout* pTopLayout=new QGridLayout();
    QLabel* hostLab=new QLabel(QObject::tr("&Host"));
    QLineEdit* hostEdit=new QLineEdit(host);
    hostLab->setBuddy(hostEdit);
    pTopLayout->addWidget(hostLab,0,0);
    pTopLayout->addWidget(hostEdit,0,1,1,2);
    QLabel* dbLab=new QLabel(QObject::tr("&Database"));
    QLineEdit* dbEdit=new QLineEdit(dbName);
    dbLab->setBuddy(dbEdit);
    pTopLayout->addWidget(dbLab,1,0);
    pTopLayout->addWidget(dbEdit,1,1,1,2);
    QLabel* userLab=new QLabel(QObject::tr("User&name"));
    QLineEdit* userEdit=new QLineEdit(username);
    userLab->setBuddy(userEdit);
    pTopLayout->addWidget(userLab,2,0);
    pTopLayout->addWidget(userEdit,2,1,1,2);
    QLabel* passwordLab=new QLabel(QObject::tr("&Password"));
    QLineEdit* passwordEdit=new QLineEdit(password);
    passwordEdit->setEchoMode(QLineEdit::Password);
    passwordLab->setBuddy(passwordEdit);
    pTopLayout->addWidget(passwordLab,3,0);
    pTopLayout->addWidget(passwordEdit,3,1,1,2);
    QPushButton* okBtn=new QPushButton(QObject::tr("OK"));
    requestCredentialsDialog->connect(okBtn,SIGNAL(clicked()),SLOT(accept()));
    pTopLayout->addWidget(okBtn,4,1);
    QPushButton* cancelBtn=new QPushButton(QObject::tr("Cancel"));
    requestCredentialsDialog->connect(cancelBtn,SIGNAL(clicked()),SLOT(reject()));
    pTopLayout->addWidget(cancelBtn,4,2);

    requestCredentialsDialog->setLayout(pTopLayout);

    bool res=false;
    if(requestCredentialsDialog->exec()==QDialog::Accepted){
        host=hostEdit->text();
        dbName=dbEdit->text();
        username=userEdit->text();
        password=passwordEdit->text();
        res=true;
    }
    delete requestCredentialsDialog;
    return res;
}
void DbReader::writeOneDirIntoDatabase(const QDir& dir){
    if(!initialized){
        qDebug()<<"Database not selected. Return";
        return;
    }
    QRegExp rx("(\\d+)");
    QSqlQuery insertPerson,insertPhoto;
    insertPerson.prepare("insert into persons(id,name) values (:id, :name)");
    insertPhoto.prepare("insert into photos(personId,photo) values (:personId, :photo)");


    QStringList photos=dir.entryList(QDir::Files);
    int progress=0;
    QProgressDialog* progressDlg=new QProgressDialog(QObject::tr("Export database"),QObject::tr("Cancel"),0,photos.size());
    foreach(QString photo, photos){
        if(progressDlg->wasCanceled())
            break;
        QString lowerPhoto=photo.toLower();
        QImage image(dir.filePath(photo));
        if(!image.isNull()){
            if(rx.indexIn(lowerPhoto)!=-1){
                int id=rx.cap(1).toInt();
                insertPerson.bindValue(":id",id);
                insertPerson.bindValue(":name",photo.mid(0,photo.indexOf('.')));
                insertPerson.exec();

                /*QFile photoFile(dir.filePath(photo));
                    photoFile.open(QIODevice::ReadOnly);
                    QByteArray buffer=photoFile.readAll();
                    qDebug()<<dir.filePath(photo);*/
                QByteArray ba;
                QBuffer buffer(&ba);
                //buffer.open(Qt::IO_WriteOnly );
                image.save( &buffer, "jpg" );
                qDebug()<<ba.size();
                insertPhoto.bindValue(":personId",id);
                insertPhoto.bindValue(":photo",ba);
                if(!insertPhoto.exec()){
                    qDebug() << "Unable to execute query id="<<id<<" file="<<photo;
                }
            }
            else
                qDebug()<<"Invalid filename "<<photo;
        }
        progressDlg->setValue(++progress);
    }
    delete progressDlg;
    qDebug()<<"write complete";
}
void DbReader::writeDirsIntoDatabase(const QDir& dir){
    if(!initialized){
        qDebug()<<"Database not selected. Return";
        return;
    }
    QSqlQuery insertPerson,insertPhoto;
    insertPerson.prepare("insert into persons(id,name) values (:id, :name)");
    insertPhoto.prepare("insert into photos(personId,photo) values (:personId, :photo)");

    QStringList persons=dir.entryList(QDir::Dirs);
    int progress=0;
    QProgressDialog* progressDlg=new QProgressDialog(QObject::tr("Export database"),QObject::tr("Cancel"),0,persons.size()-2);

    int id=0;
    foreach(QString personName, persons){
        if(progressDlg->wasCanceled())
            break;
        if(personName.startsWith("."))
            continue;
        insertPerson.bindValue(":id",id);
        insertPerson.bindValue(":name",personName);
        insertPerson.exec();
        QDir personDir(dir.filePath(personName));
        QStringList photos=personDir.entryList(QDir::Files);
        foreach(QString photo, photos){
            if(progressDlg->wasCanceled())
                break;
            QImage image(personDir.filePath(photo));
            if(!image.isNull()){
                QByteArray ba;
                QBuffer buffer(&ba);
                image.save( &buffer, "jpg" );
                insertPhoto.bindValue(":personId",id);
                insertPhoto.bindValue(":photo",ba);
                if(!insertPhoto.exec()){
                    qDebug() << "Unable to execute query id="<<id<<" file="<<photo;
                }
            }
        }
        ++id;
        progressDlg->setValue(++progress);
    }
    delete progressDlg;
    qDebug()<<"write complete";
}
void DbReader::readFacesIntoDir(QDir dir,bool detectFaces){
    if(!initialized){
        qDebug()<<"Database not selected. Return";
        return;
    }
    QSqlQuery selectPersons,selectPhotos;
    selectPersons.setForwardOnly(true);
    selectPhotos.setForwardOnly(true);
    //if (!selectPersons.exec("SELECT id,name FROM persons")) {
    if (!selectPersons.exec("SELECT R_1 as id FROM FOTOCORI.OBJ009")) {
        qDebug() << "Unable to execute query (SELECT id,name FROM persons) - exiting";
        return;
    }

    //selectPhotos.prepare("select photo from photos where personId=:id");
    selectPhotos.prepare("select A_001 AS photo from FOTOCORI.OBJ009 where R_1=:id");

    //Reading of the data
    QSqlRecord personsRec = selectPersons.record();
    int idInd=personsRec.indexOf("id");
    int nameInd=personsRec.indexOf("name");

    int id;
    QString name;

    int progress=0;
    QProgressDialog* progressDlg=new QProgressDialog(QObject::tr("Import database"),QObject::tr("Cancel"),0,selectPersons.size());

    while (selectPersons.next()) {
        /*if(progressDlg->wasCanceled())
            break;*/
        id  = selectPersons.value(idInd).toInt();
        //name  = selectPersons.value(nameInd).toString();
        name.setNum(id);
qDebug()<<name;

        selectPhotos.bindValue(":id",id);
        if(selectPhotos.exec()){
            QSqlRecord photosRec= selectPhotos.record();
            int photoInd=photosRec.indexOf("photo");

            int ind=0;
            while(selectPhotos.next()){
                QByteArray buffer=selectPhotos.value(photoInd).toByteArray();
                QImage photo=QImage::fromData(buffer,"jpg");
                qDebug()<<photo.width()<<' '<<buffer.size();
                QList<QImage> faces;
                if(detectFaces){
                    QVector<std::pair<QRect,QImage> > facesWithRect=
                            OpenCvWrapper::Instance().detectFaces(OpenCvWrapper::Instance().detectFaceRects(&photo,true),photo);
                    for(QVector<std::pair<QRect,QImage> >::const_iterator iter=facesWithRect.begin();iter!=facesWithRect.end();++iter)
                        faces.push_back(iter->second);
                }
                else
                    faces.push_back(photo);
                if(!faces.empty()){
                    if(ind==0){
                        dir.mkdir(name);
                        dir.cd(name);
                    }
                    foreach(QImage face, faces){
                        ++ind;
                        QString fileName=QString("%1.jpg").arg(ind);
                        face.save(dir.filePath(fileName),"jpg");
                    }
                }
            }
            if(ind>0)
                dir.cd("..");
        }
        else
            qDebug() << "Unable to select photo for id="<<id;
//exit(0);
        progressDlg->setValue(++progress);
    }
    delete progressDlg;
    qDebug()<<"read complete";
}
#endif //USE_ORA_DATABASE
