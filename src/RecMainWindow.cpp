#include <fstream>
#include <QTextStream>

#include "RecMainWindow.h"
#include "QAddImageDialog.h"
#include "DisplayImageDialog.h"
#include "GrabableFactory.h"
#include "DbReader.h"

#include "CameraGrabber.h"
#include "MediaPlayer.h"
#include "IpCameraGrabber.h"
#include "SingleImageGrabber.h"

#include <QtGui>
#include <QSplitter>

const QString UNRECOGNIZED=QObject::tr("unrec");
const QString NOONE=QObject::tr("noone");

const int TOTAL_CHECK_COUNT=4;
const int FOUND_COUNT_THRESHOLD=3;

#ifdef USE_DNN_FEATURES
#define DISTANCE_THRESHOLD 4
#define DISTANCE_DIFF_THRESHOLD 0.5
#define MAX_DISTANCE_FOR_DIFF_THRESHOLD 4.2
#define MAX_DISTANCE_THRESHOLD 4.5

#else
#ifdef USE_KL
#define DISTANCE_THRESHOLD 0.27
#define DISTANCE_DIFF_THRESHOLD 0.0025
#define MAX_DISTANCE_FOR_DIFF_THRESHOLD 0.35
#define MAX_DISTANCE_THRESHOLD 0.3
#else
#ifdef USE_GRADIENT_ANGLE

#if 1
#define DISTANCE_THRESHOLD 0.5 //.62
#define DISTANCE_DIFF_THRESHOLD 0.03 //0.02
#define MAX_DISTANCE_FOR_DIFF_THRESHOLD 0.7 //0.75
#define MAX_DISTANCE_THRESHOLD 0.7
#else
#define DISTANCE_THRESHOLD 0.45 //.62
#define DISTANCE_DIFF_THRESHOLD 0.05 //0.02
#define MAX_DISTANCE_FOR_DIFF_THRESHOLD 0.65 //0.75
#define MAX_DISTANCE_THRESHOLD 0.65
#endif

#else
#define DISTANCE_THRESHOLD 0.009
#define DISTANCE_DIFF_THRESHOLD 0.0012
#define MAX_DISTANCE_FOR_DIFF_THRESHOLD 0.017
#define MAX_DISTANCE_THRESHOLD 0.017
#endif
#endif //USE_KL

#endif //USE_DNN_FEATURES

const int SAME_PERSON_MAX_POINT_DIST=40;
QFile myfile;
QMap<int, int> list1;
QList<int> list2;
QTextStream out(&myfile);//,, QIODevice::Text);
// 9,03 QTimer mytimer;
//QTimer *mytimer;// = new QTimer(this);
////////////////

#include <QApplication>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QToolBar>
#include <QHBoxLayout>
#include <QTableWidget>


RecMainWindow::RecMainWindow(QWidget *parent) :
        QMainWindow(parent),
        //grabber(Grabable::Create<CameraGrabber>()),
        grabber(0),
        lastRecognized(NOONE){

    GrabableFactory::Instance().registerCreator(tr("Media file"),Grabable::Create<MediaPlayer>);
    GrabableFactory::Instance().registerCreator(tr("Image"),Grabable::Create<SingleImageGrabber>);
    GrabableFactory::Instance().registerCreator(tr("Ip camera"),Grabable::Create<IpCameraGrabber>);

    QAction *pactAdd=new QAction(QIcon(":/images/imgAdd.png"),tr("Add photo"),this);
    connect(pactAdd,SIGNAL(triggered()),SLOT(saveFace()));

    QMenu* file=new QMenu(tr("&File"));
    file->addAction(pactAdd);

#ifdef USE_ORA_DATABASE
    file->addAction(tr("&Export database"),this,SLOT(exportDatabase()));
    file->addAction(tr("&Import database"),this,SLOT(importDatabase()));
#endif

    file->addSeparator();
    file->addAction(tr("&Exit"),QApplication::instance(),SLOT(quit()));
    menuBar()->addMenu(file);
    QMenu* tools=new QMenu(tr("&Tools"));
    tools->addAction(tr("&Options"),&appSettings,SLOT(showDialog()));
    menuBar()->addMenu(tools);

    QMenu* cameras=new QMenu(tr("&Cameras"));
    QList<QString> cameraNames=GrabableFactory::Instance().getNames();
    for(QList<QString>::const_iterator iter=cameraNames.begin();iter!=cameraNames.end();++iter){
        cameras->addAction(*iter);
    }
    connect(cameras,SIGNAL(triggered(QAction*)),SLOT(changeCamera(QAction*)));
    menuBar()->addMenu(cameras);

    QToolBar* ptb=new QToolBar(this);
    ptb->addAction(pactAdd);
    addToolBar(Qt::TopToolBarArea,ptb);
    cvwidget = new QOpenCVWidget();
    cvwidget->setMinimumSize(850,480);
    nameTable=new QTableWidget(0,2);
    nameTable->setHorizontalHeaderLabels(QStringList()<<tr("persons")<<tr("Date"));
    nameTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    nameTable->connect(nameTable, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(openImageViewer(int,int)) );
    nameTable->setMinimumSize(250,480);
    QSplitter *splitter = new QSplitter(this);
    splitter->addWidget(cvwidget);
    splitter->addWidget(nameTable);
    setCentralWidget(splitter);

    timerEvent(0);
    timerId=startTimer(150);  // 0.1-second timer
}
void RecMainWindow::closeEvent(QCloseEvent *event){
    killTimer(timerId);
    grabber.reset(0);
    event->accept();
}
void RecMainWindow::changeCamera(QAction* pAction){
    Grabable* newGrabable=GrabableFactory::Instance().create(pAction->text());
    if(newGrabable)
    {
        grabber.reset(newGrabable);
    }
    else
        qDebug()<<"Invalid action "<<pAction->text();
}

void RecMainWindow::openImageViewer(int row, int column){
    QStringList names=nameTable->model()->data(nameTable->model()->index(row,0)).toString().split(",");
    QString date=nameTable->model()->data(nameTable->model()->index(row,1)).toString();
    //qDebug()<<row<<' '<<column<<' '<<names;

    QStringList imageFiles;
    foreach(QString name,names){
        if(name==NOONE)
            continue;
        if(name==UNRECOGNIZED){
            ;
        }
        else{
            if(name.startsWith("unrec(")){
                name=name.mid(6,name.length()-6-1);
            }
            QString imageFileName=db.getPersonFile(name);
            if(!imageFileName.isEmpty()){
                imageFiles.append(imageFileName);
            }
        }
    }
    if(!imageFiles.isEmpty()){
        DisplayImageDialog* dlg=new DisplayImageDialog(date,imageFiles,this);
        //dlg->setBaseSize(400,400);
        dlg->exec();
        delete dlg;
   }
}

void RecMainWindow::timerEvent(QTimerEvent*) {

    if(grabber.get()==0)
        return;
    //qDebug()<<"timer start()";
    QTime myTimer;myTimer.start();
    grabber->grab(cvwidget);
    //qDebug()<<"grabbed "<<myTimer.elapsed();
    QVector<std::pair<QRect,QImage> > face_images=cvwidget->getFaces();
    cvwidget->paintFaces();
    PatternInfo::PersonsType detectedMap,unrecognizedMap;
    PatternInfo::ImagesType unrecImages;
    for(QVector<std::pair<QRect,QImage> >::const_iterator iter=face_images.begin();iter!=face_images.end();++iter){
        const QPoint point=iter->first.center();
        const QImage& image=iter->second;
        FaceImage* inputImage=new FaceImage(&image);
        double dist,distDiff;
        FaceImage* face=db.getClosest(inputImage,&dist,&distDiff);
        if(face!=0){
            QString closestPersonName=face->personName;

            //if(dist<DISTANCE_THRESHOLD)
            if((dist<DISTANCE_THRESHOLD) || (dist<MAX_DISTANCE_FOR_DIFF_THRESHOLD && distDiff>=DISTANCE_DIFF_THRESHOLD))
                detectedMap.insert(closestPersonName,point);
            else if(dist<MAX_DISTANCE_THRESHOLD)
                unrecognizedMap.insert(closestPersonName,point);
            else
                face=0;
        }
        if(!face)
            unrecImages.push_back(std::make_pair(point,image));

        delete inputImage;
    }
    //qDebug()<<"rec="<<detectedMap<<" unrec="<<unrecognizedMap;
    patternInfoList.push_front(PatternInfo(detectedMap,unrecognizedMap,unrecImages));
    if(patternInfoList.size()>=TOTAL_CHECK_COUNT){
        PatternInfo newDetectedPersons=getDetectedPersons();
        //qDebug()<<"newRec="<<newDetectedPersons.persons<<" unrec="<<newDetectedPersons.unrecPersons;
        bool changed=(newDetectedPersons!=detectedPersons);
        bool justRecognized=false,unrecCountChanged=false;

        //qDebug()<<"lastRecognized="<<lastRecognized<<" changed="<<changed;
        if(lastRecognized!=NOONE){
            bool noPersonCountChanged=(detectedPersons.get_total_persons_count()==newDetectedPersons.get_total_persons_count());
            if(!changed && noPersonCountChanged && !detectedPersons.persons.isEmpty()){
                justRecognized=true;
                for(PatternInfo::PersonsType::const_iterator iter=detectedPersons.persons.begin();iter!=detectedPersons.persons.end();++iter){
                    if(!newDetectedPersons.persons.contains(iter.key())){
                        justRecognized=false;
                        break;
                    }
                }
            }
            if(changed && noPersonCountChanged && !justRecognized){
                for(PatternInfo::ImagesType::const_iterator unrecIter=detectedPersons.unrecImages.begin();unrecIter!=detectedPersons.unrecImages.end();++unrecIter){
                    const QPoint& unrecCenter=unrecIter->first;
                    for(PatternInfo::PersonsType::const_iterator iter=newDetectedPersons.persons.begin();iter!=newDetectedPersons.persons.end();++iter){
                        if((unrecCenter-iter.value()).manhattanLength()<SAME_PERSON_MAX_POINT_DIST){
                            justRecognized=true;
                            break;
                        }
                    }
                    if(justRecognized)
                        break;
                    for(PatternInfo::PersonsType::const_iterator iter=newDetectedPersons.unrecPersons.begin();iter!=newDetectedPersons.unrecPersons.end();++iter){
                        //qDebug()<<"manLen="<<(unrecCenter-iter.value()).manhattanLength();
                        if((unrecCenter-iter.value()).manhattanLength()<SAME_PERSON_MAX_POINT_DIST){
                            justRecognized=true;
                            break;
                        }
                    }
                    if(justRecognized)
                        break;
                }
                //qDebug()<<"HIII "<<justRecognized<<' '<<changed;
            }

            if(changed && noPersonCountChanged && !justRecognized){
                PatternInfo::ImagesType unrec;
                for(PatternInfo::ImagesType::iterator unrecIter=newDetectedPersons.unrecImages.begin();unrecIter!=newDetectedPersons.unrecImages.end();++unrecIter){
                    const QPoint& unrecCenter=unrecIter->first;
                    bool found=false;
                    for(PatternInfo::PersonsType::const_iterator iter=detectedPersons.persons.begin();iter!=detectedPersons.persons.end();++iter){
                        if((unrecCenter-iter.value()).manhattanLength()<SAME_PERSON_MAX_POINT_DIST){
                            newDetectedPersons.persons.insert(iter.key(),unrecCenter);
                            found=true;
                            break;
                        }
                    }
                    if(!found){
                        for(PatternInfo::PersonsType::const_iterator iter=detectedPersons.unrecPersons.begin();iter!=detectedPersons.unrecPersons.end();++iter){
                            //qDebug()<<"manLen="<<(unrecCenter-iter.value()).manhattanLength();
                            if((unrecCenter-iter.value()).manhattanLength()<SAME_PERSON_MAX_POINT_DIST){
                                newDetectedPersons.unrecPersons.insert(iter.key(),unrecCenter);
                                found=true;
                                break;
                            }
                        }
                    }
                    if(!found)
                        unrec.push_back(*unrecIter);
                }
                newDetectedPersons.unrecImages=unrec;
            }
        }
        int unrecCount=newDetectedPersons.unrecImages.size();
        if(!changed && unrecCount!=detectedPersons.unrecImages.size())
            unrecCountChanged=true;

        //qDebug()<<changed<<' '<<justRecognized<<' '<<unrecCountChanged;
        if(changed || justRecognized || unrecCountChanged){
            QString persons="";
            for(PatternInfo::PersonsType::const_iterator iter=newDetectedPersons.persons.begin();iter!=newDetectedPersons.persons.end();++iter){
                if(!persons.isEmpty())
                    persons+=",";
                persons+=iter.key();
            }
            for(PatternInfo::PersonsType::const_iterator iter=newDetectedPersons.unrecPersons.begin();iter!=newDetectedPersons.unrecPersons.end();++iter){
                if(!persons.isEmpty())
                    persons+=",";
                persons+="unrec("+iter.key()+")";
            }
            if(unrecCount>0){
                int partiallyUnrecCount=unrecCount;
                if(appSettings.storeUnrecognized()){
                    partiallyUnrecCount=0;
                    for(PatternInfo::ImagesType::const_iterator iter=newDetectedPersons.unrecImages.begin();iter!=newDetectedPersons.unrecImages.end();++iter){
                        const QImage& image=iter->second;
                        if(image.width()>0){
                            if(!persons.isEmpty())
                                persons+=",";
                            persons+=db.addUnrecognized(image);
                        }else
                            ++partiallyUnrecCount;
                    }
                }
                if(partiallyUnrecCount>0){
                    if(!persons.isEmpty())
                        persons+=",";
                    if(partiallyUnrecCount>1)
                        persons+=QString("%1x").arg(partiallyUnrecCount);
                    persons+=UNRECOGNIZED;
                }
            }
            if(persons.isEmpty()){
                /*if(face_images.empty())
                    persons=NOONE;
                else
                    persons=UNRECOGNIZED;*/
                 persons=NOONE;
            }



            /////////////////
            // таблица. person колонка
            if(persons!=lastRecognized)
            {
                int row=nameTable->model()->rowCount();
                if(justRecognized){
                    nameTable->setItem(row-1,0,new QTableWidgetItem(persons));
                }
               else{
                    nameTable->model()->insertRow(row);
                    qDebug() << "Person" << persons << patternInfoList.front().creationTime.toString("hh:mm:ss");
                    nameTable->setItem(row,0,new QTableWidgetItem(persons));
                    nameTable->setItem(row,1,new QTableWidgetItem(patternInfoList.front().creationTime.toString("hh:mm:ss")));
                    nameTable->scrollToBottom();
                }

                lastRecognized=persons;
            }
        }
        patternInfoList.clear();
        //patternInfoList.pop_back();
        detectedPersons=newDetectedPersons;
    }
    //qDebug()<<"end "<<myTimer.elapsed();
}

typedef QVector<std::pair<QPoint,QVector<const QImage*> > > UnrecLists;
static void checkPointPresence(UnrecLists & unrecLists, const QPoint& point, const QImage* image){
    bool found=false;

    for(UnrecLists::iterator unrecIter=unrecLists.begin();unrecIter!=unrecLists.end();++unrecIter){
        if((unrecIter->first-point).manhattanLength()<SAME_PERSON_MAX_POINT_DIST){
            unrecIter->first=point;
            unrecIter->second.push_back(image);
            found=true;
            break;
        }
    }
    if(!found){
        unrecLists.push_back(std::make_pair(point,QVector<const QImage*>()<<image));
    }
}
///////////////////////
// метод записи в файл
void RecMainWindow::toWrite() {
    qDebug() << "WRITE!!!!!!!!!!!!!";

    int row=nameTable->model()->rowCount();
    if (myfile.open(QFile::WriteOnly)| QFile::Truncate) {
       QTextStream out(&myfile);
// тут было много строчек кода, которые я случайно удалила
     for(int i=0;i<row;i++){
        out << ";" << "Result: " <<  ";" << nameTable->item(i,0)->text() <<";" << nameTable->item(i,1)->text() ;
         qDebug() << "ResultRESULT: " <<   nameTable->item(i,0)->text() << nameTable->item(i,1)->text() ;
         out<<'\n';//}
     }
    }
    myfile.close();
}


PatternInfo RecMainWindow::getDetectedPersons(){
    QMap<QString,int> recPerson2Count,unrecPerson2Count;
    QMap<QString,QPoint> person2PointMap;
    for(QList<PatternInfo>::const_iterator iter=patternInfoList.begin();iter!=patternInfoList.end();++iter){
        for(PatternInfo::PersonsType::const_iterator setIter=iter->persons.begin();setIter!=iter->persons.end();++setIter){
            ++recPerson2Count[setIter.key()];
            person2PointMap[setIter.key()]+=setIter.value();
        }
        for(PatternInfo::PersonsType::const_iterator setIter=iter->unrecPersons.begin();setIter!=iter->unrecPersons.end();++setIter){
            ++unrecPerson2Count[setIter.key()];
            person2PointMap[setIter.key()]+=setIter.value();
        }
    }
    for(QMap<QString,QPoint>::iterator iter=person2PointMap.begin();iter!=person2PointMap.end();++iter)
        iter.value()/=(recPerson2Count[iter.key()]+unrecPerson2Count[iter.key()]);

    PatternInfo::PersonsType finalRecPersons,finalUnrecPersons;
    for(QMap<QString,int>::const_iterator iter=recPerson2Count.begin();iter!=recPerson2Count.end();++iter)
        if(iter.value()>=FOUND_COUNT_THRESHOLD)
            finalRecPersons.insert(iter.key(),person2PointMap[iter.key()]);
        else
            ++unrecPerson2Count[iter.key()];

    for(QMap<QString,int>::const_iterator iter=unrecPerson2Count.begin();iter!=unrecPerson2Count.end();++iter)
        if(iter.value()>=FOUND_COUNT_THRESHOLD)
            finalUnrecPersons.insert(iter.key(),person2PointMap[iter.key()]);

    //fill unrec vector
    UnrecLists unrecLists;
    for(QList<PatternInfo>::const_iterator iter=patternInfoList.begin();iter!=patternInfoList.end();++iter){
        for(PatternInfo::PersonsType::const_iterator setIter=iter->persons.begin();setIter!=iter->unrecPersons.end();){
            if(setIter==iter->persons.end()){
                setIter=iter->unrecPersons.begin();
                if(setIter==iter->unrecPersons.end())
                    break;
            }
            if(!finalRecPersons.contains(setIter.key()) && !finalUnrecPersons.contains(setIter.key())){
                checkPointPresence(unrecLists,setIter.value(),(QImage*)0);
            }

            ++setIter;
        }
        for(PatternInfo::ImagesType::const_iterator unrecImageIter=iter->unrecImages.begin();unrecImageIter!=iter->unrecImages.end();++unrecImageIter){
            checkPointPresence(unrecLists,unrecImageIter->first,  &unrecImageIter->second);
        }
    }

    PatternInfo::ImagesType finalUnrecVector;
    for(UnrecLists::iterator unrecIter=unrecLists.begin();unrecIter!=unrecLists.end();++unrecIter){
        if(unrecIter->second.size()>=FOUND_COUNT_THRESHOLD){
            QVector<const QImage*>& unrecImages=unrecIter->second;
            for(QVector<const QImage*>::iterator iter=unrecImages.begin();iter!=unrecImages.end();){
                if(*iter==0)
                    iter=unrecImages.erase(iter);
                else
                    ++iter;
            }
            const QImage* center=FaceImage::get_center(unrecImages);
            //qDebug()<<"new size="<<unrecIter->second.size();
            finalUnrecVector.push_back(std::make_pair(unrecIter->first,(center==0)?QImage():*center));
        }
    }
    qDebug()<<"final rec="<<finalRecPersons<<" unrec="<<finalUnrecPersons<<finalUnrecVector.size();
    return PatternInfo(finalRecPersons,finalUnrecPersons,finalUnrecVector);
}

void RecMainWindow::saveFace(){
    //timerEvent(0);
    QVector<std::pair<QRect,QImage> > face_images=cvwidget->getFaces();
    for(QVector<std::pair<QRect,QImage> >::const_iterator iter=face_images.begin();iter!=face_images.end();++iter){
        const QImage& img=iter->second;
        FaceImage* inputImage=new FaceImage(&img);
        FaceImage* face=db.getClosest(inputImage);
        QString defaultPersonName=(face!=0)?face->personName:"";
        delete inputImage;
        QAddImageDialog* dlg=new QAddImageDialog(img,defaultPersonName,this);
        if(dlg->exec()==QDialog::Accepted){
            db.addImage(dlg->personName(),&img);
        }
        delete dlg;
        /*QString filename=QFileDialog::getSaveFileName(this,"",QDir::currentPath(),tr("Image file (.png *.jpg *.jpeg *.bmp)"));
        iter->save(filename);*/
    }
}



void RecMainWindow::exportDatabase(){
#ifdef USE_ORA_DATABASE
    DbReader dbReader;
    if(dbReader.isInitialized())
        dbReader.writeDirsIntoDatabase(db.getBaseDir());
#endif
}
void RecMainWindow::importDatabase(){
#ifdef USE_ORA_DATABASE
    DbReader dbReader;
    if(dbReader.isInitialized())
        dbReader.readFacesIntoDir(db.getBaseDir(),false);
#endif
}
