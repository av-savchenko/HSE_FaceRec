#include "imagedatabase.h"
#include <QtCore>
#include <QImage>
#include <QThread>

const QString UNRECOGNIZED_FOLDER="unrecognized";
const QString UNRECOGNIZED_PREFIX="unrec_";

ImageDatabase::ImageDatabase()
{
    baseDir=QDir::home();
    baseDir.mkdir("database");
    baseDir.cd("database");
    QStringList persons=baseDir.entryList(QDir::Dirs);
    foreach(QString personName, persons){
        if(personName.startsWith("."))
            continue;
        QDir personDir(baseDir.filePath(personName));
        QStringList photos=personDir.entryList(QDir::Files);
        foreach(QString photo, photos){
            QImage image(personDir.filePath(photo));
            if(!image.isNull()){
                //image->copy(
                faceImages.push_back(new FaceImage(&image,personName,photo));
            }
        }
    }
    //qDebug()<<faceImages.size();
}
ImageDatabase::~ImageDatabase()
{
    for(QVector<FaceImage*>::iterator iter=faceImages.begin();iter!=faceImages.end();++iter)
        delete *iter;
}
void ImageDatabase::addImage(const QString& personName, const QImage* image){
    baseDir.mkdir(personName);
    QDir personDir(baseDir.filePath(personName));
    int size=personDir.entryList(QDir::Files).size();
    QString fileName;
    QTextStream(&fileName)<<(size+1)<<".png";
    image->save(personDir.filePath(fileName),"png");
    FaceImage* faceImage=new FaceImage(image,personName,fileName);
    faceImages.push_back(faceImage);
}
QString ImageDatabase::addUnrecognized(const QImage& image){
    baseDir.mkdir(UNRECOGNIZED_FOLDER);
    QDir unrecDir(baseDir.filePath(UNRECOGNIZED_FOLDER));
    int size=unrecDir.entryList(QDir::Dirs).size();
    QString dirName;
    QTextStream(&dirName)<<UNRECOGNIZED_PREFIX<<(size-1);
    unrecDir.mkdir(dirName);
    unrecDir.cd(dirName);
    QString fileName=unrecDir.filePath("1.png");
    image.save(fileName,"png");
    FaceImage* faceImage=new FaceImage(&image,dirName,fileName,true);
    faceImages.push_back(faceImage);
    return dirName;
}
QString ImageDatabase::getPersonFile(const QString& personName){
    QDir curDir(baseDir);
    if(personName.startsWith(UNRECOGNIZED_PREFIX))
        curDir.cd(UNRECOGNIZED_FOLDER);
    QDir personDir(curDir.filePath(personName));
    QString res="";
    QStringList files;
    if(personDir.exists() && (files=personDir.entryList(QDir::Files),!files.isEmpty())){
        res=personDir.filePath(files.at(0));
    }
    return res;
}

namespace{
    QMutex mutex;
    QWaitCondition newDist;

    QMap<QString,double> class2MinDistMap;
    double min_dist;
    FaceImage* closestPerson;
    int threadCount;

    void setDist(FaceImage* dbImage,double dist){
         QString personName=dbImage->personName;
         QMutexLocker locker(&mutex);
         if(!class2MinDistMap.contains(personName) || dist<class2MinDistMap[personName])
             class2MinDistMap[personName]=dist;
         if(dist<min_dist){
            min_dist=dist;
            closestPerson=dbImage;
        }
         if(--threadCount==0)
            newDist.wakeOne();
    }
    void waitForAllThreads(){
        QMutexLocker locker(&mutex);
        while (threadCount>0)
            newDist.wait(&mutex);
    }
}
class DistanceCalculatorTask : public QRunnable
 {
public:
    DistanceCalculatorTask( FaceImage* input,FaceImage* dbImage):inputImage(input),databaseImage(dbImage){
    }

     void run()
     {
        double dist=inputImage->distance(databaseImage);
        setDist(databaseImage,dist);
     }

 private:
     FaceImage* inputImage;
     FaceImage* databaseImage;
 };


FaceImage* ImageDatabase::getClosest(FaceImage* inputImage,double *distance/*=0*/,double *distDiff/*=0*/){
    min_dist=1000;
    closestPerson=0;
    /*
    QMap<QString,double> class2MinDistMap;
    for(QVector<FaceImage*>::iterator iter=faceImages.begin();iter!=faceImages.end();++iter){
        double dist=inputImage->distance(*iter);
        QString personName=(*iter)->personName;
        if(!class2MinDistMap.contains(personName) || dist<class2MinDistMap[personName])
            class2MinDistMap[personName]=dist;

        //qDebug()<<dist<<(*iter)->personName;
        if(dist<min_dist){
            min_dist=dist;
            closestPerson=(*iter);
        }
    }
    */
    class2MinDistMap.clear();
    threadCount=faceImages.size();
    for(QVector<FaceImage*>::iterator iter=faceImages.begin();iter!=faceImages.end();++iter){
        QThreadPool::globalInstance()->start(new DistanceCalculatorTask(inputImage,*iter));
    }
    waitForAllThreads();

    if(distance)
        *distance=min_dist;

    if(distDiff){
        if(class2MinDistMap.size()>1){
            QList<double> distances=class2MinDistMap.values();
            qSort(distances);
            *distDiff=(*(++distances.begin())-min_dist);
        }else
            *distDiff=-1;
    }
    //qDebug()<<((++class2MinDistMap.begin()).value()-min_dist);
    qDebug()<<class2MinDistMap;

    /*
    if(false && !class2MinDistMap.isEmpty()){
        double sum=0;
        QMap<QString,double>::iterator iter;
        for(iter=class2MinDistMap.begin();iter!=class2MinDistMap.end();++iter)
            sum+=iter.value();
        double entropy=0;
        for(iter=class2MinDistMap.begin();iter!=class2MinDistMap.end();++iter)
            entropy+=-log(iter.value()/sum)*exp(-iter.value()/sum);
        qDebug()<<entropy;
    }*/
    return closestPerson;
}
