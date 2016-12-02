
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QtGui>

void process_dir(QString db, QString dest){
    cv::CascadeClassifier  cascade;
    if( !cascade.load(
                //"haarcascade_frontalface_default.xml"
                "lbpcascade_frontalface.xml"
                ) )
        qDebug()<<"--(!)Error loading\n";

    QDir baseDir(db);
    QDir destDir(dest);
#if 1
    QStringList photos=baseDir.entryList(QDir::Files);
    foreach(QString photo, photos){
        cv::Mat matImage=cv::imread(baseDir.filePath(photo).toStdString().c_str(),CV_LOAD_IMAGE_COLOR);
        if(! matImage.data )
            continue;
        cv::Mat img_gray;
        cv::cvtColor( matImage, img_gray, CV_BGR2GRAY );
        cv::equalizeHist( img_gray, img_gray );
        //qDebug()<<img_small.cols<<' '<<img_small.rows;
        std::vector<cv::Rect> faces;
        cascade.detectMultiScale( img_gray, faces,
                                  1.3, 2, cv::CASCADE_FIND_BIGGEST_OBJECT
                                  |cv::CASCADE_DO_ROUGH_SEARCH
                                  /*|cv::CASCADE_DO_CANNY_PRUNING*/
                                  |cv::CASCADE_SCALE_IMAGE
                                  ,
                                  cv::Size(16, 16));
        //qDebug()<<destDir.filePath(photo);
        cv::Mat croppedImage;
        if(faces.empty()){
            qDebug()<<"no faces at "<<baseDir.filePath(photo);
            matImage(cv::Rect(matImage.cols/5, matImage.rows/5,3*matImage.cols/5, 3*matImage.rows/5 )).copyTo(croppedImage);
            //QFile::copy(baseDir.filePath(photo), destDir.filePath(photo));
        }
        else{
            if(faces.size()>1)
                qDebug()<<"too many faces at "<<baseDir.filePath(photo);
            matImage(faces[0]).copyTo(croppedImage);
        }
        cv::imwrite(destDir.filePath(photo).toStdString().c_str(),croppedImage);
    }

#else
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
#endif

}
