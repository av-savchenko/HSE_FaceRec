#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H

#include "FaceImage.h"
#include <QVector>
#include <QDir>
#include <QString>


class ImageDatabase
{
private:
    QVector<FaceImage*> faceImages;
    QDir baseDir;
public:
    ImageDatabase();
    ~ImageDatabase();
    void addImage(const QString& personName, const QImage* image);
    QString addUnrecognized(const QImage& image);
    FaceImage* getClosest(FaceImage* inputImage,double *distance=0,double *distDiff=0);
    QString getPersonFile(const QString& personName);
    QDir getBaseDir(){return baseDir;}
};

#endif // IMAGEDATABASE_H
