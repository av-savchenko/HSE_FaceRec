#ifndef DBREADER_H
#define DBREADER_H

#define USE_ORA_DATABASE

#ifdef USE_ORA_DATABASE
#include <QSqlDatabase>

class QString;
class QDir;

class DbReader
{
private:
    bool initialized;
    QSqlDatabase db;
    bool requestCredentials(QString& host, QString& dbName, QString& username,QString& password);
public:
    DbReader();
    ~DbReader();
    bool isInitialized(){return initialized;}

    void writeOneDirIntoDatabase(const QDir& dir);
    void writeDirsIntoDatabase(const QDir& dir);
    void readFacesIntoDir(QDir dir,bool detectFaces);
};
#endif //USE_ORA_DATABASE

#endif // DBREADER_H
