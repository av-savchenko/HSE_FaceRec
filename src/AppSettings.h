#ifndef APPSETTINGS_H
#define APPSETTINGS_H

#include <QObject>
#include <QSettings>

class AppSettings: public QObject
{
Q_OBJECT
private:
    QSettings settings;
public:
    AppSettings();
    bool storeUnrecognized();
    void setStoreUnrecognized(bool storeUnrecognized);
    bool addUnrecognizedToDatabase();
    void setAddUnrecognizedToDatabase(bool addUnrecognizedToDatabase);

    public slots:
    void showDialog();
};

#endif // APPSETTINGS_H
