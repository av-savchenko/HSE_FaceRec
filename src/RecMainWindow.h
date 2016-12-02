#ifndef RECMAINWINDOW_H_
#define RECMAINWINDOW_H_

#include "QOpenCVWidget.h"
#include "ImageDatabase.h"
#include "PatternInfo.h"
#include "AppSettings.h"
#include "Grabable.h"

#include <QMainWindow>
#include <QString>
#include <QList>
#include <memory>

class QTableWidget;
class QWidget;
class QTimerEvent;
class QAction;

class RecMainWindow : public QMainWindow
{
    Q_OBJECT
    private:
        QOpenCVWidget *cvwidget;
        std::auto_ptr<Grabable> grabber;
        QTableWidget *nameTable;
        QString lastRecognized;
        ImageDatabase db;
        AppSettings appSettings;
        int timerId;

        QList<PatternInfo> patternInfoList;
        PatternInfo detectedPersons;

        PatternInfo getDetectedPersons();
    public:
        RecMainWindow(QWidget *parent=0);
      //  QFile data;
         
    protected:
        void timerEvent(QTimerEvent*);
        void closeEvent(QCloseEvent *event);

    public slots:
        void saveFace();
        void exportDatabase();
        void importDatabase();
        void openImageViewer(int row, int column);
        void changeCamera(QAction*);
        // мое
        void toWrite();
        //void update();
};


#endif /*RECMAINWINDOW_H_*/
