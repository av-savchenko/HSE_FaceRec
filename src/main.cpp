
#include <QtGui>

#include <QUrl>
#include <QApplication>

#include "RecMainWindow.h"
#include "DbReader.h"
#include "DEMTesting.h"
#include "DnnFeatureExtractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    /*QTranslator translator;
    QString locale=QLocale::system().name();
    qDebug()<<locale;
    if(locale=="ru_RU"){
        qDebug()<<translator.load("faces_ru.qm",".");
        app.installTranslator(&translator);
    }
    qDebug()<<QDir::currentPath();*/

    RecMainWindow *mainWin = new RecMainWindow();
    mainWin->setWindowTitle(QObject::tr("HSE Face recognition"));
    mainWin->show();
    int retval=app.exec();
    delete mainWin;
    return retval;

}

