#ifndef QADDIMAGEDIALOG_H
#define QADDIMAGEDIALOG_H

#include <QDialog>
#include <QString>
#include <QLineEdit>

class QAddImageDialog : public QDialog
{
    Q_OBJECT 
private:
    const QImage& image;
    QLineEdit *editClassName;
public:
    QAddImageDialog(const QImage& img, QString defaultPersonName="person", QWidget* pwgt=0);
    QString personName(){
        return editClassName->text();
    }
public slots:
    void okClicked();
};

#endif // QADDIMAGEDIALOG_H
