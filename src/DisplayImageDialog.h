#ifndef DISPLAYIMAGEDIALOG_H
#define DISPLAYIMAGEDIALOG_H

#include <QDialog>

class DisplayImageDialog : public QDialog
{
public:
    DisplayImageDialog(const QString& date, const QStringList& imageFiles, QWidget* pwgt=0);
};

#endif // DISPLAYIMAGEDIALOG_H
