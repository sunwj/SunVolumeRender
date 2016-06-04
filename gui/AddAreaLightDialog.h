//
// Created by 孙万捷 on 16/6/2.
//

#ifndef SUNVOLUMERENDER_ADDAREALIGHTDIALOG_H
#define SUNVOLUMERENDER_ADDAREALIGHTDIALOG_H

#include <QDialog>

namespace Ui{
class AddAreaLightDialog;
};

class AddAreaLightDialog: public QDialog
{
    Q_OBJECT
public:
    explicit AddAreaLightDialog(QWidget* parent = 0);
    ~AddAreaLightDialog();

    QString GetLightName() const;

private:
    Ui::AddAreaLightDialog* ui;
};


#endif //SUNVOLUMERENDER_ADDAREALIGHTDIALOG_H
