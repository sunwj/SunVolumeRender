//
// Created by 孙万捷 on 16/6/2.
//

#include "AddAreaLightDialog.h"
#include "ui_AddAreaLightDialog.h"

AddAreaLightDialog::AddAreaLightDialog(QWidget *parent):
QDialog(parent),
ui(new Ui::AddAreaLightDialog)
{
    ui->setupUi(this);
}

AddAreaLightDialog::~AddAreaLightDialog()
{
    delete ui;
}

QString AddAreaLightDialog::GetLightName() const
{
    return ui->lineEdit_lightName->text();
}
