# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.resize(840, 520)
        widget.setMinimumSize(QtCore.QSize(840, 520))
        widget.setMaximumSize(QtCore.QSize(840, 520))
        self.label_camera = QtWidgets.QLabel(widget)
        self.label_camera.setEnabled(True)
        self.label_camera.setGeometry(QtCore.QRect(200, 60, 600, 400))
        self.label_camera.setMinimumSize(QtCore.QSize(600, 400))
        self.label_camera.setMaximumSize(QtCore.QSize(600, 400))
        self.label_camera.setText("")
        self.label_camera.setObjectName("label_camera")
        #self.graphicsView = QtWidgets.QGraphicsView(widget)
        #self.graphicsView.setGeometry(QtCore.QRect(200, 60, 600, 400))
        #self.graphicsView.setMinimumSize(QtCore.QSize(600, 400))
        #self.graphicsView.setMaximumSize(QtCore.QSize(600, 400))
        #self.graphicsView.setObjectName("graphicsView")
        #self.layoutWidget = QtWidgets.QWidget(widget)
        #self.layoutWidget.setGeometry(QtCore.QRect(40, 430, 281, 101))
        #self.layoutWidget.setObjectName("layoutWidget")
        #self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        #self.gridLayout.setContentsMargins(0, 0, 0, 0)
        #self.gridLayout.setObjectName("gridLayout")
        self.layoutWidget1 = QtWidgets.QWidget(widget)
        self.layoutWidget1.setGeometry(QtCore.QRect(40, 60, 120, 200))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        #self.gridLayout_2.setGeometry(QtCore.QRect(40,60,120,200))
        self.btn_local_camera = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_local_camera.setObjectName("btn_local_camera")
        self.gridLayout_2.addWidget(self.btn_local_camera, 0, 0, 1, 1)

        self.btn_from_video= QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_from_video.setObjectName("btn_web_camera")
        self.gridLayout_2.addWidget(self.btn_from_video, 1, 0, 1, 1)

        self.btn_get_faces = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_get_faces.setObjectName("btn_get_faces")
        self.gridLayout_2.addWidget(self.btn_get_faces, 2, 0, 1, 1)

        self.btn_delete_face = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_delete_face.setObjectName("btn_delete_face")
        self.gridLayout_2.addWidget(self.btn_delete_face ,3, 0, 1, 1)

        self.btn_train_classifier = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_train_classifier.setObjectName("btn_train_classifier")
        self.gridLayout_2.addWidget(self.btn_train_classifier, 4, 0, 1, 1)

        self.btn_close = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_close.setObjectName("btn_close")
        self.gridLayout_2.addWidget(self.btn_close, 5, 0, 1, 1)

        self.textEdit = QtWidgets.QTextEdit(widget)
        self.textEdit.setGeometry(QtCore.QRect(40, 300, 120, 100))
        self.textEdit.setObjectName("textEdit")
        #self.layoutWidget1.addWidget(self.textEdit, 1, 0, 1, 1)

        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "Form"))
        self.textEdit.setHtml(_translate("widget", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">说明：</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">1.初次使用软件，请确保model文件夹存有训练好的权重</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">2.初次使用请新建人脸数据，名字必须为英文字符</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">3.欧式距离越大，代表越不相似，越小代表越相似</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">4.每次点击</span><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">人脸识别</span><span style=\" font-size:8pt;\">前都必须先</span><span style=\" font-size:8pt; font-weight:600; text-decoration: underline;\">获取人脸</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt; font-weight:600;\">步骤：</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">1.打开本地摄像头或者网络摄像头</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">2.点击获取人脸</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">3.点击人脸识别</span></p></body></html>"))
        self.btn_local_camera.setText(_translate("widget", "打开本地摄像头"))
        self.btn_from_video.setText(_translate("widget", "打开本地视频"))
        self.btn_close.setText(_translate("widget", "关闭程序"))
        self.btn_get_faces.setText(_translate("widget", "添加人脸数据"))
        self.btn_train_classifier.setText(_translate("widget", "重新训练分类器"))
        self.btn_delete_face.setText(_translate("widget", "删除人脸数据"))

