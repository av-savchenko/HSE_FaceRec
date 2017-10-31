QT += gui \
network sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets multimedia multimediawidgets
TEMPLATE = app
TARGET = HSE_FaceRec
DEPENDPATH += .
DEFINES += DLIB_NO_GUI_SUPPORT
CONFIG += c++11 #-DNOMINMAX
QMAKE_CXXFLAGS_WARN_ON = -Wall -Wno-sign-compare -Wno-unused-variable

# INCLUDEPATH += C:\\face_recognition\\opencv2.3.1\\opencv\\build\\include
# win32:LIBS += -LC:\\face_recognition\\opencv2.3.1\\opencv\\build\\x86\\vc10\\lib \
# -lopencv_core231 -lopencv_objdetect231 -lopencv_imgproc231 -lopencv_highgui231
INCLUDEPATH += /Users/avsavchenko/anaconda/include/ /usr/local/Cellar/opencv3/3.0.0/include/ /usr/local/include /Users/avsavchenko/Documents/my_soft/github/caffe/distribute/include/ /Users/avsavchenko/Documents/my_soft/github/dlib
LIBS += -L/usr/local/Cellar/opencv3/3.0.0/lib -L/Users/avsavchenko/Documents/my_soft/github/caffe/distribute/lib \
    -L/Users/avsavchenko/anaconda/pkgs/gflags-2.2.0-1/lib -L/Users/avsavchenko/anaconda/pkgs/protobuf-3.1.0-py27_0/lib -L/Users/avsavchenko/anaconda/pkgs/hdf5-1.8.17-1/lib -L/Users/avsavchenko/anaconda/pkgs/lmdb-0.9.18-1/lib -L/Users/avsavchenko/anaconda/pkgs/openblas-0.2.19-2/lib -L/Users/avsavchenko/anaconda/pkgs/libgfortran-3.0.0-0/lib \
     -L/usr/lib \#-L/usr/local/lib \
     -L/usr/local/Cellar/boost/1.64.0_1/lib/ -L/usr/local/Cellar/glog/0.3.4/lib/  \
    -lglog \
    -lboost_system -lboost_filesystem \
    -lcaffe \
    -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab


SOURCES += main.cpp \
    imagedatabase.cpp \
    FaceImage.cpp \
    DbReader.cpp \
    AppSettings.cpp \
    CameraGrabber.cpp \
    DisplayImageDialog.cpp \
    Grabable.cpp \
    GrabableFactory.cpp \
    IpCameraGrabber.cpp \
    MediaPlayer.cpp \
    OpenCvWrapper.cpp \
    PatternInfo.cpp \
    QAddImageDialog.cpp \
    QOpenCVWidget.cpp \
    RecMainWindow.cpp \
    SingleImageGrabber.cpp \
    opencv_dir_processor.cpp \
    DnnFeatureExtractor.cpp \
    /Users/avsavchenko/Documents/my_soft/github/dlib/dlib/all/source.cpp
HEADERS += \
    imagedatabase.h \
    FaceImage.h \
    DbReader.h \
    AppSettings.h \
    CameraGrabber.h \
    DisplayImageDialog.h \
    Grabable.h \
    GrabableFactory.h \
    IpCameraGrabber.h \
    MediaPlayer.h \
    OpenCvWrapper.h \
    PatternInfo.h \
    QAddImageDialog.h \
    QOpenCVWidget.h \
    RecMainWindow.h \
    SingleImageGrabber.h \
    db.h \
    DnnFeatureExtractor.h
CONFIG += qt \
    release
    #debug
RESOURCES += images.qrc
#TRANSLATIONS += faces_ru.ts
