#include "OpenCvWrapper.h"

#include <QtGui>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
using namespace cv;
OpenCvWrapper::OpenCvWrapper()
{   
    QFile f_cascade(":/lbpcascade_frontalface.xml");
    QTemporaryFile* tmp_cascade=QTemporaryFile::createNativeFile(f_cascade);
    tmp_cascade->setAutoRemove(true);
    qDebug()<<tmp_cascade->fileName();

    if( !cascade.load(tmp_cascade->fileName().toStdString().c_str()
                //"haarcascade_frontalface_default.xml"
                //"lbpcascade_frontalface.xml"
                //"/Users/avsavchenko/Documents/my_soft/face_rec_x64/lbpcascade_frontalface.xml"
                //"C:/Users/Andrey/Documents/faces/face_rec_new_1/lbpcascade_profileface.xml"
                //"C:/Users/Andrey/Documents/faces/face_rec_new_1/haarcascade_frontalface_default.xml"
                ) )
        qDebug()<<"--(!)Error loading\n";
    else
        qDebug()<<"ok";
    delete tmp_cascade;
}
OpenCvWrapper& OpenCvWrapper::Instance(){
    static OpenCvWrapper* openCvWrapper=0;
    if(!openCvWrapper){
        openCvWrapper=new OpenCvWrapper();
    }
    return *openCvWrapper;
}
QVector<QRect> OpenCvWrapper::detectFaceRects(const Mat&/*IplImage* */img,  bool oneFace){
    const double scale=1.3;
#if 0
    static IplImage *gray, *small_img;
    if(!gray || gray->width!=img->width || gray->height!=img->height){
        if(gray!=0){
            cvReleaseImage( &gray );
            cvReleaseImage( &small_img );
        }
        gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
        small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                             cvRound (img->height/scale)), 8, 1 );
    }
    cvCvtColor( img, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );
    //QTime myTimer;myTimer.start();

    cvClearMemStorage( storage );
    CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,
                                    1.3, 2, (oneFace?CV_HAAR_FIND_BIGGEST_OBJECT:0)
                                    |CV_HAAR_DO_ROUGH_SEARCH
                                    |CV_HAAR_DO_CANNY_PRUNING
                                    |CV_HAAR_SCALE_IMAGE
                                    ,
                                    cvSize(30, 30) );

   //qDebug()<<myTimer.elapsed();
    int size=(faces ? faces->total : 0);
    QVector<QRect> faceRects;
    for(int i = 0; i < size; i++ )
    {
        CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
        r->x = (int)((double)r->x * scale);
        r->y = (int)((double)r->y * scale);
        r->width = (int)((double)r->width * scale);
        r->height = (int)((double)r->height * scale);
        //qDebug()<<r->width;
        if(r->width>100)
            faceRects.push_back(QRect(r->x+r->width/8,r->y,r->width*3/4,r->height));
        else
            faceRects.push_back(QRect(r->x,r->y,r->width,r->height));
        //faceRects.push_back(QRect(r->x,r->y,r->width,r->height));
        //faceRects.push_back(QRect(r->x+r->width/8,r->y+r->height/16,r->width*3/4,r->height*7/8));
        //faceRects.push_back(QRect(r->x+r->width/8,r->y+r->height/8,r->width*3/4,r->height*7/8));
        //faceRects.push_back(QRect(r->x+r->width/6,r->y+r->height/6,r->width*2/3,r->height*2/3));
    }
    /*cvReleaseImage( &gray );
    cvReleaseImage( &small_img );*/

#else
    //QTime myTimer;myTimer.start();
    Mat matImage=img;//cv::cvarrToMat(img);
    Mat img_gray, img_small;
    cvtColor( matImage, img_gray, CV_BGR2GRAY );
    resize( img_gray, img_small,Size( round (img_gray.cols/scale),
                                      round (img_gray.rows/scale)) );
    equalizeHist( img_small, img_small );
    //qDebug()<<img_small.cols<<' '<<img_small.rows;
    std::vector<Rect> faces;
    QVector<QRect> faceRects;
#if 1
    cascade.detectMultiScale( img_small, faces,
                              1.3, 2, (oneFace?CASCADE_FIND_BIGGEST_OBJECT:0)
                              |CASCADE_DO_ROUGH_SEARCH
                              /*|CASCADE_DO_CANNY_PRUNING*/
                              |CASCADE_SCALE_IMAGE
                              ,
                              Size(30, 30));
#else
    std::vector<int> rejectLevels;
    vector<double> levelWeights;
    cascade.detectMultiScale( img_small, faces,rejectLevels, levelWeights,
                              1.3, 2, (oneFace?CASCADE_FIND_BIGGEST_OBJECT:0)
                              |CASCADE_DO_ROUGH_SEARCH
                              /*|CASCADE_DO_CANNY_PRUNING*/
                              |CASCADE_SCALE_IMAGE
                              ,
                              Size(30, 30),Size(),true );
    qDebug()<<"rejectLevels";
    for(int i=0;i<rejectLevels.size();++i)
        qDebug()<<rejectLevels[i]<<" ";
    qDebug()<<"\n";
    qDebug()<<"levelWeights";
    for(int i=0;i<levelWeights.size();++i)
        qDebug()<<levelWeights[i]<<" ";
    qDebug()<<"\n";
    qDebug()<<"faces";
    for(int i=0;i<faces.size();++i)
        qDebug()<<faces[i].x<<' '<<faces[i].y<<' '<<faces[i].width<<' '<<faces[i].height<<" ";
    qDebug()<<"\n";
#endif
    for(int i = 0; i < faces.size(); i++ )
    {
        /*if(levelWeights[i]<0)
            continue;*/
        Rect& r = faces[i];
        r.x = (int)((double)r.x * scale);
        r.y = (int)((double)r.y * scale);
        r.width = (int)((double)r.width * scale);
        r.height = (int)((double)r.height * scale);
        //qDebug()<<r->width;
        if(r.width>100)
            faceRects.push_back(QRect(r.x+r.width/8,r.y,r.width*3/4,r.height));
        else
            faceRects.push_back(QRect(r.x,r.y,r.width,r.height));
        //faceRects.push_back(QRect(r->x,r->y,r->width,r->height));
        //faceRects.push_back(QRect(r->x+r->width/8,r->y+r->height/16,r->width*3/4,r->height*7/8));
        //faceRects.push_back(QRect(r->x+r->width/8,r->y+r->height/8,r->width*3/4,r->height*7/8));
        //faceRects.push_back(QRect(r->x+r->width/6,r->y+r->height/6,r->width*2/3,r->height*2/3));
    }
    //qDebug()<<myTimer.elapsed();
#endif

    return faceRects;
}
QVector<QRect> OpenCvWrapper::detectFaceRects(const QImage* srcImage, bool oneFace){
    /*IplImage *cvimage = cvCreateImageHeader(cvSize(srcImage->width(), srcImage->height()), IPL_DEPTH_8U, 4);
    cvimage->imageData = (char*)srcImage->bits();
    cv::Mat image=cv::cvarrToMat(cvimage);*/

    cv::Mat cvimage(srcImage->height(),srcImage->width(),CV_8UC4,const_cast<uchar *>(srcImage->bits()),srcImage->bytesPerLine());
    QVector<QRect> res=detectFaceRects(cvimage,oneFace);
    //cvReleaseImageHeader(&cvimage);
    return res;
}
QVector<std::pair<QRect,QImage> > OpenCvWrapper::detectFaces(const QVector<QRect>& faceRects, const QImage& image){
    QVector<std::pair<QRect,QImage> > res;
    for(QVector<QRect>::const_iterator iter=faceRects.begin();iter!=faceRects.end();++iter){
        QPainterPath clipPath;
        clipPath.addEllipse(0,0,(*iter).width(),(*iter).height());
        QImage face((*iter).width(),(*iter).height(),image.format());
        QPainter painter;
        painter.begin(&face);
        painter.fillRect(0,0,(*iter).width(),(*iter).height(),Qt::white);
        painter.setClipPath(clipPath);
        painter.drawImage(QPoint(0,0),image,*iter);
        painter.end();
        res.push_back(std::make_pair(*iter,face));
    }
    return res;
}



//noise
QImage OpenCvWrapper::putFilter(QImage* srcImage, FilterType filterType, int level)
{
#if 1
    cv::Mat cvimage(srcImage->height(),srcImage->width(),CV_8UC4,const_cast<uchar *>(srcImage->bits()),srcImage->bytesPerLine());
    QImage qt_img((const uchar *) cvimage.data, cvimage.cols, cvimage.rows, cvimage.step, QImage::Format_RGB888);
    return qt_img.copy();
#else
    IplImage *image = cvCreateImageHeader(cvSize(srcImage->width(), srcImage->height()), IPL_DEPTH_8U, 4);
    image->imageData = (char*)srcImage->bits();
    if (filterType == Noise)
    {
        CvRNG rng = cvRNG(0xffffffff);
        int count = 0;
        for( int y=0; y<image->height; y++ ) {
            uchar* ptr = (uchar*) (image->imageData + y*image->widthStep);
            for( int x=0; x<image->width; x++ ) {
                if(cvRandInt(&rng)%100 >= (100-level))
                {
                    ptr[image->nChannels*x] = cvRandInt(&rng)%255;
                    ptr[image->nChannels*x+1] = cvRandInt(&rng)%255;
                    ptr[image->nChannels*x+2] = cvRandInt(&rng)%255;
                    count++;
                }
            }
        }
    }
    if (filterType == Smooth)
    {
        switch (level)
        {
        case 0:
            cvSmooth(image, image, CV_BLUR_NO_SCALE, 3, 3);
            break;
        case 1:
            cvSmooth(image, image, CV_BLUR, 3, 3);
            break;
        case 2:
            cvSmooth(image, image, CV_GAUSSIAN, 3, 3);
            break;
        case 3:
            cvSmooth(image, image, CV_MEDIAN, 3, 3);
            break;
        }
    }
    if (filterType == Erode)
    {
        IplConvKernel* Kern = cvCreateStructuringElementEx(level*2+1, level*2+1, level, level, CV_SHAPE_ELLIPSE);
        cvErode(image, image, Kern, 1);
    }
    if (filterType == Dilate)
    {
        IplConvKernel* Kern = cvCreateStructuringElementEx(level*2+1, level*2+1, level, level, CV_SHAPE_ELLIPSE);
        cvDilate(image, image, Kern, 1);
    }
    if (filterType == Morphology)
    {
        IplConvKernel* Kern = cvCreateStructuringElementEx(1*2+1, 1*2+1, 1, 1, CV_SHAPE_ELLIPSE);
        IplImage* Temp = 0;
        Temp = cvCreateImage(cvSize(image->width, image->height) , IPL_DEPTH_8U, 1);
        switch (level)
        {
        case 0:
            cvMorphologyEx(image, image, Temp, Kern, CV_MOP_OPEN, 1);
            break;
        case 1:
            cvMorphologyEx(image, image, Temp, Kern, CV_MOP_CLOSE, 1);
            break;
        case 3:
            cvMorphologyEx(image, image, Temp, Kern, CV_MOP_GRADIENT, 1);
            break;
        case 4:
            cvMorphologyEx(image, image, Temp, Kern, CV_MOP_TOPHAT, 1);
            break;
        case 5:
            cvMorphologyEx(image, image, Temp, Kern, CV_MOP_BLACKHAT, 1);
            break;
        }
    }
    if (filterType == Threshold)
    {
        switch (level)
        {
        case 0:
            cvThreshold(image, image, 50, 250, CV_THRESH_BINARY);
            break;
        case 1:
            cvThreshold(image, image, 50, 250, CV_THRESH_BINARY_INV);
            break;
        case 2:
            cvThreshold(image, image, 50, 250, CV_THRESH_TRUNC);
            break;
        case 3:
            cvThreshold(image, image, 50, 250, CV_THRESH_TOZERO);
            break;
        case 4:
            cvThreshold(image, image, 50, 250, CV_THRESH_TOZERO_INV);
            break;
        }
    }
    if (filterType == Sharpness)
    {
        float kernel[9];
        kernel[0]=-0.1;
        kernel[1]=-0.1;
        kernel[2]=-0.1;
        kernel[3]=-0.1;
        kernel[4]=2;
        kernel[5]=-0.1;
        kernel[6]=-0.1;
        kernel[7]=-0.1;
        kernel[8]=-0.1;
        CvMat kernel_matrix=cvMat(3,3,CV_32FC1,kernel);
        cvFilter2D(image, image, &kernel_matrix, cvPoint(-1,-1));
    }
    if (filterType == Brightness)
    {
        float kernel[9];
        kernel[0]=-0.1;
        kernel[1]=0.2;
        kernel[2]=-0.1;
        kernel[3]=0.2;
        kernel[4]=3;
        kernel[5]=0.2;
        kernel[6]=-0.1;
        kernel[7]=0.2;
        kernel[8]=-0.1;
        CvMat kernel_matrix=cvMat(3,3,CV_32FC1,kernel);
        cvFilter2D(image, image, &kernel_matrix, cvPoint(-1,-1));
    }
    if (filterType == Blackout)
    {
        float kernel[9];
        kernel[0]=-0.1;
        kernel[1]=0.1;
        kernel[2]=-0.1;
        kernel[3]=0.1;
        kernel[4]=0.5;
        kernel[5]=0.1;
        kernel[6]=-0.1;
        kernel[7]=0.1;
        kernel[8]=-0.1;
        CvMat kernel_matrix=cvMat(3,3,CV_32FC1,kernel);
        cvFilter2D(image, image, &kernel_matrix, cvPoint(-1,-1));
    }
    if (filterType == Sobel)
    {
        int xorder = 0;
        int yorder = 1;
        int aperture = 3;
        cvSobel(image, image, xorder, yorder, aperture);
        cvConvertScale(image, image);
    }

    if (filterType == Laplace)
    {
        int aperture = 3;
        cvLaplace(image, image, aperture);
        cvConvertScale(image, image);
    }
    QImage qt_img =
            QImage ((uchar*) image->imageData, image->width, image->height, QImage::Format_RGB888 ).rgbSwapped();
    cvReleaseImageHeader(&image);
    return qt_img;
#endif
}
