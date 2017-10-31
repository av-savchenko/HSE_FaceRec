#include "FaceImage.h"
#include "DnnFeatureExtractor.h"

#include <QtGui>
#include <opencv2/imgproc.hpp>

static void segmentGamma(int* pixels, int width,int height, int colorCount){
    //segment gamma
    const int K = 256;
    const int aMax = K - 1;
    double gamma;

    // create a lookup table for the mapping function
    int Fgc[K];

    const int PART_W=3,PART_H=1;
    int pixelsInW=width/PART_W;
    int pixelsInH=height/PART_H;
    int i,j,di,dj,ind;
    for(int k=0;k<1;++k)
        for(i=0;i<PART_H;++i)
            for(j=0;j<PART_W;++j){
                int sum=0,count=0;
                int minDi=qMax(-pixelsInH/4,-i*pixelsInH);
                int maxDi=qMin(5*pixelsInH/4,height-i*pixelsInH);
                for(di=minDi;di<maxDi;++di){
                //for(int di=0;di<pixelsInH;++di){
                    int minDj=qMax(-pixelsInW/4,-j*pixelsInW);
                    int maxDj=qMin(5*pixelsInW/4,width-j*pixelsInW);
                    for(int dj=minDj;dj<maxDj;++dj){
                    //for(int dj=0;dj<pixelsInW;++dj){for(di=0;di<pixelsInH;++di)
                        ind=((i*pixelsInH+di)*width+j*pixelsInW+dj)*colorCount+k;
                        if(pixels[ind]!=255){
                            sum+=pixels[ind];
                            ++count;
                        }
                    }
                }
                gamma=log(.5)/log((double)sum/(count*256));
                for(ind=0;ind<K;++ind)
                    Fgc[ind]=-1;
                for(di=0;di<pixelsInH;++di)
                    for(dj=0;dj<pixelsInW;++dj){
                    ind=((i*pixelsInH+di)*width+j*pixelsInW+dj)*colorCount+k;
                    if(Fgc[pixels[ind]]<0){
                        // scale to $[0,1]$
                        double bb = pow((double) pixels[ind] / aMax,gamma);	// gamma function \indexmeth{pow}
                        // scale back to [0,255]
                        if(bb>1)
                            qDebug()<<"bb="<<bb;
                        Fgc[pixels[ind]] = (int) floor(0.5+bb * aMax);

                    }
                    pixels[ind]=Fgc[pixels[ind]];
                }
            }
}
static void gammaLine(int* pixels, int width,int height, int colorCount){
        int i,j,l,ind;
        const int max=255-1;
        int histo[max+1];
        double desiredCdf[max+1];
        int lookup[max+2];

        for(i=0;i<=max;++i)
            desiredCdf[i]=1/(1+exp(-0.05*(i-max/2)));
        for(int k=0;k<1;++k)
            for(j=0;j<width;++j){
               for(l=0;l<=max;++l)
                   histo[l]=0;
                for(i=0;i<height;++i){
                    ind=(i*width+j)*colorCount+k;
                    ++histo[pixels[ind]];
                }
                for(l=1;l<=max;++l){
                    histo[l]+=histo[l-1];
                }

                ind=0;
                for(l=0;l<=max;++l){
                    if(histo[l]<=(desiredCdf[ind]*height))
                        lookup[l]=ind;
                    else{
                        while(ind<max && histo[l]>(desiredCdf[ind]*height))
                            ++ind;
                        if((desiredCdf[ind]*height-histo[l])>(histo[l]-desiredCdf[ind-1]*height))
                            lookup[l]=ind--;
                        else
                            lookup[l]=ind;
                    }
                }
                lookup[max+1]=255;
                for(i=0;i<height;++i){
                    ind=(i*width+j)*colorCount+k;
                    pixels[ind]=lookup[pixels[ind]];
                }
            }
}
static void preprocessPixelsLine(int* pixels, int width,int height, int colorCount){
        int i,j,ind;
        for(int k=0;k<1;++k)
            for(j=0;j<width;++j){
                int min=255,max=0;
                for(i=0;i<height;++i){
                    ind=(i*width+j)*colorCount+k;
                    if(pixels[ind]!=255 && pixels[ind]>max)
                        max=pixels[ind];
                    if(pixels[ind]<min)
                        min=pixels[ind];
                }
                if(min!=max){
                    for(i=0;i<height;++i){
                        ind=(i*width+j)*colorCount+k;
                        if(pixels[ind]!=255)
                            pixels[ind]=(pixels[ind]-min)*255/(max-min);
                    }
                }
            }
}
static void median(int* pixels, int width,int height, int colorCount){
    int *prev_pixels=new int[width*height*colorCount];
    int i,j,ind;
    for(j=0;j<width;++j){
        for(i=0;i<height;++i){
            ind=(i*width+j)*colorCount+0;
            prev_pixels[ind]=pixels[ind];
        }
    }
    const int MEDIAN_DIFF=1;
    QVector<int> neighbors;
    for(int k=0;k<1;++k)
        for(i=0;i<height;++i){
            for(j=1;j<width-1;++j){
                ind=(i*width+j)*colorCount+k;
                neighbors.clear();
                for(int i1=i-MEDIAN_DIFF;i1<=i+MEDIAN_DIFF;++i1)
                    if(i1>=0 & i1<height)
                        for(int j1=j-MEDIAN_DIFF;j1<=j+MEDIAN_DIFF;++j1)
                            if(j1>=0 & j1<width)
                                neighbors.push_back(prev_pixels[i1*width+j1]);

                neighbors.push_back(pixels[ind+k]);
                std::sort(neighbors.begin(),neighbors.end());
                pixels[ind+k]=neighbors[neighbors.size()/2];            }
        }
   delete[] prev_pixels;
}
static int weights[]={
    0,1,1,0,
    1,2,2,1,
    1,2,2,1,
    0,1,1,0
};
FaceImage::FaceImage(const QImage* image,const QString& pName,const QString& fName,bool isUnrecAdded):
        //image(img->copy(img->width()/8,img->height()/8,img->width()*7/8,img->height()*7/8)),
        personName(pName),
        fileName(fName),
        isUnrec(isUnrecAdded),
        age("")
{
    if(image->isNull())
        return;
    const int FRACTION=8;
    //QImage img=image->copy(image->width()/FRACTION,image->height()/FRACTION,image->width()*(FRACTION-1)/FRACTION,image->height()*(FRACTION-1)/FRACTION);
    //QImage img=image->copy(image->width()/FRACTION,0,image->width()*(FRACTION-2)/FRACTION,image->height());
    int width=image->width();
    int height=image->height();
#ifndef USE_DNN_FEATURES
    if((width%POINTS_IN_W)!=0)
        width=(width/POINTS_IN_W)*POINTS_IN_W;
    if((height%POINTS_IN_H)!=0)
        height=(height/POINTS_IN_H)*POINTS_IN_H;
#else
    width = height = 128;
#endif
    const QImage img=image->scaled(width,height);

    int* pixels=new int[width*height*COLORS_COUNT];
    int i,j,k;
    for(i=0;i<width;++i)
        for(j=0;j<height;++j){
            QColor color=QColor(img.pixel(i,j));
            int colorPart[3];
            if(true){
                color.getRgb(colorPart+2,colorPart+1,colorPart+0,0);
                colorPart[0]=(int)(0.11*colorPart[0]+.56*colorPart[1]+.33*colorPart[2]);
            }else{
                color.getHsv(colorPart+1,colorPart+2,colorPart+0);
                colorPart[1]=colorPart[1]*255/360;
                //colorPart[1]=0;
            }
            for(k=0;k<COLORS_COUNT;++k)
                pixels[(j*width+i)*COLORS_COUNT+k]=colorPart[k];
        }


    cv::Mat cvimage(image->height(),image->width(),CV_8UC4,const_cast<uchar *>(image->bits()),image->bytesPerLine());

    init(cvimage,pixels, width, height);
    delete[] pixels;
}
FaceImage::FaceImage(cv::Mat& image,const QString& pName):
        personName(pName),age("")
{
    /*int i,j,k;

    int* pixels=new int[image.rows*image.cols*COLORS_COUNT];
    CvScalar s;
    for(i=0;i<img->width;++i)
        for(j=0;j<img->height;++j){
            s=cvGet2D(img,j,i);
            int color=(int)(0.11*s.val[0]+.56*s.val[1]+.33*s.val[2]);
            for(k=0;k<COLORS_COUNT;++k)
                pixels[(j*img->width+i)*COLORS_COUNT+k]=color;
        }
    init(image, pixels, img->width, img->height);
    delete[] pixels;*/
}
void FaceImage::init(cv::Mat& image, int *pixels, int width, int height){
    int i,j,k,di,dj,x,y,superInd,ind;
    float sum;
#ifdef USE_GRADIENT_ANGLE
    median(pixels,width,height,1);

    int histoGranularityW=width/POINTS_IN_W;
    int histoGranularityH=height/POINTS_IN_H;

    //const double PI=4*atan(1.);

    for(k=0;k<COLORS_COUNT;++k)
        for(i=0;i<POINTS_IN_H;++i)
            for(j=0;j<POINTS_IN_W;++j){
                superInd=i*POINTS_IN_W+j;
                QVector<float> mags;
                for(di=0;di<histoGranularityH;++di){
                    x=i*histoGranularityH+di;
                    if(x>=height-1){
                        break;
                    }
                    for(dj=0;dj<histoGranularityW;++dj){
                        y=j*histoGranularityW+dj;
                        if(y>=width-1){
                            break;
                        }
                        ind=x*width+y;
                        float dfdx=(pixels[(x+1)*width+y]-pixels[x*width+y+1]);
                        float dfdy=(pixels[x*width+y]-pixels[(x+1)*width+y+1]);
                        float mag=fabs(dfdx)>fabs(dfdy)?fabs(dfdx):fabs(dfdy);
                        mags.push_back(mag);
                    }
                }
                qSort(mags);
                float maxMag=mags[(mags.size()-1)*9/10]+0.1;

                for(ind=0;ind<HISTO_SIZE;++ind){
                    histos[k][i][j][ind]=0;
                }
                sum=0;
                for(di=0;di<histoGranularityH;++di){
                    x=i*histoGranularityH+di;
                    if(x>=height-1){
                        break;
                    }
                    for(dj=0;dj<histoGranularityW;++dj){
                        y=j*histoGranularityW+dj;
                        if(y>=width-1){
                            break;
                        }
                        ind=x*width+y;
                        float dfdx=(pixels[(x+1)*width+y]-pixels[x*width+y+1]);
                        float dfdy=(pixels[x*width+y]-pixels[(x+1)*width+y+1]);
                        //float magSqr=dfdx*dfdx+dfdy*dfdy;
                        float mag=fabs(dfdx)>fabs(dfdy)?fabs(dfdx):fabs(dfdy)+0.1;
                        int angleInd=0;
                        /*
                        float angle=atan2(dfdy,dfdx);
                        if(angle<0)
                            angle+=2*PI;
                        int angleInd=(int)(angle/(2*PI)*HISTO_SIZE);
                        */
                        int bit0=(fabs(dfdy)<fabs(dfdx))?1:0;
                        int bit1=(dfdx<0)?2:0;
                        int bit2=(dfdy<0)?4:0;
                        angleInd=bit0+bit1+bit2;

                        float curIncrement=(mag>=maxMag)?1:(mag/maxMag);
                        histos[k][i][j][angleInd]+=curIncrement;
                        //qDebug()<<curIncrement;
                    }
                }
                //qDebug()<<sum;
                sum=0;
                for(ind=0;ind<HISTO_SIZE;++ind){
                    //histos[k][i][j][ind]+=0.2;
                    sum+=histos[k][i][j][ind];
                }
                if(sum>0){
                    for(ind=0;ind<HISTO_SIZE;++ind){
                        histos[k][i][j][ind]/=sum;
                    }
                }
            }
#elif defined(USE_DNN_FEATURES)
    cv::Mat dst1, dst2;
    cv::cvtColor(image, dst1, CV_RGB2GRAY);
    cv::medianBlur(dst1, dst2, 3);
    featureVector.resize(FEATURES_COUNT);
    DnnFeatureExtractor::GetInstance()->extractFeatures(dst2, &featureVector[0]);
#ifdef DETECT_AGE
    age=DnnFeatureExtractor::GetInstance()->detect_age(dst2);
#endif
#else
    //segmentGamma(pixels,width,height,COLORS_COUNT);
    segmentGamma(pixels,width,height,1);
    //gammaLine(pixels,width,height,COLORS_COUNT);
    //preprocessPixelsLine(pixels,width,height,COLORS_COUNT);

    int histoGranularityW=width/((POINTS_IN_W+1)/2);
    int histoGranularityH=height/((POINTS_IN_H+1)/2);

    for(k=0;k<COLORS_COUNT;++k)
        for(i=0;i<POINTS_IN_H;++i)
            for(j=0;j<POINTS_IN_W;++j){
                superInd=i*POINTS_IN_W+j;
                sum=HISTO_SIZE;
                for(ind=0;ind<HISTO_SIZE;++ind){
                    histos[k][superInd][ind]=1;
                }
                /*if((i==0 && j==0) ||
                   (i==0 && j==(POINTS_IN_W-1)) ||
                   (i==(POINTS_IN_H-1) && j==0)||
                   (i==(POINTS_IN_H-1) && j==(POINTS_IN_W-1)))
                    continue;*/
                for(di=0;di<histoGranularityH;++di){
                    x=i*histoGranularityH/2+di;
                    //x=i*histoGranularityH+di;
                    if(x>=height){
                        x=height-1;
                    }
                    for(dj=0;dj<histoGranularityW;++dj){
                        y=j*histoGranularityW/2+dj;
                        //y=j*histoGranularityW+dj;
                        if(y>=width){
                            y=width-1;
                        }
                        ind=x*width+y;
                        histos[k][superInd][pixels[ind]/NUM_OF_PIXELS_IN_ONE_SLOT]+=1;
                        ++sum;
                    }
                }
                for(ind=0;ind<HISTO_SIZE;++ind){
                    histos[k][superInd][ind]/=sum;
                }
            }
#endif
}

#if 1
inline float fast_sqrt(float x)
{
    unsigned int i = *(unsigned int*) &x;
    // adjust bias
    i  += 127 << 23;
    // approximation of square root
    i >>= 1;
    return *(float*) &i;
}
float FaceImage::distance(const FaceImage* rhs){
    float res=0;
#ifdef USE_DNN_FEATURES
    const float* search_features=rhs->getFeatures();
    const float* features=getFeatures();
    for(int k=0;k<FEATURES_COUNT;++k){
        res+=fabs(features[k]-search_features[k]);
    }
    return res/FEATURES_COUNT;
#else
    const int DELTA=1;
    int iMin,iMax,jMin,jMax;
    for(int k=0;k<COLORS_COUNT;++k){
        for(int i=0;i<POINTS_IN_H;++i){
            iMin=i>=DELTA?i-DELTA:0;
            iMax=i+DELTA;
            if(iMax>=POINTS_IN_H)
                iMax=POINTS_IN_H-1;
            for(int j=0;j<POINTS_IN_W;++j){
                jMin=j>=DELTA?j-DELTA:0;
                jMax=j+DELTA;
                if(jMax>=POINTS_IN_W)
                    jMax=POINTS_IN_W-1;
                float minSum=1000000;
                for(int i2=iMin;i2<=iMax;++i2){
                    for(int j2=jMin;j2<=jMax;++j2){
                        float curSum=0;
                        for(int ind=0;ind<HISTO_SIZE;++ind){
                            float d1=histos[k][i][j][ind];
                            float d2=rhs->histos[k][i2][j2][ind];
                            //curSum+=fabs(d1-d2);
                            if((d1+d2)>0)
                                curSum+=(d1-d2)*(d1-d2)/(d1+d2);
                        }
                        if(minSum>curSum){
                            minSum=curSum;
                        }
                    }
                }
                res+=fast_sqrt(minSum);
            }
        }
    }
    return res/(COLORS_COUNT*POINTS_IN_H*POINTS_IN_W);
#endif
}
#else
float FaceImage::distance(const FaceImage* rhs){
    float res=0;
#ifdef USE_GRADIENT_ANGLE
    const int DELTA=1;
    for(int k=0;k<COLORS_COUNT;++k){
        for(int i=0;i<POINTS_IN_H;++i){
            for(int j=0;j<POINTS_IN_W;++j){
                float minSum=10000;
                for(int di=-DELTA;di<=DELTA;++di){
                    if((i+di)<0 || (i+di)>=POINTS_IN_H)
                        continue;
                    for(int dj=-DELTA;dj<=DELTA;++dj){
                        if((j+dj)<0 || (j+dj)>=POINTS_IN_W)
                            continue;
                        float curSum=0;
                        for(int ind=0;ind<HISTO_SIZE;++ind){
                            float d1=histos[k][i][j][ind];
                            float d2=rhs->histos[k][i+di][j+dj][ind];
                            curSum+=fabs(d1-d2);
                        }
                        if(minSum>curSum)
                            minSum=curSum;
                    }
                }
                res+=minSum;
            }
        }
    }
    return res/(COLORS_COUNT*POINTS_IN_H*POINTS_IN_W);
#else
    static float curHisto[HISTO_SIZE];
    static float distResults[POINTS_IN_W*POINTS_IN_H];
    const int missedCount=0;
    for(int k=0;k<COLORS_COUNT;++k){
        for(int i=0;i<HISTO_COUNT;++i){
            float minSum=10000;
            int bestShiftInd=0;
            for(int shiftInd=-1;shiftInd<=1;++shiftInd){
                shift(histos[k][i],curHisto,shiftInd);
                float tmp=0;
                for(int j=0;j<HISTO_SIZE;++j){
    #ifdef USE_KL
                    tmp+=histos[k][i][j]*log(curHisto[j]/rhs->histos[k][i][j]);
    #else
                    tmp+=/*weights[i]**/fabs(curHisto[j]-rhs->histos[k][i][j]);
    #endif
                    //tmp+=fabs(histos[k][i][j]-rhs->histos[k][i][j]);
                }
                //qDebug()<<tmp;
                if(tmp<minSum){
                    minSum=tmp;
                    bestShiftInd=shiftInd;
                }
            }
            distResults[i]=minSum;
        }
        //std::sort(distResults,distResults+HISTO_COUNT);
        //qDebug()<<distResults[0];
        for(int i=0;i<(HISTO_COUNT-missedCount);++i)
            res+=distResults[i];
    }
/*qDebug()<<res<<' '<<HISTO_SIZE<<' '<<(POINTS_IN_W*POINTS_IN_H)<<' '<<COLORS_COUNT;
for(int j=0;j<HISTO_SIZE;++j)
    qDebug()<<histos[0][0][j];
qDebug()<<' ';
for(int j=0;j<HISTO_SIZE;++j)
    qDebug()<<rhs->histos[0][0][j];*/
#ifndef USE_KL
    res/=HISTO_SIZE;
#endif
    return res/((HISTO_COUNT-missedCount)*COLORS_COUNT);
#endif
}
#endif

#ifndef USE_DNN_FEATURES
void FaceImage::shift(float* histo, float* resHisto, int shiftInd) {
    int i,ind;
    for(i=0;i<HISTO_SIZE;++i)
        resHisto[i]=0;
    for(i=0;i<HISTO_SIZE;++i){
        ind=i+shiftInd;
        if(ind>=0 && ind<HISTO_SIZE)
            resHisto[ind]=histo[i];
    }
    if(shiftInd>0)
        for(i=1;i<=shiftInd;++i)
            resHisto[HISTO_SIZE-1]+=histo[HISTO_SIZE-i];
    else if(shiftInd<0)
        for(i=0;i < -shiftInd;++i)
            resHisto[0]+=histo[i];
}
#endif

const QImage* FaceImage::get_center(const QVector<const QImage*>& images){
    const int MAX_DIST=1000;
    int centroid=-1;
    int len=images.size();
    if(len>0){
        FaceImage** faces=new FaceImage*[len];
        for(int i=0;i<len;++i)
            faces[i]=new FaceImage(images[i]);
        float* distances=new float[len*len];
        for(int i=0;i<len;++i)
            for(int j=0;j<len;++j)
                distances[i*len+j]=(i==j)?0:images[i]->isNull()?MAX_DIST:faces[i]->distance(faces[j]);

        float minDist=MAX_DIST;
        float curDist;
        for(int i=0;i<len;++i){
            curDist=0;
            for(int j=0;j<len;++j){
                curDist+=distances[i*len+j];
            }
            if(curDist<minDist){
                minDist=curDist;
                centroid=i;
            }
        }

        delete[] distances;
        for(int i=0;i<len;++i)
            delete faces[i];
        delete[] faces;
    }
    return centroid==-1?0:images[centroid];
}
