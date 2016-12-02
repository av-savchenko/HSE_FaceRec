
#include "DEMTesting.h"

#include "FaceImage.h"
#include "TestDb.h"
#include "db.h"

#include "FacesData.h"

#include <windows.h>


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include <QtGui>



static void loadImages(std::string path,std::vector<FaceImage*>& faceImages){
        WIN32_FIND_DATAA ffd,ffd1;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	HANDLE hFindFile = INVALID_HANDLE_VALUE;
	
    qDebug() <<"start load";
    QDir baseDir(path.c_str());
    QStringList persons=baseDir.entryList(QDir::Dirs);
    foreach(QString personName, persons){
        if(personName.startsWith("."))
            continue;
        QDir personDir(baseDir.filePath(personName));
        QStringList photos=personDir.entryList(QDir::Files);
        foreach(QString photo, photos){
            QImage image(personDir.filePath(photo));
            if(!image.isNull()){
                //image->copy(
                faceImages.push_back(new FaceImage(&image,personName,photo));
            }
        }
    }
   qDebug() <<"end load";

}

void runOneTest(std::string prefix, FacesData& facesData, const vector<FaceImage*>& testImages, std::ofstream& resFile)
{
	double mean_error=0, std_error=0;
	double mean_time=0, std_time=0;
	double total_time=0;
	LARGE_INTEGER freq, start, end; 
	QueryPerformanceFrequency(&freq); 
	
        facesData.init();

	const int TESTS_COUNT=1;
	for(int t=0;t<TESTS_COUNT;++t){
		int errorsCount=0;
		QueryPerformanceCounter(&start); 
		for(vector<FaceImage*>::const_iterator iter=testImages.begin();iter!=testImages.end();++iter){
			FaceImage* test=*iter;
            FaceImage* bestImage=facesData.recognize(test);
            if(test->personName!=bestImage->personName){
				++errorsCount;
				//std::cout<<test->fileName<<" "<<faceImages[ind]->personName<<" "<<faceImages[ind]->distance(test)<<" "<<faceImages[ind]->distanceToTest<<std::endl;
			}
		}
		QueryPerformanceCounter(&end); 
		double error_rate=100.*errorsCount/testImages.size();
		double delta_microseconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart*1000000.0; 
		total_time+=delta_microseconds;

		mean_error+=error_rate;
		std_error+=error_rate*error_rate;

		delta_microseconds/=testImages.size();
		mean_time+=delta_microseconds;
		std_time+=delta_microseconds*delta_microseconds;
	}
	mean_error/=TESTS_COUNT;
	std_error=std_error/TESTS_COUNT-mean_error*mean_error;
	std_error=(std_error>0)?sqrt(std_error):0;
	
	mean_time/=TESTS_COUNT;
	std_time=std_time/TESTS_COUNT-mean_time*mean_time;
	std_time=(std_time>0)?sqrt(std_time):0;
	
	double avgDEMCheckedPercent=-1;

        qDebug()<<prefix.c_str()<<" error="<<mean_error<<'('<<std_error<<')'<<
			" time="<<(total_time/1000)<<" ms, rel="<<
                mean_time<<'('<<std_time<<") us";
	resFile<<prefix<<'\t'<<mean_error<<'('<<std_error<<')'<<'\t'<<mean_time<<'('<<std_time<<')'<<std::endl;
}

#define USE_PARALLEL_DEM

void testRecTime(){
    //fill_image();
    //return;
    vector<FaceImage*> dbImages;
    vector<FaceImage*> testImages;

#if DB_USED != USE_TEST_DB
    loadImages(DB,dbImages);
    loadImages(TEST,testImages);
#else
    qDebug()<<"start load";
    load_model_and_test_images(dbImages,testImages);
    qDebug()<<"loaded";
#endif
    FacesData facesData(dbImages);
    qDebug()<<"thread pool loaded";
    facesData.setImageCountToCheck(100);

    FaceImage* bestImage=0;
    double bestDist=100000,tmpDist;

    std::vector<FaceImage*>::iterator iter;

    int errorsCount=0;

    LARGE_INTEGER freq, start, end;
    double delta_milliseconds;

    QueryPerformanceFrequency(&freq);


    std::ofstream resFile("res.txt");
    resFile.imbue(std::locale());

    FacesData::nnMethod=Simple;
    runOneTest("Brute force", facesData, testImages, resFile);

    int high=NUM_OF_THREADS==1?dbImages.size():2*dbImages.size()/NUM_OF_THREADS;
    for(int imageCountToCheck=0;imageCountToCheck<=high;imageCountToCheck+=100)
    {
            facesData.setImageCountToCheck(imageCountToCheck);
            resFile<<imageCountToCheck<<std::endl;
            qDebug()<<imageCountToCheck;
#ifdef NEED_KD_TREE
            FacesData::nnMethod=KD_TREE;
            if(imageCountToCheck!=0)
                    runOneTest("BBF", facesData, testImages, resFile);
            //return;
#endif
#ifdef NEED_FLANN
            FacesData::nnMethod=FLANN;
            runOneTest("FLANN", facesData, testImages, resFile);
            //return;
#endif
    }
    resFile.close();
}
