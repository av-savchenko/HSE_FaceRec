#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <string>
#include <chrono>
#include <functional>

using namespace std;
//g++ -std=c++11 ImageTesting.cpp -o ImageTesting;./ImageTesting


#include "db.h"
#include "db_features.h"


#ifdef QT_BUILD
#include <QtCore>
#include <QDebug>
#undef cout
#define cout qDebug()
#define print_endl
#else
#define print_endl cout<<endl;
#endif

int num_of_unreliable = 0;
const int reduced_features_count =
#ifdef USE_LCNN
        64;
#else
        256;
#endif
int recognize_image(const vector<ImageInfo>& dbImages, const ImageInfo& testImageInfo, int class_count){
        const int DIST_WEIGHT=100;
        int bestInd = -1;
        double bestDist = 100000, secondBestDist=100000;
        vector<double> distances(dbImages.size());
        vector<double> probabs(class_count);
        double max_probab,probab;
        for (int j = 0; j < dbImages.size(); ++j){
                distances[j] = testImageInfo.distance(dbImages[j], 0, reduced_features_count);
                probab = exp(-distances[j]*DIST_WEIGHT);
                if(probab>probabs[dbImages[j].classNo])
                    probabs[dbImages[j].classNo]=probab;
                if (distances[j] < bestDist){
                        secondBestDist = bestDist;
                        bestDist = distances[j];
                        bestInd = j;
                        max_probab=probab;
                }
        }

        int MAX_PROBABS_COUNT=5;//probabs.size();//10;
        nth_element(probabs.begin(),probabs.begin()+MAX_PROBABS_COUNT,probabs.end(), std::greater<double>());
        //std::sort(probabs.begin(),probabs.end(), std::greater<double>());
        double sum=0;
        for(int i=0;i<MAX_PROBABS_COUNT;++i){
            sum+=probabs[i];
        }
        max_probab /= sum;
        bool is_reliable =
                //true;
                false;
                //(bestDist / secondBestDist) < 0.75;
                //max_probab > 0.0008;//0.4 - 1000 weight
                //max_probab > 0.09;
               //max_probab > 0.0025;
                //max_probab > 0.24;
                //max_probab > 0.21;
//cout<<max_probab<<endl;
        if (!is_reliable){
                ++num_of_unreliable;
                bestInd = -1;
                bestDist = 100000;
                for (int j = 0; j < dbImages.size(); ++j){
                        distances[j] = (distances[j] * reduced_features_count +
                                testImageInfo.distance(dbImages[j], reduced_features_count)*(FEATURES_COUNT - reduced_features_count)) / FEATURES_COUNT;
                        if (distances[j] < bestDist){
                                bestDist = distances[j];
                                bestInd = j;
                        }
                }
        }
        return bestInd;
}

#if 0
//#define USE_OUTER
void testVerification(){
    ImagesDatabase trainImages;
    unordered_map<string, int> person2indexMap;
    loadImages(trainImages,PCA_CASIA_FEATURES_FILE_NAME, person2indexMap,true);

    int withinCount=0, outerCount=0;
    for (auto& identity : trainImages){
        if(identity.size()>1)
            withinCount+=identity.size();
        outerCount+=identity.size();
    }
    Mat mat_within_features(withinCount, FEATURES_COUNT, CV_32F);
    int ind = 0;

    //bayesian faces
    for (auto& identity : trainImages){
        int identity_size=identity.size();
        if(identity_size>1){
            for (int i=0;i<identity_size;++i){
                int other_i=i;
                while(i==other_i)
                    other_i=rand()%identity_size;
                for (int j = 0; j < FEATURES_COUNT; ++j){
                    mat_within_features.at<float>(ind, j) =identity[i][j]-identity[other_i][j];
                }
                ++ind;
            }
        }
    }
    const int num_of_inout_features=96;
    PCA pca(mat_within_features, Mat(), CV_PCA_DATA_AS_ROW, num_of_inout_features);
    Mat mat_projection_result=
            pca.project(mat_within_features);
            //mat_within_features;
    cout << "rows="<<mat_projection_result.rows << " cols=" << mat_projection_result.cols;
    cout<<"pca smallest EV="<<pca.eigenvalues.at<float>(0,pca.eigenvalues.rows-1)<<' '<<pca.eigenvalues.at<float>(0,0);
    Mat covar;
    mulTransposed(mat_projection_result,covar,true);
    //covar=mat_projection_result.t()*mat_projection_result;
    covar/=withinCount;
    covar+=Mat::eye(covar.cols,covar.rows,CV_32F)*0.9;
    cout<<"det="<<determinant(covar);
    Mat inv_covar=covar.inv();
#ifdef USE_OUTER
    inv_covar/=sqrt(determinant(covar));
#endif
    cout << "inv_covar rows="<<inv_covar.rows << " cols=" << inv_covar.cols;

#ifdef USE_OUTER
    ind=0;
    Mat mat_outer_features(outerCount, FEATURES_COUNT, CV_32F);
    for (int train_ind=0;train_ind< trainImages.size();++train_ind){
        auto& identity = trainImages[train_ind];
        int identity_size=identity.size();
        for (int i=0;i<identity_size;++i){
            int other_ind=train_ind;
            while(train_ind==other_ind)
                other_ind=rand()%trainImages.size();
            int other_i=rand()%trainImages[other_ind].size();
            auto& other_identity = trainImages[other_ind];

            for (int j = 0; j < FEATURES_COUNT; ++j){
                mat_outer_features.at<float>(ind, j) =identity[i][j]-other_identity[other_i][j];
            }
            ++ind;
        }
    }
    PCA pca_outer(mat_outer_features, Mat(), CV_PCA_DATA_AS_ROW, num_of_inout_features);
    mat_projection_result=pca_outer.project(mat_outer_features);
    mulTransposed(mat_projection_result,covar,true);
    covar/=outerCount;
    covar+=Mat::eye(covar.cols,covar.rows,CV_32F)*0.9;
    cout<<"det1="<<determinant(covar);
    Mat outer_inv_covar=covar.inv()/sqrt(determinant(covar));
    cout << "outer_inv_covar rows="<<outer_inv_covar.rows << " cols=" << outer_inv_covar.cols;
#endif

    ImagesDatabase origImages, totalImages;
    loadImages(origImages,FEATURES_FILE_NAME,person2indexMap);
    totalImages.resize(origImages.size());

    ind=0;
    Mat image_features(1, FEATURES_COUNT, CV_32F);
    for (auto& identity : origImages){
        totalImages[ind].resize(identity.size());
        for (int i=0;i<identity.size();++i){
            for (int j = 0; j < FEATURES_COUNT; ++j){
                image_features.at<float>(0, j) =identity[i][j];
            }
#ifdef USE_OUTER
            totalImages[ind][i].resize(2*num_of_inout_features);
#else
            totalImages[ind][i].resize(num_of_inout_features);
#endif
            Mat projection_image_features=
                    pca.project(image_features);
                    //image_features;
            for (int j = 0; j < num_of_inout_features; ++j){
                totalImages[ind][i][j]=projection_image_features.at<float>(0, j);
            }
#ifdef USE_OUTER
            projection_image_features=pca_outer.project(image_features);
            for (int j = 0; j < num_of_inout_features; ++j){
                totalImages[ind][i][num_of_inout_features+j]=projection_image_features.at<float>(0, j);
            }
#endif
        }
        ++ind;
    }
    const int TESTS = 1;
    std::vector<ImageInfo> dbImages, testImages;
    double total_time = 0;
    double totalTestsErrorRate = 0, errorRateVar = 0;
    for (int testCount = 0; testCount < TESTS; ++testCount)
    {
            int errorsCount = 0;
            getTrainingAndTestImages(totalImages, dbImages, testImages);

            auto t1 = chrono::high_resolution_clock::now();
            for (ImageInfo testImageInfo: testImages){
                int bestInd = -1;
                double bestDist = 100000;
                for (int j = 0; j < dbImages.size(); ++j){
                    double dist = 0;
                    for(int f1=0;f1<num_of_inout_features;++f1){
                        double diff_1=testImageInfo.features[f1]-dbImages[j].features[f1];
#ifdef USE_OUTER
                        double out_diff_1=testImageInfo.features[num_of_inout_features+f1]-
                                dbImages[j].features[num_of_inout_features+f1];
#endif
                        for(int f2=0;f2<num_of_inout_features;++f2){
                            double diff_2=testImageInfo.features[f2]-dbImages[j].features[f2];
                            double d1=inv_covar.at<float>(f1,f2)*diff_1*diff_2;
                            dist+=d1;
#ifdef USE_OUTER
                            double out_diff_2=testImageInfo.features[num_of_inout_features+f2]-
                                    dbImages[j].features[num_of_inout_features+f2];
                            double d2=outer_inv_covar.at<float>(f1,f2)*out_diff_1*out_diff_2;
                            //cout<<diff_1<<' '<<diff_2<<' '<<d1<<' '<<d2;
                            dist-=d2;
#endif
                        }
                    }
                    if (dist < bestDist){
                            bestDist = dist;
                            bestInd = j;
                    }
                }
                if (bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo){
                    ++errorsCount;
                }
            }
            auto t2 = chrono::high_resolution_clock::now();
            double rec_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/(1.0*testImages.size());
            total_time += rec_time;
            double errorRate = 100.*errorsCount / testImages.size();
            cout << "test=" << testCount << " error=" << errorRate << " rec_time(ms)=" << rec_time<<" dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
            print_endl;
    }
}
#else //joint bayesian
void testVerification(){    
    const int num_of_inout_features=256;//FEATURES_COUNT;
    ImagesDatabase totalImages;
    unordered_map<string, int> person2indexMap;
    int total_images_size=loadImages(totalImages,FEATURES_FILE_NAME,person2indexMap);
#if 0
    ImagesDatabase trainImages;
    loadImages(trainImages,PCA_CASIA_FEATURES_FILE_NAME, person2indexMap,true);

    int n = num_of_inout_features;
    Mat u(0,n, CV_64F);
    Mat SW(n, n, CV_64F);
    int within_count = 0;
    for (auto& identity : trainImages){
        int cur_size = identity.size();
        Mat cur(cur_size, n, CV_64F);
        for (int i = 0; i < cur_size; ++i)
            for (int j = 0; j < n; ++j)
                cur.at<double>(i, j) = identity[i][j];

        Mat cov, mu;
        calcCovarMatrix(cur, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        //cout << identity.first<<' '<<u.rows<<' '<<u.cols<<' '<<cov.rows << ' ' << cov.cols << ' ' << mu.rows << ' ' << mu.cols << '\n';
        u.push_back(mu);

        if (cur_size > 1){
            //cov = cov / (cur.rows - 1);
            within_count+=cur_size;
            cov/= cur_size - 1;
            cov+=Mat::eye(cov.cols,cov.rows,CV_64F)*0.5;
            SW += cov*cur_size/within_count;
        }
    }
    cout<<"u size="<<u.rows<<' '<<u.cols;
    Mat SU, mean_img;
    calcCovarMatrix(u, SU, mean_img, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    SU /= (u.rows - 1);
    SU+=Mat::eye(SU.cols,SU.rows,CV_64F)*0.5;

    Mat F = SW.inv();
    cout << "after all:" << within_count << ' ' <<  SU.at<double>(0, 0) << ' ' << SU.at<double>(1, 0) << ' ' <<  SW.at<double>(0, 0) << ' ' <<  SW.at<double>(1, 0);
    cout << "f:" << F.at<double>(0, 0) << ' ' << F.at<double>(1, 0) << ' ' << F.at<double>(0, 1) << ' ' << F.at<double>(1, 1) << '\n';
    Mat G = -(2 * SU + SW).inv() * SU*F;
    cout << "g:" << G.at<double>(0, 0) << ' ' << G.at<double>(1, 0) << ' ' << G.at<double>(0, 1) << ' ' << G.at<double>(1, 1);
    Mat A = (SU + SW).inv() - (F + G);


    cout<<"A="<<A.at<double>(0,0)<<' '<<A.at<double>(A.rows-1,A.cols-1);
    cout<<"G="<<G.at<double>(0,0)<<' '<<G.at<double>(G.rows-1,G.cols-1);
    cout<<"F="<<F.at<double>(0,0)<<' '<<F.at<double>(F.rows-1,F.cols-1);

    vector<double> xax(total_images_size);
    int ind=0;
    for(auto& identity : totalImages){
        for(auto& features:identity){
            xax[ind]=0;
            for (int i = 0; i < num_of_inout_features; ++i)
                for (int j = 0; j < num_of_inout_features; ++j)
                    xax[ind] += A.at<double>(i,j) * features[i] * features[j];
            ++ind;
        }
    }

#endif
    const int TESTS = 10;
    std::vector<ImageInfo> dbImages, testImages;
    double total_time = 0;
    double totalTestsErrorRate = 0, errorRateVar = 0;
    unordered_map<int,double> image_distances;
    cout<<"start testing";
    print_endl;
    for (int testCount = 0; testCount < TESTS; ++testCount)
    {
            getTrainingAndTestImages(totalImages, dbImages, testImages);

            int errorsCount = 0;
            auto t1 = chrono::high_resolution_clock::now();
            for (ImageInfo testImageInfo: testImages){
                int bestInd = -1;
                double bestDist = 100000;
                for (int j = 0; j < dbImages.size(); ++j){
                    double dist = 0;
                    int pair_ind;
                    if(testImageInfo.indexInDatabase<dbImages[j].indexInDatabase)
                        pair_ind=testImageInfo.indexInDatabase*total_images_size+dbImages[j].indexInDatabase;
                    else
                        pair_ind=testImageInfo.indexInDatabase+total_images_size*dbImages[j].indexInDatabase;

                    if(image_distances.find(pair_ind)==image_distances.end())
                    {
#if 0
                        dist = -xax[testImageInfo.indexInDatabase]-xax[dbImages[j].indexInDatabase];
                        for(int f1=0;f1<num_of_inout_features;++f1){
                            for(int f2=0;f2<num_of_inout_features;++f2){
                                dist+=2*G.at<double>(f1,f2)*testImageInfo.features[f1]*dbImages[j].features[f2];
                            }
                        }
#else
                        for(int f1=0;f1<num_of_inout_features;++f1){
                            dist+=(testImageInfo.features[f1]-dbImages[j].features[f1])*(testImageInfo.features[f1]-dbImages[j].features[f1]);
                        }
#endif
                        image_distances.insert(make_pair(pair_ind,dist));
                    }
                    dist=image_distances[pair_ind];
                    if (dist < bestDist){
                            bestDist = dist;
                            bestInd = j;
                    }
                }
                if (bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo){
                    ++errorsCount;
                }
            }
            auto t2 = chrono::high_resolution_clock::now();
            double rec_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/(1.0*testImages.size());
            total_time += rec_time;
            double errorRate = 100.*errorsCount / testImages.size();
            cout << "test=" << testCount << " error=" << errorRate << " rec_time(ms)=" << rec_time<<" dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
            print_endl;

            totalTestsErrorRate += errorRate;
            errorRateVar += errorRate * errorRate;
    }
    totalTestsErrorRate /= TESTS;
    total_time /= TESTS;
    errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));
    cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar;
    print_endl;
}
#endif

void testRecognition(){
        ImagesDatabase totalImages;
        unordered_map<string, int> person2indexMap;
#ifdef USE_PCA
        ImagesDatabase orig_database;
        loadImages(orig_database,FEATURES_FILE_NAME,person2indexMap);
        extractPCA(orig_database, totalImages);
#else
        //loadImages(totalImages,FEATURES_FILE_NAME,person2indexMap);
        ImagesDatabase orig_database;
        loadImages(orig_database, FEATURES_FILE_NAME,person2indexMap);
        for (auto& features:orig_database){
            if (features.size() > 1)
                totalImages.push_back(features);
        }
        cout<<"total size="<<totalImages.size();
#endif
        int num_of_classes=totalImages.size();
        const int TESTS = 10;
        std::vector<ImageInfo> dbImages, testImages;
        double total_time = 0;
        double totalTestsErrorRate = 0, errorRateVar = 0, totalRecall=0;
        for (int testCount = 0; testCount < TESTS; ++testCount){
                int errorsCount = 0;
                getTrainingAndTestImages(totalImages, dbImages, testImages);
                vector<int> testClassCount(num_of_classes),testErrorRates(num_of_classes);
                for (ImageInfo testImageInfo: testImages){
                    ++testClassCount[testImageInfo.classNo];
                }

                num_of_unreliable = 0;

                auto t1 = chrono::high_resolution_clock::now();
                for (ImageInfo testImageInfo: testImages){
                        int bestInd = recognize_image(dbImages, testImageInfo, totalImages.size());
                        if (bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo){
                                ++errorsCount;
                                ++testErrorRates[testImageInfo.classNo];
                        }
                }
                auto t2 = chrono::high_resolution_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/(1.0*testImages.size());

                double unreliable_ratio = 100.0*num_of_unreliable / testImages.size();

                double errorRate = 100.*errorsCount / testImages.size();
                double recall=0;
                for(int i=0;i<num_of_classes;++i){
                    if(testClassCount[i]>0)
                        recall+=100-100.*testErrorRates[i] / testClassCount[i];
                }
                recall/=num_of_classes;
                totalRecall+=recall;
                cout << "test=" << testCount << " error=" << errorRate<<" recall="<<recall << " unreliable=" << unreliable_ratio<<"% dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
                print_endl;

                totalTestsErrorRate += errorRate;
                errorRateVar += errorRate * errorRate;

        }
        totalTestsErrorRate /= TESTS;
        total_time /= TESTS;
        errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));
        totalRecall/=TESTS;
        cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar <<" time for 1 image (ms)="<<total_time<< " Recall="<<totalRecall;
        print_endl;
}


