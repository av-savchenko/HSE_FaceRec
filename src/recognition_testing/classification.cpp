#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <chrono>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

/*
 *
 * */

typedef double FEATURE_TYPE;

const FEATURE_TYPE PI=atan(1.0)*4;

class Feature_vector{
public:
    Feature_vector(const vector<FEATURE_TYPE>& fv, const vector<string>& df, double out):
      features(fv),discrete_features(df), output(out)
    {
    }
    vector<FEATURE_TYPE> features;
    vector<string> discrete_features;
    double output;
};
ostream& operator<<(ostream& os, const Feature_vector& feature){
    os<<"cn="<<feature.output<<": ";
    for(int i=0;i<feature.features.size();++i){
        os<<feature.features[i]<<' ';
    }
    return os;
}

template<typename map_type>
class key_iterator : public map_type::iterator
{
public:
    typedef typename map_type::iterator map_iterator;
    typedef typename map_iterator::value_type::first_type key_type;

    key_iterator(const map_iterator& other) : map_type::iterator(other) {}

    key_type& operator *()
    {
        return map_type::iterator::operator*().first;
    }
};

// helpers to create iterators easier:
template<typename map_type>
key_iterator<map_type> key_begin(map_type& m)
{
    return key_iterator<map_type>(m.begin());
}
template<typename map_type>
key_iterator<map_type> key_end(map_type& m)
{
    return key_iterator<map_type>(m.end());
}


#if 0
#define NORMALIZE(ind)  \
    FEATURE_TYPE val=(stdValues[fi]!=0)?0.9*(tmp_dataset[ind].features[fi]-avgValues[fi])/stdValues[fi]:0; \
    val= \
        /*(maxValues[fi]==minValues[fi])?0.0:(dataset[ind].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;*/ \
        (1-exp(-val))/(1+exp(-val)); \
    if(val<-1) \
        val=-1; \
    else if (val>1) \
        val=1;
#elif 0
#define NORMALIZE(ind)  \
    FEATURE_TYPE val=(maxValues[fi]==minValues[fi])?0.0:(dataset[ind].features[fi]-avgValues[fi])/(maxValues[fi]-minValues[fi]);
#elif 0
#define NORMALIZE(ind)  \
    FEATURE_TYPE val=(stdValues[fi]!=0)?0.1*(tmp_dataset[ind].features[fi]-avgValues[fi])/stdValues[fi]:0;
#else
#define NORMALIZE(ind)    FEATURE_TYPE val=(tmp_dataset[ind].features[fi]-avgValues[fi]);
#endif
    /*cout<<dataset[training_ind].features[fi]<<' '<<val<<' '<<normVal<<' '<<avgValues[fi]<<' '<<stdValues[fi]<<'\n';*/


//#define REGRESSION

#define USE_PROJECTION 1
#define USE_PNN 2
#define USE_KNN 3
#define USE_SVM 4

//#define CLASSIFIER USE_PROJECTION
//#define CLASSIFIER USE_PNN
//#define CLASSIFIER USE_KNN
#define CLASSIFIER USE_SVM

//#define DEBUG_PNN

#if CLASSIFIER==USE_PROJECTION
#define USE_FEJER 1
#define USE_HERMITT 2
#define KERNEL USE_FEJER
//#define KERNEL USE_HERMITT

//#define CALC_M

#elif CLASSIFIER==USE_PNN
//#define KMEANS_CLUSTERS 5 //10 //
#endif

#define NO_PCA_FEATURES 256

int num_of_classes, num_of_features, num_of_discrete_features, num_of_cont_features;
int num_of_cont_features_orig;

vector<Feature_vector> dataset, tmp_dataset;
vector<vector<map<string, double> > > discrete_class_histos;
vector <vector<int> > indices;
vector<FEATURE_TYPE> prior_probabs;
vector<vector<int> > training_set;
vector<int> test_set;
vector<FEATURE_TYPE> minValues, maxValues, avgValues, stdValues, weights;

#if CLASSIFIER==USE_PROJECTION

int M;
vector<float> a;
vector<int> num_of_a_coeffs;
#elif CLASSIFIER==USE_SVM
Ptr<SVM> svmClassifier;
#endif


#ifdef QT_BUILD
#include <QtCore>
#include <QDebug>
#undef cout
#define cout qDebug()
#define print_endl
#else
#define print_endl cout<<endl;
#endif

void load_dataset(){
    ifstream inF(
#ifdef REGRESSION
        //"/Users/avsavchenko/Documents/shared/UCI/winequality-red_new.txt"
        "/Users/avsavchenko/Documents/shared/UCI/winequality-white_new.txt"
#else
        //"/Users/avsavchenko/Documents/shared/UCI/iris_new.txt"
        //"/Users/avsavchenko/Documents/shared/UCI/wine_new.txt"
        //"C:/Users/avsavchenko/Documents/shared/UCI/adult_test_new.txt"
        //"/Users/avsavchenko/Documents/shared/UCI/adult_data_all.txt"
        //"/Users/avsavchenko/Documents/shared/UCI/adult_data_new.txt"
        //"/Users/avsavchenko/Documents/shared/UCI/winequality-red_new.txt"
        "/Users/avsavchenko/Documents/shared/UCI/Skin_NonSkin.txt"
#endif
        );

    inF>>num_of_features;
#ifdef REGRESSION
    num_of_classes=1;
#else
    inF>>num_of_classes;
    map<string,int> class_names;
    string class_name;
    for(int i=0;i<num_of_classes;++i){
        inF>>class_name;
        class_names[class_name]=i;
        cout<<class_name.c_str();
        print_endl;
    }
#endif

    inF>>num_of_discrete_features;
    num_of_cont_features=num_of_features-num_of_discrete_features;
    cout<<num_of_classes<<' '<<num_of_features<<' '<<num_of_cont_features<<' '<<num_of_discrete_features;
    print_endl;

    indices.resize(num_of_classes);
    vector<int> discrete_pattern_indices(num_of_discrete_features);
    discrete_class_histos.resize(num_of_discrete_features);
    int discrete_pattern_index;
    for(int i=0;i<num_of_discrete_features;++i){
        inF>>discrete_pattern_index;
        discrete_pattern_indices[i]=discrete_pattern_index;
        cout<<discrete_pattern_indices[i]<<' ';

        discrete_class_histos[i].resize(num_of_classes);
    }
    print_endl;

    while(!inF.eof()){
        vector<FEATURE_TYPE> features(num_of_cont_features);
        vector<string> discrete_features(num_of_discrete_features);
        int cont_ind=0, discrete_ind=0;
        for(int i=0;i<num_of_features;++i){
            if(find(discrete_pattern_indices.begin(),discrete_pattern_indices.end(),i)!=discrete_pattern_indices.end()){
                string feature;
                inF>>feature;
                if(inF.eof())
                    break;
                discrete_features[discrete_ind++]=feature;
            }
            else{
                FEATURE_TYPE feature;
                inF>>feature;
                if(inF.eof())
                    break;
                features[cont_ind++]=feature;
            }
        }
        if(inF.eof())
            break;
        int class_ind;
        double output;
#ifdef REGRESSION
        class_ind=0;
        inF>>output;
#else
        inF>>class_name;
        class_ind=class_names[class_name];
        output=class_ind;
#endif
        indices[class_ind].push_back(dataset.size());
        FEATURE_TYPE sum=0;
        for(int i=0;i<num_of_cont_features;++i)
            sum+=features[i]*features[i];
        sum=sqrt(sum);
        for(int i=0;i<num_of_cont_features;++i)
            //features[i]=(sum>0)?features[i]/sum:0;
            features[i]=features[i]/128.0-1;

        dataset.push_back(Feature_vector(features,discrete_features, output));

        for(int i=0;i<num_of_discrete_features;++i){
            ++discrete_class_histos[i][class_ind][discrete_features[i]];
        }

        //cout<<dataset.size()<<' '<<class_name<<'\n';
    }
    inF.close();
    cout<<dataset.size();
    print_endl;
    /*cout << dataset[0].features[0] << ' ' << dataset[0].features[num_of_cont_features-1];
    cout << dataset[dataset.size()-1].features[0] << ' ' << dataset[dataset.size()-1].features[num_of_cont_features-1];*/
}

#include "db.h"
void load_image_dataset(){
    num_of_discrete_features=0;
    num_of_cont_features=FEATURES_COUNT;///2;
    num_of_features=num_of_cont_features+num_of_discrete_features;

    ifstream ifs(FEATURES_FILE_NAME);
    if (ifs){
        int total_images = 0;
        map<string, int> person2indexMap;
        while (ifs){
                std::string fileName, personName, feat_str;
                if (!getline(ifs, fileName))
                        break;
                if (!getline(ifs, personName))
                        break;
                if (!getline(ifs, feat_str))
                        break;
                //cout << fileName.c_str() << ' ' << personName.c_str() << '\n';
                personName.erase(0, personName.find_first_not_of(" \t\n\r\f\v"));
#ifdef USE_CALTECH
                if(personName.find("BACKGROUND_Google")!=string::npos ||
                        personName.find("257.clutter")!=string::npos)
                        continue;
#endif
                if (person2indexMap.find(personName) == person2indexMap.end()){
#if defined (USE_LFW) && defined(USE_CASIA)
                    if(person2indexMap.size()>=1000)
                        ;//break;
#endif
                    person2indexMap.insert(std::make_pair(personName, person2indexMap.size()));
                }
                vector<FEATURE_TYPE> features(num_of_cont_features);
                vector<string> discrete_features(num_of_discrete_features);
                istringstream iss(feat_str);
                //of << fileName << endl << personName << endl;
                FEATURE_TYPE sum=0;
                for (int i = 0; i < FEATURES_COUNT; ++i){
                    FEATURE_TYPE feature;
                    iss >> feature;
#if 1
                    sum+=feature*feature;
#else
                    sum+=abs(feature);
#endif
                    if(i<num_of_cont_features)
                        features[i]=feature;
                }
#if 1
                sum=sqrt(sum);
#endif
                for (int i = 0; i < num_of_cont_features; ++i){
                    features[i]/=sum;
                }
                dataset.push_back(Feature_vector(features,discrete_features, person2indexMap[personName]));

                ++total_images;
        }
        ifs.close();
        num_of_classes=person2indexMap.size();
        indices.resize(num_of_classes);
        for(int i=0;i<dataset.size();++i){
            int class_ind=(int)(dataset[i].output);
            indices[class_ind].push_back(i);
        }
        cout<<num_of_classes<<' '<<num_of_features<<' '<<num_of_cont_features<<' '<<num_of_discrete_features<<' '<<total_images;
        print_endl;
    }
}

double classify(int test_ind, int total_training_size);

void testClassification(){
    //load_dataset();
    load_image_dataset();
    num_of_cont_features_orig=num_of_cont_features;

    for(int i=0;i<num_of_discrete_features;++i){
        set<string> values;
        for(int class_ind=0;class_ind<num_of_classes;++class_ind){
            values.insert(key_begin(discrete_class_histos[i][class_ind]), key_end(discrete_class_histos[i][class_ind]));
        }
        for(int class_ind=0;class_ind<num_of_classes;++class_ind){
            for(set<string>::iterator iter=values.begin();iter!=values.end();++iter){
                if(discrete_class_histos[i][class_ind].find(*iter)==discrete_class_histos[i][class_ind].end()){
                    discrete_class_histos[i][class_ind].insert(std::pair<string,double>(*iter,0.5));
                }
            }
            double sum=0;
            for(map<string, double>::iterator iter=discrete_class_histos[i][class_ind].begin();iter!=discrete_class_histos[i][class_ind].end();++iter){
                sum+=iter->second;
            }
            for(map<string, double>::iterator iter=discrete_class_histos[i][class_ind].begin();iter!=discrete_class_histos[i][class_ind].end();++iter){
                iter->second/=sum;
            }
        }
    }
    /*cout<<dataset[0]<<'\n';
    cout<<dataset[dataset.size()-1]<<'\n';*/
#ifdef KMEANS_CLUSTERS
    double* distances=0;

    /*distances=new double[dataset.size()*dataset.size()];
    memset(distances,0, sizeof(double)*dataset.size()*dataset.size());
    for(int d1=0;d1<dataset.size()-1;++d1){
        for(int d2=d1+1;d2<dataset.size();++d2){
            double dist=0;
            for(int fi=0;fi<num_of_cont_features;++fi){
                dist+=(dataset[d1].features[fi]-dataset[d2].features[fi])*(dataset[d1].features[fi]-dataset[d2].features[fi]);
            }
            dist/=num_of_cont_features;
            distances[d1*dataset.size()+d2]=distances[d2*dataset.size()+d1]=dist;
        }
    }*/
#endif

    prior_probabs.resize(num_of_classes);
    training_set.resize(num_of_classes);

    minValues.resize(num_of_cont_features);
    maxValues.resize(num_of_cont_features);
    avgValues.resize(num_of_cont_features);
    stdValues.resize(num_of_cont_features);
    weights.resize(num_of_cont_features);

#ifdef REGRESSION
    const int TEST_COUNT=10;
#else
    const int TEST_COUNT = 1;//10;
    const int max_fraction = 10;
#endif
    int num_of_tests=TEST_COUNT;

    ofstream fres("classification_res.txt");
#if 1
#ifdef USE_CALTECH
    //for(double fraction=0.025;fraction<=0.15;fraction+=0.025){
    for(double fraction=0.05;fraction<=0.3;fraction+=0.05){
    //for(double fraction=FRACTION;fraction<=FRACTION;fraction+=0.1){
#elif defined(USE_LFW)
    //for(double fraction=FRACTION;fraction<=FRACTION;fraction+=0.1){
    for(double fraction=0.5;fraction<=0.5;fraction+=0.1){
#else
    //for(double fraction=0.00025;fraction<=0.004;fraction+=0.00025){
    for(double fraction=0.02;fraction<=0.2;fraction+=0.02){
#endif
        double totalErrorRate=0, totalErrorSquare=0,totalRecall=0;
        double total=0;

        for(int shuffle_num=0;shuffle_num<1;++shuffle_num){
            for(int test_num=0;test_num<num_of_tests;++test_num){
                test_set.clear();
                test_set.reserve(dataset.size()*(1-fraction));
                for(int i=0;i<num_of_classes;++i){
                    std::random_shuffle ( indices[i].begin(), indices[i].end() );
                    int end=ceil(fraction*indices[i].size());
#ifdef USE_CALTECH
                    //end=30;
#endif
                    if(end==0 && !indices[i].empty())
                        end=1;
                    else if (end>=indices[i].size())
                        end=indices[i].size();
                    //++end;
                    training_set[i].clear();
                    training_set[i].assign(indices[i].begin(), indices[i].begin()+end);
                    test_set.insert(test_set.end(),indices[i].begin()+end, indices[i].end());
                }
#if defined (USE_LFW) && defined(USE_CASIA)
                //test_set.resize(500);
#endif
#elif 0
    num_of_tests=1;
    int fraction=1;
    test_set.clear();
    for(int i=0;i<num_of_classes;++i)
        training_set[i].clear();
    const int tr_size=30162;
    for(int i=0;i<dataset.size();++i)
        if(i<tr_size)
            training_set[(int)dataset[i].output].push_back(i);
        else
            test_set.push_back(i);
    {
        double totalErrorRate=0, totalErrorSquare=0,totalRecall=0;
        double total=0;
        {
            {
#else
    for (int fraction = max_fraction; fraction >= 2; --fraction){
        double totalErrorRate=0, totalErrorSquare=0,totalRecall=0;
        double total=0;

        num_of_tests=fraction*TEST_COUNT;
        for(int i=0;i<num_of_classes;++i){
            std::sort ( indices[i].begin(), indices[i].end() );
        }
        for(int shuffle_num=0;shuffle_num<TEST_COUNT;++shuffle_num){
            for(int test_num=0;test_num<fraction;++test_num){
                test_set.clear();
                test_set.reserve(dataset.size()/fraction);
                for(int i=0;i<num_of_classes;++i){
                    training_set[i].clear();
                    int start=test_num*indices[i].size()/fraction;
                    int end=(test_num+1)*indices[i].size()/fraction;
                    if(end>indices[i].size())
                        end=indices[i].size();
                    for(int j=0;j<indices[i].size();++j){
                        bool isInRange=(j>=start && j<end);
                        if(isInRange)
                            training_set[i].push_back(indices[i][j]);
                        else
                            test_set.push_back(indices[i][j]);
                    }
                }
#endif

                tmp_dataset=dataset;
                num_of_cont_features=num_of_cont_features_orig;

                for(int fi=0;fi<num_of_cont_features;++fi){
                    minValues[fi]=FLT_MAX;
                    maxValues[fi]=-FLT_MAX;
                    avgValues[fi]=stdValues[fi]=0;
                    int count=0;
                    for(int i=0;i<num_of_classes;++i){
                        for(int t=0;t<training_set[i].size();++t){
                            ++count;
                            FEATURE_TYPE feature=dataset[training_set[i][t]].features[fi];
                            if(feature<minValues[fi])
                                minValues[fi]=feature;
                            if(maxValues[fi]<feature)
                                maxValues[fi]=feature;
                            avgValues[fi]+=feature;
                            stdValues[fi]+=feature*feature;
                        }
                    }

                    avgValues[fi]/=count;
                    FEATURE_TYPE prev=stdValues[fi];
                    stdValues[fi]=sqrt((stdValues[fi] - avgValues[fi] * avgValues[fi] * count) / (count - 1));
                    if(stdValues[fi]!=stdValues[fi]){
                        cout<<"err "<<prev<<' '<<stdValues[fi]<<' '<<avgValues[fi]<<' '<<count;
                        print_endl;
                    }
                }

#if NO_PCA_FEATURES>0

                Mat training_mat(dataset.size()-test_set.size(), num_of_cont_features, CV_64F);
                int mat_ind=0;
                for(int i=0;i<num_of_classes;++i){
                    for(int t=0;t<training_set[i].size();++t){
                        for(int fi=0;fi<num_of_cont_features;++fi){
                            training_mat.at<double>(mat_ind,fi)=
                                //dataset[training_set[i][t]].features[fi];
                                //(dataset[training_set[i][t]].features[fi]-avgValues[fi])/stdValues[fi];
                                //(dataset[training_set[i][t]].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
                                    dataset[training_set[i][t]].features[fi]-avgValues[fi];
                        }
                        ++mat_ind;
                    }
                }

                PCA pca(training_mat,Mat(),CV_PCA_DATA_AS_ROW,
                    NO_PCA_FEATURES);
                    //training_mat.cols);
                Mat test_mat(1, num_of_cont_features, CV_64F);
                Mat point;
                /*cout<<"PCA "<<pca.eigenvalues.rows<<' '<<pca.eigenvalues.cols<<' '<<pca.eigenvalues.at<double>(0,0);
                for(int i=0;i<NO_PCA_FEATURES;++i)
                    cout<<pca.eigenvalues.at<double>(0,i);*/
                for(int i=0;i<dataset.size();++i){

                    for(int fi=0;fi<num_of_cont_features;++fi){
                        test_mat.at<double>(0,fi)=
                            //dataset[i].features[fi];
                            //(dataset[i].features[fi]-avgValues[fi])/stdValues[fi];
                            //(dataset[i].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
                                dataset[i].features[fi]-avgValues[fi];
                    }
                    point=pca.project(test_mat);
                    //cout<<point.rows<<' '<<point.cols<<'\n';
                    /*for(int fi=0;fi<num_of_cont_features;++fi){
                        tmp_dataset[i].features[fi]=0;
                    }*/
                    tmp_dataset[i].features.resize(point.cols);
                    for(int fi=0;fi<point.cols;++fi){
                        tmp_dataset[i].features[fi]=point.at<double>(0,fi);
                        if(tmp_dataset[i].features[fi]!=tmp_dataset[i].features[fi]){
                            cout<<"error "<<tmp_dataset[i].features[fi]<<' '<<i<<' '<<fi;
                            print_endl;
                        }
                        //cout<<tmp_dataset[i].features[fi]<<'\n';
                    }
                    //cout<<tmp_dataset[i].features[0]<<' '<<tmp_dataset[i].features[FEATURES_COUNT-1];
                }
                num_of_cont_features=NO_PCA_FEATURES;

                for(int fi=0;fi<num_of_cont_features;++fi){
                    minValues[fi]=FLT_MAX;
                    maxValues[fi]=-FLT_MAX;
                    avgValues[fi]=stdValues[fi]=0;
                    int count=0;
                    for(int i=0;i<num_of_classes;++i){
                        for(int t=0;t<training_set[i].size();++t){
                            ++count;
                            FEATURE_TYPE feature=tmp_dataset[training_set[i][t]].features[fi];
                            if(feature<minValues[fi])
                                minValues[fi]=feature;
                            if(maxValues[fi]<feature)
                                maxValues[fi]=feature;
                            avgValues[fi]+=feature;
                            //cout<<i<<' '<<t<<' '<<feature<<'\n';
                        }
                    }

                    avgValues[fi]/=count;

                    for(int i=0;i<num_of_classes;++i){
                        for(int t=0;t<training_set[i].size();++t){
                            FEATURE_TYPE feature=tmp_dataset[training_set[i][t]].features[fi];
                            stdValues[fi]+=(feature-avgValues[fi])*(feature-avgValues[fi]);
                            //cout<<feature<<' '<<avgValues[fi]<<' '<<stdValues[fi]<<'\n';
                        }
                    }
                    FEATURE_TYPE prev=stdValues[fi];
                    stdValues[fi]=sqrt(stdValues[fi]/ (count - 1));
                    if(stdValues[fi]!=stdValues[fi]){
                        cout<<"error "<<prev<<' '<<stdValues[fi]<<' '<<avgValues[fi]<<' '<<count;
                        print_endl;
                    }
                }

#endif

                double weights_sum=0;
                for(int fi=0;fi<num_of_cont_features;++fi){
                    weights[fi]=
                            //stdValues[fi];
                            //abs(avgValues[fi]);
                            1;
                            //pca.eigenvalues.at<double>(0,fi);
                    weights_sum+=weights[fi];
                }
                if(weights_sum>0){
                    for(int fi=0;fi<num_of_cont_features;++fi){
                        weights[fi]/=weights_sum;
                        //cout<<fi<<' '<<weights[fi];
                    }
                }


                int num_of_training_data=dataset.size()-test_set.size();

#if CLASSIFIER==USE_PROJECTION
                M=(int)ceil(pow(1.0*num_of_training_data/num_of_classes,1.0/3)*2);
#ifdef USE_CALTECH
                M*=2;
#endif

#ifdef CALC_M
                M=max(20,(int)ceil(num_of_training_data/num_of_classes));
#endif
                //M=4;
                //M=(int)ceil(pow(1.0*num_of_training_data,1.0/3)*2);
                //M=(int)ceil(sqrt(1.0*num_of_training_data/num_of_classes));
                const int min_M=2;
                if(M<=min_M)
                   M = min_M;
#if defined (USE_LFW) && !defined(USE_CASIA)
                M=2;
#endif
                num_of_a_coeffs.resize(num_of_classes*num_of_cont_features);
                a.resize(num_of_classes*num_of_cont_features*(2*M+1));
                fill(a.begin(),a.end(),0);
                //cout<<"M="<<M<<" a_size="<<a.size();
                for(int i=0;i<num_of_classes;++i){
                    double mult=1.0/training_set[i].size();
                    prior_probabs[i]=1.0*training_set[i].size()/num_of_training_data;

                    for(int fi=0;fi<num_of_cont_features;++fi){
                        int num_of_a_ind=fi*num_of_classes+i;
                        int model_ind=num_of_a_ind*(2*M+1);

                        for(int t=0;t<training_set[i].size();++t){
                            double cur_mult=mult;
#ifdef REGRESSION
                            cur_mult*=dataset[training_set[i][t]].output;
#endif
                            a[model_ind]+=0.5*cur_mult;
                            NORMALIZE(training_set[i][t]);
                            FEATURE_TYPE prev=0,cur=0;
                            for(int k=0;k<M;++k){
#if KERNEL==USE_FEJER
                                /*
                                a[model_ind+k]+=cos(PI*(k+1)*val)*cur_mult*(M-k)/(M+1);
                                b[model_ind+k]+=sin(PI*(k+1)*val)*cur_mult*(M-k)/(M+1);
                                */
#ifndef CALC_M
                                a[model_ind+2*k+1]+=cos(PI*(k+1)*val)*cur_mult*(M-k)/(M+1);
                                a[model_ind+2*k+2]+=sin(PI*(k+1)*val)*cur_mult*(M-k)/(M+1);
#else
                                a[model_ind+2*k+1]+=cos(PI*(k+1)*val)*cur_mult;
                                a[model_ind+2*k+2]+=sin(PI*(k+1)*val)*cur_mult;
#endif
                                //cout<<"train coefs="<<a[model_ind+k]<<' '<<b[model_ind+k]<<' '<<k<<' '<<val<<'\n';
#elif KERNEL==USE_HERMITT
                                FEATURE_TYPE cur_val;
                                switch (k){
                                case 0:
                                    cur_val = exp(-val*val / 2);
                                    prev = cur_val;
                                    break;
                                case 1:
                                    cur_val = sqrt(2) * val*prev;
                                    cur = cur_val;
                                    break;
                                default:
                                    cur_val = sqrt(2 / k)*val*cur - sqrt((k - 1) / k) *prev;
                                    prev = cur;
                                    cur = cur_val;
                                    break;
                                }
                                a[model_ind+k]+=cur_val*mult;
#endif
                            }
                            /*a[model_ind+k]=a[model_ind+k]*frac;
                            b[model_ind+k]=b[model_ind+k]*frac;*/
                        }

#ifndef CALC_M
                        num_of_a_coeffs[num_of_a_ind]=M;
#else
                        FEATURE_TYPE sum_sqr_a=0, max_J=-100000000;
                        int best_k=-1;
                        for(int k=0;k<M;++k){
                            sum_sqr_a+=(a[model_ind+2*k+1]*a[model_ind+2*k+1]+
                                    a[model_ind+2*k+2]*a[model_ind+2*k+2])/4;
                            FEATURE_TYPE J=sum_sqr_a-2*(k+1)/(training_set[i].size()+1);
                            if(J>max_J && k>=min_M){
                                max_J=J;
                                best_k=k;
                            }
                        }
                        if(best_k==-1){
                            best_k=M-1;
                            //cout<<"Error!";
                        }
                        num_of_a_coeffs[num_of_a_ind]=best_k+1;//M;
#endif
                    }
                }
#ifdef CALC_M

                int num_of_a=accumulate(num_of_a_coeffs.begin(),num_of_a_coeffs.end(),0)/num_of_a_coeffs.size();
                std::nth_element(num_of_a_coeffs.begin(), num_of_a_coeffs.begin() + num_of_a_coeffs.size() / 2, num_of_a_coeffs.end());
                num_of_a=num_of_a_coeffs[num_of_a_coeffs.size() / 2];
                //cout<<num_of_a<<' '<<M;

                for(int num_of_a_ind=0;num_of_a_ind<num_of_a_coeffs.size();++num_of_a_ind){
                    //num_of_a_coeffs[num_of_a_ind]=(num_of_a_coeffs[num_of_a_ind]<=3)?3:4;
                    //int num_of_a=num_of_a_coeffs[num_of_a_ind];
                    num_of_a_coeffs[num_of_a_ind]=num_of_a;
                    //cout<<num_of_a<<' '<<M;
                    int model_ind=num_of_a_ind*(2*M+1);
                    for(int k=0;k<num_of_a;++k){
                        a[model_ind+2*k+1]*=1.0*(num_of_a-k)/(num_of_a+1);
                        a[model_ind+2*k+2]*=1.0*(num_of_a-k)/(num_of_a+1);
                    }
                    /*for(int k=num_of_a;k<M;++k)
                        a[model_ind+2*k+1]=a[model_ind+2*k+2]=0;*/
                }
#endif
#elif KMEANS_CLUSTERS
                for(int i=0;i<num_of_classes;++i){
                    if(training_set[i].size()>KMEANS_CLUSTERS){
                        int centroid_indices[KMEANS_CLUSTERS];
                        vector<int> bestClustIndices(training_set[i].size());
                        for(int c=0;c<KMEANS_CLUSTERS;++c)
                            centroid_indices[c]=c;
                        for (int step = 0; step < 100; ++step){
                            for(int t=0;t<training_set[i].size();++t){
                                bestClustIndices[t]=-1;
                                double bestDist=DBL_MAX;
                                for(int c=0;c<KMEANS_CLUSTERS;++c)
                                    if(centroid_indices[c]>=0){
#if 1
                                        double dist=0;
                                        int d1=training_set[i][centroid_indices[c]];
                                        int d2=training_set[i][t];
                                        for(int fi_12=0;fi_12<num_of_cont_features;++fi_12){
                                            dist+=(dataset[d1].features[fi_12]-dataset[d2].features[fi_12])*(dataset[d1].features[fi_12]-dataset[d2].features[fi_12]);
                                        }
                                        dist/=num_of_cont_features;
                                        if(dist<bestDist){
                                            bestDist=dist;
                                            bestClustIndices[t]=c;
                                        }
#else
                                        if(distances[training_set[i][centroid_indices[c]]*dataset.size()+training_set[i][t]]<bestDist){
                                            bestDist=distances[training_set[i][centroid_indices[c]]*dataset.size()+training_set[i][t]];
                                            bestClustIndices[t]=c;
                                        }
#endif
                                    }
                            }
                            for(int c=0;c<KMEANS_CLUSTERS;++c){
                                double bestClustDist=DBL_MAX;
                                centroid_indices[c]=-1;
                                for(int t=0;t<training_set[i].size();++t){
                                    if(bestClustIndices[t]==c){
                                        double clustDist = 0;
                                        for(int t1=0;t1<training_set[i].size();++t1)
                                            if(bestClustIndices[t1]==c){
#if 1
                                                double dist=0;
                                                int d1=training_set[i][t];
                                                int d2=training_set[i][t1];
                                                for(int fi_12=0;fi_12<num_of_cont_features;++fi_12){
                                                    dist+=(dataset[d1].features[fi_12]-dataset[d2].features[fi_12])*(dataset[d1].features[fi_12]-dataset[d2].features[fi_12]);
                                                }
                                                dist/=num_of_cont_features;
                                                clustDist+=dist;
#else
                                                clustDist+=distances[training_set[i][t]*dataset.size()+training_set[i][t1]];
#endif
                                            }
                                        if(clustDist<bestClustDist){
                                            bestClustDist=clustDist;
                                            centroid_indices[c]=t;
                                        }
                                    }
                                }
                            }
                        }
                        int cur_size=training_set[i].size();
                        for(int c=0;c<KMEANS_CLUSTERS;++c)
                            if(centroid_indices[c]>=0)
                                training_set[i].push_back(training_set[i][centroid_indices[c]]);
                        //cout<<cur_size<<' '<<training_set[i].size()<<'\n';
                        training_set[i].erase(training_set[i].begin(),training_set[i].begin()+cur_size);
                        //cout<<"end"<<'\n';
                    }
                }

#elif CLASSIFIER==USE_SVM
                Mat labelsMat(num_of_training_data, 1, CV_32S);
                Mat trainingDataMat(num_of_training_data, num_of_cont_features, CV_32FC1);
                int ind=0;
                for(int i=0;i<num_of_classes;++i){
                    for(int t=0;t<training_set[i].size();++t){
                        for(int fi=0;fi<num_of_cont_features;++fi){
                            NORMALIZE(training_set[i][t]);
                            trainingDataMat.at<float>(ind,fi)=
                                val;
                                //dataset[training_set[i][t]].features[fi];
                                //(dataset[training_set[i][t]].features[fi]-avgValues[fi])/stdValues[fi];
                                //(dataset[training_set[i][t]].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
                        }
                        labelsMat.at<int>(ind,0)=
#ifdef REGRESSION
                            dataset[training_set[i][t]].output;
#else
                            i;
#endif
                        ++ind;
                    }
                }

                // Set up SVM's parameters
                svmClassifier = SVM::create();
                svmClassifier->setType(SVM::C_SVC);
                svmClassifier->setKernel(SVM::LINEAR);
                //svmClassifier->setKernel(SVM::RBF);
                svmClassifier->setGamma(1.0 / num_of_cont_features);
                //params.term_crit = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);

                // Train the SVM
                svmClassifier->train(trainingDataMat, ROW_SAMPLE, labelsMat);
                //bool res=svmClassifier.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);
                //std::cout<<"svmc="<<SVM.get_support_vector_count()<<" res="<<res<<"\n";
#endif
                //cout<<training_set[0].size()<<' '<<test_set.size()<<'\n';
                int total_training_size=dataset.size()-test_set.size();
                double errorRate=0,recall=0;
                vector<int> testClassCount(num_of_classes),testErrorRates(num_of_classes);
                for (int j=0;j<test_set.size();++j){
                    ++testClassCount[(int)dataset[test_set[j]].output];
                }

                auto t1 = chrono::high_resolution_clock::now();
                for(int j=0;j<test_set.size();++j){
                    int is_error=classify(test_set[j], total_training_size);
                    errorRate+=is_error;
                    testErrorRates[(int)dataset[test_set[j]].output]+=is_error;
                }
                auto t2 = chrono::high_resolution_clock::now();
                auto diff=std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/(1.0*test_set.size());
                total += diff;

                errorRate/=test_set.size();
                for(int i=0;i<num_of_classes;++i){
                    if(testClassCount[i]>0)
                        recall+=1-1.*testErrorRates[i] / testClassCount[i];
                }
                recall/=num_of_classes;
#if defined(USE_LFW) || 1
                //cout<<"db_size="<<total_training_size<<"test_num="<<test_num<<" error="<< (100 * errorRate)<<" time(us)="<<diff<<" recall="<<(100.*recall);
                print_endl;
#endif
                totalErrorRate+=errorRate;
                totalErrorSquare += errorRate*errorRate;
                totalRecall+=recall;
                //cout<<errorRate<<'\n';
            }
            for(int i=0;i<num_of_classes;++i){
                std::random_shuffle ( indices[i].begin(), indices[i].end() );
            }
        }
        totalErrorRate/=num_of_tests;
        totalErrorSquare = sqrt((totalErrorSquare - totalErrorRate * totalErrorRate * num_of_tests) / (num_of_tests - 1));
        totalRecall/=num_of_tests;
#ifdef REGRESSION
        cout<<"fraction="<<fraction<<" error="<<totalErrorRate<<" avg time(us)="<<(total/num_of_tests);
        fres<<fraction<<"\t"<<totalErrorRate<<"\t"<<(total/num_of_tests);
#else
        cout<<"fraction="<<fraction<<" db_size="<<(dataset.size()-test_set.size())<<" error="<< (100 * totalErrorRate) << " sigma=" << (100 * totalErrorSquare)<<" avg time(us)="<<(total/num_of_tests)<<" recall="<< (100 * totalRecall);
        fres<<fraction<<"\t"<< (100 * totalErrorRate) << " \t" << (100 * totalErrorSquare)<<"\t"<<(total/num_of_tests);
#endif
        print_endl;
    }

#ifdef KMEANS_CLUSTERS
    delete[] distances;
#endif
}


inline float
fasterlog2 (float x)
{
    union { float f; uint32_t i; } vx = { x };
      union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | (0x7e << 23) };
      float y = vx.i;
      y *= 1.0 / (1 << 23);

      return
        y - 124.22544637f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

#if 1
#define fastlog fasterlog2
#else
#define fastlog log10
#endif
#if 1 && CLASSIFIER==USE_PROJECTION
double classify(int test_ind, int ){
    vector<float> outputs(num_of_classes);
    vector<float> cos_vals(M),sin_vals(M);
    for(int i=0;i<num_of_classes;++i){
        outputs[i] = 0;
    }
    for(int fi=0;fi<num_of_cont_features;++fi){
        NORMALIZE(test_ind);
        cos_vals[0]=cos(PI*val);
        sin_vals[0]=sin(PI*val);
        for(int k=1;k<M;++k){
            cos_vals[k]=cos_vals[k-1]*cos_vals[0]-sin_vals[k-1]*sin_vals[0];
            sin_vals[k]=cos_vals[k-1]*sin_vals[0]+sin_vals[k-1]*cos_vals[0];
        }
        for(int i=0;i<num_of_classes;++i){
            int num_of_a_ind=fi*num_of_classes+i;
            int model_ind=num_of_a_ind*(2*M+1);
            float probab=a[model_ind];
#ifdef CALC_M
            for(int k=0;k<num_of_a_coeffs[num_of_a_ind];++k){
#else
            for(int k=0;k<M;++k){
#endif
                probab+=(a[model_ind+2*k+1]*cos_vals[k]+a[model_ind+2*k+2]*sin_vals[k]);
            }
            //qDebug()<<"M="<<M<<" before "<<probab<<" val="<<val<<" acos (val)="<< acos(cos_vals[0])/PI<<" cos_model"<<a[model_ind+1]<<" model="<<acos(a[model_ind+1])/PI;
            //probab=acos(probab-0.5)/PI;
            //qDebug()<<"after "<<probab<<"val-model="<<(val-acos(a[model_ind+1])/PI);
            outputs[i] += fastlog(probab);//num_of_cont_features;
        }
    }

    if(num_of_discrete_features>0){
        for(int i=0;i<num_of_classes;++i){
            outputs[i]/=num_of_cont_features;
            for(int fi=0;fi<num_of_discrete_features;++fi){
                outputs[i]+=fastlog(discrete_class_histos[fi][i][dataset[test_ind].discrete_features[fi]]);
            }
        }
    }

    float max_output=-FLT_MAX;
    int bestClass=-1;
    for(int i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }
    double error=0;
    if(bestClass!=dataset[test_ind].output)
        ++error;
    return error;
}
#else
double classify(int test_ind, int total_training_size){
    double var=0.00002;
    //var*=10;
    if(num_of_cont_features>2000)
        var/=10;
    vector<float> outputs(num_of_classes);
#if CLASSIFIER==USE_PROJECTION
    vector<float> cos_vals(M),sin_vals(M);
    for(int i=0;i<num_of_classes;++i){
        outputs[i]=0;
    }
    //num_of_cont_features = 1;
    for(int fi=0;fi<num_of_cont_features;++fi){
        NORMALIZE(test_ind);
        //cout << "val=" << val << '\n';
        cos_vals[0]=cos(PI*val);
        sin_vals[0]=sin(PI*val);
        for(int k=1;k<M;++k){
            /*cos_vals[k]=cos(PI*(k+1)*val);
            sin_vals[k]=sin(PI*(k+1)*val);*/
            cos_vals[k]=cos_vals[k-1]*cos_vals[0]-sin_vals[k-1]*sin_vals[0];
            sin_vals[k]=cos_vals[k-1]*sin_vals[0]+sin_vals[k-1]*cos_vals[0];
        }
        for(int i=0;i<num_of_classes;++i){
            int model_ind=
                    //(i*num_of_cont_features+fi);
                    (fi*num_of_classes+i)*(2*M+1);
            float probab=a[model_ind];
    #if KERNEL!=USE_FEJER
            probab=0;
    #endif
            FEATURE_TYPE prev=0,cur=0;
            for(int k=0;k<M;++k){
    #if KERNEL==USE_FEJER
                //cout<<"coefs="<<a[model_ind+k]<<' '<<b[model_ind+k]<<' '<<cos_vals[k]<<' '<<sin_vals[k]<<' '<<probab<<'\n';
                probab+=(a[model_ind+2*k+1]*cos_vals[k]+a[model_ind+2*k+2]*sin_vals[k]);
    #elif KERNEL==USE_HERMITT
                FEATURE_TYPE cur_val;
                switch (k){
                case 0:
                    cur_val = exp(-val*val / 2);
                    prev = cur_val;
                    break;
                case 1:
                    cur_val = sqrt(2) * val*prev;
                    cur = cur_val;
                    break;
                default:
                    cur_val = sqrt(2 / k)*val*cur - sqrt((k - 1) / k) *prev;
                    prev = cur;
                    cur = cur_val;
                    break;
                }
                probab += a[model_ind + k] * cur_val;
#endif
            }
#if 0
            if(probab<-0.001){
                cout<<probab<<' '<<val<<' '<<fi;
                /*for(int t=0;t<training_set[i].size();++t){
                    NORMALIZE(training_set[i][t]);
                    cout<<val<<'('<<dataset[training_ind].features[fi]<<") ";
                }
                cout<<"\n";
                for(int k=0;k<M;++k){
                    cout<<a[model_ind+k]<<' '<<b[model_ind+k]<<' ';
                }*/
                cout<<"\n";
                exit(0);
            }
#endif
            outputs[i] +=fastlog(probab);
        }
    }
    /*for(int i=0;i<num_of_classes;++i){
        outputs[i] = 0;// log(prior_probabs[i]);
        for(int fi=0;fi<num_of_cont_features;++fi){
            outputs[i] += fasterlog2(probabs[i*num_of_cont_features + fi]);
        }
        if(num_of_discrete_features>0){
            outputs[i]/=num_of_cont_features;
            for(int fi=0;fi<num_of_discrete_features;++fi){
                outputs[i]+=fasterlog2(discrete_class_histos[fi][i][dataset[test_ind].discrete_features[fi]]);
            }
        }
    }*/
    if(num_of_discrete_features>0){
        for(int i=0;i<num_of_classes;++i){
            outputs[i]/=num_of_cont_features;
            for(int fi=0;fi<num_of_discrete_features;++fi){
                outputs[i]+=fastlog(discrete_class_histos[fi][i][dataset[test_ind].discrete_features[fi]]);
            }
        }
    }
#ifdef REGRESSION
    for(int i=0;i<num_of_classes;++i){
        outputs[i]=exp(outputs[i]);
    }
#endif
#elif CLASSIFIER==USE_PNN
    for(int i=0;i<num_of_classes;++i){
        outputs[i]=0;
        /*outputs[i]=1;
        for(int fi=0;fi<num_of_cont_features;++fi){
            double probab=0;
            for(int t=0;t<training_set[i].size();++t){
                int training_ind=training_set[i][t];
                //double dist=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                double diff=0;
                {
                NORMALIZE(training_ind);
                diff=val;
                }
                {
                NORMALIZE(test_ind);
                diff-=val;
                }
                double dist=diff*diff;
                probab+=exp(-dist/(2*var*num_of_cont_features));
            }
            outputs[i]*=probab;
        }*/
        double den=
#ifdef REGRESSION
            0;
#else
            total_training_size;
#endif

#if 1
        for(int t=0;t<training_set[i].size();++t){
            int training_ind=training_set[i][t];
            FEATURE_TYPE dist=0;
            for(int fi=0;fi<num_of_cont_features;++fi){
                //dist+=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff=0;
                {
                NORMALIZE(training_ind);
                diff=val;
                }
                {
                NORMALIZE(test_ind);
                diff-=val;
                }
                dist+=diff*diff;
            }
            dist/=num_of_cont_features;
#ifdef REGRESSION
            double tmp=exp(-dist/(2*var));
            outputs[i]+=dataset[training_ind].output*tmp;
            den+=tmp;
#else
            outputs[i]+=exp(-dist/(2*var));
#endif
        }
        outputs[i]=log(outputs[i]);
        outputs[i] /= den;
        //cout<<i<<' '<<outputs[i];
#else
        outputs[i] = 0;
#if 0
        for (int fi = 0; fi<num_of_cont_features; ++fi){
            double fi_log_probab = 0;
            for (int t = 0; t<training_set[i].size(); ++t){
                int training_ind = training_set[i][t];
                //dist+=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff = 0;
                {
                    NORMALIZE(training_ind);
                    diff = val;
                    //cout << "train=" << val << '\n';
                }
                {
                    NORMALIZE(test_ind);
#ifdef DEBUG_PNN
                    cout << "val=" << val << '\n';
#endif
                    diff -= val;
                }
                const int M = 5;
                double tmp = 1;
#if KERNEL==USE_FEJER
                if (diff != 0){
                    tmp = sin(PI*diff*(M + 1) / 2) / sin(PI*diff / 2);
                }
                else
                    tmp = M + 1;
                tmp = (tmp*tmp / (2 * (M + 1)));
#elif KERNEL==USE_HERMITT
#error no hermitt
#endif
                fi_log_probab += tmp;
                //outputs[i] *=tmp;
#ifdef DEBUG_PNN
                cout << "i=" << i << " probab=" << tmp << ' ' << fi_log_probab << ' ' << diff << '\n';
#endif
            }
            outputs[i] += log(fi_log_probab / (num_of_cont_features*training_set[i].size()));
        }
#else
        for (int t = 0; t<training_set[i].size(); ++t){
            int training_ind = training_set[i][t];
            double probab = 1;
            for (int fi = 0; fi<num_of_cont_features; ++fi){
                FEATURE_TYPE diff = 0;
                {
                    NORMALIZE(training_ind);
                    diff = val;
                }
                {
                    NORMALIZE(test_ind);
                    diff -= val;
                }
                const int M = 5;
                double tmp = 1;
                if (diff != 0){
                    tmp = sin(PI*diff*(M + 1) / 2) / sin(PI*diff / 2);
                }
                else
                    tmp = M + 1;
                tmp = (tmp*tmp / (2 * (M + 1)));
                probab *= tmp;
            }
            outputs[i] += probab;
        }
        outputs[i] = log(outputs[i]);
#endif
        //outputs[i] /= num_of_cont_features;
#endif
        for(int fi=0;fi<num_of_discrete_features;++fi){
            outputs[i]+=log(discrete_class_histos[fi][i][dataset[test_ind].discrete_features[fi]]);
        }
#ifdef DEBUG_PNN
        cout << outputs[i] << '\n';
#endif
    }
#ifdef DEBUG_PNN
    cout << '\n';
    /*static int tests = 0;
    if(++tests>=5)
        exit(0);
    else
        cout<<'\n';*/
#endif
#elif CLASSIFIER==USE_KNN
    vector<FEATURE_TYPE> distances(total_training_size);
    vector<int> idx(total_training_size), class_label(total_training_size);
    int cur_ind = 0;

    for (int i = 0; i < num_of_classes; ++i){
        outputs[i] = 0;
        for (int t = 0; t < training_set[i].size(); ++t){
            FEATURE_TYPE dist = 0;
            for (int fi = 0; fi < num_of_cont_features; ++fi){
                int training_ind = training_set[i][t];
                //FEATURE_TYPE dist=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff = 0;
                {
                    NORMALIZE(training_ind);
                    diff = val;
                }
                {
                    NORMALIZE(test_ind);
                    diff -= val;
                }
                dist += diff*diff;
            }
            dist/=num_of_cont_features;
            distances[cur_ind] = dist;
            idx[cur_ind] = cur_ind;
            class_label[cur_ind] = i;
            ++cur_ind;
        }
    }

    std::sort(idx.begin(), idx.end(),
        [&distances](size_t i1, size_t i2) {return distances[i1] < distances[i2]; });
    const int K=1;
    //cout << distances[idx[0]] << '\n';
    for (int i = 0; i < total_training_size; ++i){
        int c = class_label[idx[i]];
        ++outputs[c];
        if (outputs[c] >= K){
            break;
        }
    }


#elif CLASSIFIER==USE_SVM
    Mat queryMat(1, num_of_cont_features, CV_32FC1);
    for(int fi=0;fi<num_of_cont_features;++fi){
        NORMALIZE(test_ind);
        queryMat.at<float>(0,fi)=
            val;
            //dataset[test_ind].features[fi];
            //(dataset[test_ind].features[fi]-avgValues[fi])/stdValues[fi];
            //(dataset[test_ind].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
    }
    float response = svmClassifier->predict(queryMat);
    for(int i=0;i<num_of_classes;++i){
        outputs[i]=0;
    }
#ifdef REGRESSION
    outputs[0]=response;
#else
    outputs[(int)response]=1;
    /*for(int fi=0;fi<num_of_discrete_features;++fi){
        outputs[response]*=(discrete_class_histos[fi][response][dataset[test_ind].discrete_features[fi]]);
    }*/
#endif
#endif
    float max_output=-DBL_MAX;
    int bestClass=-1;
    for(int i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }
    double error=0;
#ifdef REGRESSION
    error=abs(dataset[test_ind].output-max_output);
    //cout<<max_output<<' '<<dataset[test_ind].output<<'\n';
#else
    if(bestClass!=dataset[test_ind].output)
        ++error;
#endif
    return error;
}
#endif
