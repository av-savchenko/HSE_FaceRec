#include "stdafx.h"

#include "RecognitionThreadPool.h"

#include <windows.h>
#include <strsafe.h>


#include <vector>
#include <string>
#include <iostream>
#include <fstream>


#include "DEMTesting.h"
#include "db.h"
#include "FaceDatabase.h"
#include "DirectedEnumeration.h"
#include "TestDb.h"
#include "DnnFeatureExtractor.h"


#ifdef NEED_NON_METRIC_SPACE_LIB
using namespace similarity;
#endif

using namespace std;

//noise level x_\eta
const int rndImageRange = 0;

static void loadImages(std::string path,std::vector<FaceImage*>& faceImages){
	WIN32_FIND_DATA ffd,ffd1;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	HANDLE hFindFile = INVALID_HANDLE_VALUE;
	
    std::cout <<"start load"<<std::endl;

   
	hFind = FindFirstFile((path+"\\*").c_str(), &ffd);
	if (INVALID_HANDLE_VALUE == hFind) {
		std::cout<<"no dirs. Return."<<std::endl;
		return;
	}
	do
	{
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
      {
		  std::string dirName=ffd.cFileName;
		  if(dirName[0]=='.')
			  continue;
		 std::string fullDirName=path+"\\"+dirName+"\\";
		 hFindFile = FindFirstFile((fullDirName+"*").c_str(), &ffd1);
		if (INVALID_HANDLE_VALUE == hFindFile) {
			std::cout<<"no files."<<std::endl;
		}
		do
		{
		  if ((ffd1.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)==0)
		  {
			  faceImages.push_back(new FaceImage((fullDirName + ffd1.cFileName).c_str(), dirName, POINTS_IN_W, POINTS_IN_H, rndImageRange));
		  }
	   }
	   while (FindNextFile(hFindFile, &ffd1) != 0);
      }
   }
   while (FindNextFile(hFind, &ffd) != 0);
    std::cout <<"end load"<<std::endl;

}


void runOneTest(std::string prefix, RecognitionThreadPool& recognitionThreadPool, const vector<FaceImage*>& testImages, std::ofstream& resFile)
{
	double mean_error=0, std_error=0;
	double mean_time=0, std_time=0;
	double total_time=0;
	LARGE_INTEGER freq, start, end; 
	QueryPerformanceFrequency(&freq); 
	
	recognitionThreadPool.init();

	const int TESTS_COUNT=10;

	for(int t=0;t<TESTS_COUNT;++t){
		int errorsCount=0;
		QueryPerformanceCounter(&start); 
		for(vector<FaceImage*>::const_iterator iter=testImages.begin();iter!=testImages.end();++iter){
			FaceImage* test=*iter;
			FaceImage* bestImage=recognitionThreadPool.recognizeTestImage(test);
			if(test->personName!=bestImage->personName){
				++errorsCount;
				//std::cout << test->fileName << " " << bestImage->fileName << " "  << bestImage->distance(test) << std::endl;
				//resFile << test->fileName << " " << bestImage->fileName << " " << bestImage->distance(test) << std::endl;
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
	
	mean_error /= TESTS_COUNT;
	std_error=std_error/TESTS_COUNT-mean_error*mean_error;
	std_error=(std_error>0)?sqrt(std_error):0;
	
	mean_time/=TESTS_COUNT;
	std_time=std_time/TESTS_COUNT-mean_time*mean_time;
	std_time = (std_time>0) ? sqrt(std_time*testImages.size()) : 0;
	
	float avgCheckedPercent=-1;
	if (RecognitionThreadPool::nnMethod == DEM || RecognitionThreadPool::nnMethod == NON_METRIC_SPACE_LIB || RecognitionThreadPool::nnMethod == SMALL_WORLD)
		avgCheckedPercent = recognitionThreadPool.getAverageCheckedPercent() / (testImages.size()*TESTS_COUNT);
	std::cout<<prefix<<" error="<<mean_error<<'('<<std_error<<')'<<
			" time="<<(total_time/1000)<<" ms, rel="<<
		(mean_time/1000)<<'('<<(std_time/1000)<<") ms";
	//resFile<<prefix<<'\t'<<mean_error<<'('<<std_error<<')'<<'\t'<<mean_time<<'('<<std_time<<')';
	resFile << prefix << '\t' << mean_error << '\t' << mean_time;
	if (avgCheckedPercent >= 0){
		std::cout << " ModelCheck=" << avgCheckedPercent;
		resFile << " ModelCheck=" << avgCheckedPercent;
	}
	std::cout<<std::endl;
	resFile<<std::endl;
}



static void loadImagesFromOneDir(std::string path, std::vector<FaceImage*>& dbImages, std::vector<FaceImage*>& testImages){
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	std::cout << "start load from one dir" << std::endl;

	hFind = FindFirstFile((path + "\\*").c_str(), &ffd);
	if (INVALID_HANDLE_VALUE == hFind) {
		std::cout << "no files. Return." << std::endl;
		return;
	}
	do
	{
		if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
		{
			string fileName = ffd.cFileName;
			int dashInd = fileName.find('-');
			int startUserInd = fileName.find('_');
			if (dashInd != -1 && startUserInd != -1 && startUserInd < dashInd - 1){
				string personName = fileName.substr(startUserInd + 1, dashInd - startUserInd - 1);
				int uid = atoi(personName.c_str());
				if (uid>20)
					continue;
				//cout << uid << ' ' << personName << endl;
				string prefix = fileName.substr(0, startUserInd);
				string fullFileName = path + "\\" + fileName;
				//cout << prefix << ' ' << personName << endl;
				if (prefix == "train")
					dbImages.push_back(new FaceImage(fullFileName.c_str(), personName, POINTS_IN_W, POINTS_IN_H, 0));
				else if (prefix == "test")
					testImages.push_back(new FaceImage(fullFileName.c_str(), personName, POINTS_IN_W, POINTS_IN_H, rndImageRange));
			}
			else
				cout << "Invalid file name " << fileName << endl;
		}
	} while (FindNextFile(hFind, &ffd) != 0);
	std::cout << "end load dbSize=" << dbImages.size() << " testSize=" << testImages.size() << std::endl;

}
using namespace cv;
void runFaceRecognitionTest(vector<FaceImage*>& dbImages, const vector<FaceImage*>& testImages){
	//FacesDatabase fdb(dbImages);
	double mean_error = 0, std_error = 0;
	double mean_time = 0, std_time = 0;
	double total_time = 0;
	LARGE_INTEGER freq, start, end;

	int errorsCount = 0;
	QueryPerformanceFrequency(&freq);
	int dbSize, testSize,featuresCount;

#if 0
	/*{
		cout << "before write\n";
		FileStorage fs("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\test_files\\imgs.yml", FileStorage::WRITE);
		Mat mat(2, 3, CV_32F);
		for (int i = 0; i < mat.rows; ++i){
			for (int j = 0; j < mat.cols; ++j){
				mat.at<float>(i, j)=i*j;
			}
		}
		fs << "Train" << mat;
		fs.release();
		cout << "after write\n";
		exit(0);
	}*/
#if 0
	Mat test_features, db_features;
	FileStorage fs("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\test_files\\imgs.yml", FileStorage::READ);
	fs ["Train"]>> db_features;
	fs ["Test"]>> test_features;
	dbSize = db_features.rows;
	featuresCount = db_features.cols;
	testSize = test_features.rows;
#else
	ifstream ifTrainFeatures("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\trainImgsHogNoPca.txt");
	ifstream ifTestFeatures("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\testImgsHogNoPca.txt");
	ifTrainFeatures >> featuresCount >> dbSize;
	ifTestFeatures >> featuresCount >> testSize;


	Mat test_features(testSize, featuresCount, CV_32F);
	for (int j = 0; j < featuresCount; ++j){
		for (int i = 0; i < testSize; ++i){
			ifTestFeatures >> test_features.at<float>(i, j);
		}
	}


	Mat db_features(dbSize, featuresCount, CV_32F);
	for (int j = 0; j < featuresCount; ++j){
		for (int i1 = 0; i1 < dbSize; ++i1){
			ifTrainFeatures >> db_features.at<float>(i1, j);
		}
	}

	ifTrainFeatures.close();
	ifTestFeatures.close();

	/*{
		cout << "before write\n";
		FileStorage fs("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\test_files\\imgs.yml", FileStorage::WRITE);
		fs << "Train" << db_features;
		fs << "Test" << test_features;
		fs.release();
		cout << "after write\n";
	}*/
#endif
	cout << featuresCount << ' ' << dbSize << ' ' << testSize<<endl;
	vector<string> db_classes(dbSize), test_classes(testSize);
	ifstream ifTrainIds("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\trainIds1.txt");
	ifstream ifTestIds("D:\\HSE\\experiments\\images\\PubFig+LFW\\test_all\\detected\\testIds1.txt");
	for (int i = 0; i < testSize; ++i){
		ifTestIds >> test_classes[i];
	}
	for (int i1 = 0; i1 < dbSize; ++i1){
		ifTrainIds >> db_classes[i1];
	}

	cout << "fbgTestImgs(1536,1)" << test_features.at<float>(0, 1535) << endl;
	cout << "fbgTestImgs(1535,1)" << test_features.at<float>(0, 1534) << endl;
	ifTrainIds.close();
	ifTestIds.close();

#else
	dbSize = dbImages.size();
	testSize = testImages.size();
	featuresCount =
#ifdef USE_EXTRA_FEATURES
		dbImages[0]->getFeatureVector().size();
#else
		FEATURES_COUNT;
#endif
	vector<string> db_classes(dbSize), test_classes(testSize);

	//double* db_features = new double[dbImages.size()*FEATURES_COUNT];
	Mat db_features(dbSize, featuresCount, CV_32F);
	for (int i1 = 0; i1 < dbSize; ++i1){
		db_classes[i1] = dbImages[i1]->personName;
		for (int j = 0; j < featuresCount; ++j){
			//db_features[i1*featuresCount + j] = 
			db_features.at<float>(i1, j) =
				dbImages[i1]->getFeatures()[j];
		}
	}
	//double* test_features = new double[testImages.size()*FEATURES_COUNT];
	Mat test_features(testSize, featuresCount, CV_32F);
	for (int i = 0; i < testImages.size(); ++i){
		test_classes[i] = testImages[i]->personName;
		for (int j = 0; j < featuresCount; ++j){
			//test_features[i*featuresCount + j] = 
			test_features.at<float>(i, j) =
				testImages[i]->getFeatures()[j];
		}
	}
#endif
#if 0
	Mat db_projection_result, test_projection_result;
	PCA pca(db_features, Mat(), CV_PCA_DATA_AS_ROW, 1536);
	pca.project(db_features, db_projection_result);
	pca.project(test_features, test_projection_result);
	cout << db_projection_result.rows << ' ' << db_projection_result.cols << endl;
	
	Mat& input_train_features = db_projection_result;
	Mat& input_test_features = test_projection_result;
#else
	Mat& input_train_features = db_features;
	Mat& input_test_features = test_features;
#endif

#if 0
	Mat preprocessed_train(input_train_features.rows, input_train_features.cols, CV_32F);
	Mat preprocessed_test(input_test_features.rows, input_test_features.cols, CV_32F);
	Mat avgImage;
	reduce(input_train_features, avgImage, 0, CV_REDUCE_AVG);
	for (int r = 0; r < preprocessed_train.rows; ++r) {
		preprocessed_train.row(r) = input_train_features.row(r);// -avgImage;
	}
	for (int r = 0; r < preprocessed_test.rows; ++r) {
		preprocessed_test.row(r) = input_test_features.row(r);// -avgImage;
	}
#else
	Mat& preprocessed_train = input_train_features;
	Mat& preprocessed_test = input_test_features;
#endif

	cout << input_train_features.rows << ' ' << input_train_features.cols << endl;

	QueryPerformanceCounter(&start);
	for (int i = 0; i < testSize; ++i){
#if 1
		int bestInd = -1;
		double bestDist = FLT_MAX;
		for (int i1 = 0; i1 < dbSize; ++i1){
			double tmp_dist = 0;
			for (int j = 0; j < preprocessed_train.cols; j+=1){
				double tmp = 0;
				for (int k = 0; k < 1; ++k){
					if (abs(preprocessed_train.at<float>(i1, j + k) - preprocessed_test.at<float>(i, j + k))>0.001)
						//tmp_dist +=
						tmp +=
						//abs(db_features[i1*featuresCount + j] - test_features[i*featuresCount + j]);
						abs(preprocessed_train.at<float>(i1, j + k) - preprocessed_test.at<float>(i, j + k));
						//(preprocessed_train.at<float>(i1, j + k) - preprocessed_test.at<float>(i, j + k))*(preprocessed_train.at<float>(i1, j + k) - preprocessed_test.at<float>(i, j + k)) / (preprocessed_train.at<float>(i1, j + k) + preprocessed_test.at<float>(i, j + k));
						//abs(input_train_features.at<float>(i1, j) - input_test_features.at<float>(i, j));
						//(preprocessed_train.at<float>(i1, j) - preprocessed_test.at<float>(i, j))*(preprocessed_train.at<float>(i1, j) - preprocessed_test.at<float>(i, j));
				}
				//tmp_dist += sqrt(tmp);
				tmp_dist += tmp;
			}
			if (tmp_dist < bestDist){
				bestDist = tmp_dist;
				bestInd = i1;
			}
		}
		/*if(i<=10)
			cout<<bestInd<<' '<<bestDist<<endl;*/
		if (test_classes[i] != db_classes[bestInd]){
#else
		FaceImage* test = testImages[i];
		fdb.setTestImage(test);
		int bestInd=fdb.getClosestIndex(0);
		FaceImage* bestImage = dbImages[bestInd];
		if (test->personName != bestImage->personName){
#endif
			++errorsCount;
		}
	}
	QueryPerformanceCounter(&end);
	double error_rate = 100.*errorsCount / testSize;
	double delta_microseconds = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart*1000000.0;
	total_time += delta_microseconds;

	mean_error += error_rate;
	std_error += error_rate*error_rate;

	delta_microseconds /= testSize;
	mean_time += delta_microseconds;
	std_time += delta_microseconds*delta_microseconds;

	std_error = std_error - mean_error*mean_error;
	std_error = (std_error>0) ? sqrt(std_error) : 0;

	std_time = std_time - mean_time*mean_time;
	std_time = (std_time>0) ? sqrt(std_time) : 0;

	std::cout <<  "error=" << mean_error << " accuracy="<<(100-mean_error)<<'(' << std_error << ')' <<
		" time=" << (total_time / 1000) << " ms, rel=" <<
		(mean_time / 1000) << '(' << std_time << ") us\n";

	/*delete[] db_features;
	delete[] test_features;*/
	exit(0);
}
#define USE_PARALLEL_DEM

template<typename Cont, typename It>
auto ToggleIndices(Cont &cont, It beg, It end) -> decltype(std::end(cont))
{
	int helpIndx(0);
	return std::remove_if(std::begin(cont), std::end(cont),
		[&](typename Cont::value_type const& val) -> bool {
		return std::find(beg, end, helpIndx++) != end;
	});
}

//supervised clastering (SC)
void cluster_database(vector<FaceImage*>& dbImages){
	if (1)
		return;
	const float threshold =
		//0.584f;
		
		//0.382f;
		0.698f;

		//0.309f;
		//0.52f;
		//0.459f;//FERET, delta=0
		//0.345f;//FERET, delta=1
		//0.571f; //Essex, delta=0

	cout << "begin clustering dbSize=" << dbImages.size() << "\n";
	map<string, vector<int> > classes;
	for (int i = 0; i < dbImages.size(); ++i){
		//if (classes.find(dbImages[i]->personName) == classes.end())
		classes[dbImages[i]->personName].push_back(i);
	}

	vector<int> indices_to_remove;

	for (auto class_indices : classes){
		vector<int>& indices = class_indices.second;
		int ind_count = indices.size();
		if (ind_count <= 1)
			continue;
		vector<float> distance_matrix(ind_count*ind_count);

		float min_sum = numeric_limits<float>::max();
		int best_index = -1;
		for (int i = 0; i < ind_count; ++i){
			float sum = 0;
			for (int j = 0; j < ind_count; ++j){
				distance_matrix[i*ind_count + j] = dbImages[indices[j]]->distance(dbImages[indices[i]]);
				sum += distance_matrix[i*ind_count + j];
			}
			if (sum < min_sum){
				min_sum = sum;
				best_index = i;
			}
		}
#if 1
		set<int> indices_to_leave;
		indices_to_leave.insert(best_index);
		while (true){
			float worst_distance=-1;
			int worst_index = -1;
			for (int i = 0; i < ind_count; ++i){
				float best_distance = numeric_limits<float>::max();
				for (int j :indices_to_leave){
					//cout << i << ' ' << worst_distance << ' ' << j << ' ' << distance_matrix[i*ind_count + j] << endl;
					if (best_distance> distance_matrix[i*ind_count + j]){
						best_distance = distance_matrix[i*ind_count + j];
					}
				}
				if (worst_distance < best_distance){
					worst_distance = best_distance;
					worst_index = i;
				}
			}
			if (worst_distance <= threshold)
				break;
			indices_to_leave.insert(worst_index);
			//cout << worst_distance << ' ' << worst_index << ' ' << indices_to_leave.size() << endl;
		}
#else
		int K = (int)(sqrt(ind_count/2.));
		if (K < 1)
			K = 1;
		vector<int> centroids(K);
		vector<int> clusters(ind_count);
		std::fill(clusters.begin(), clusters.end(), -1);
		bool modified = true;
		while (modified){
			//maximization
			modified = false;
			for (int j = 0; j < ind_count; ++j){
				float min_dist = numeric_limits<float>::max();
				int best_centroid = -1;
				for (int ii = 0; ii < centroids.size(); ++ii){
					if (distance_matrix[centroids[ii] * ind_count + j] < min_dist){
						min_dist = distance_matrix[centroids[ii] * ind_count + j];
						best_centroid = ii;
					}
				}
				if (clusters[j] != best_centroid){
					modified = true;
					clusters[j] = best_centroid;
				}
			}

			//expectation
			for (int c = 0; c < centroids.size(); ++c){
				int best_ind = -1;
				float min_sum = numeric_limits<float>::max();
				for (int j = 0; j < ind_count; ++j){
					if (clusters[j] == c){
						float sum = 0;
						int num_of_elements = 0;
						for (int j1 = 0; j1 < ind_count; ++j1){
							if (clusters[j1] == c){
								sum += distance_matrix[j * ind_count + j1];
								++num_of_elements;
							}
						}
						if (num_of_elements>0)
							sum /= num_of_elements;
						if (sum < min_sum){
							min_sum = sum;
							best_ind = j;
						}
					}
				}
				centroids[c] = best_ind;
			}
		}
		set<int> indices_to_leave(centroids.begin(),centroids.end());
#endif
		for (int i = 0; i < ind_count; ++i){
			if (indices_to_leave.find(i) == indices_to_leave.end()){
				indices_to_remove.push_back(indices[i]);
			}
		}
	}
	cout << indices_to_remove.size() << endl;
	dbImages.erase(ToggleIndices(dbImages, std::begin(indices_to_remove), std::end(indices_to_remove)), dbImages.end());
	cout << "end clustering dbSize=" << dbImages.size()<<"\n";
}
void testDEM(){
	/*float features[256];
	std::cout << "start\n";
	DnnFeatureExtractor::GetInstance()->extractFeatures("D:\\HSE\\experiments\\images\\lfw-deepfunneled\\DNN_results\\save_dir\\Aaron_Eckhart\\Aaron_Eckhart_0001.bmp", features);
	for (int i = 0; i < 10;++i)
		std::cout << features[i] << ' ';
	std::cout << features[255] << endl;

	DnnFeatureExtractor::GetInstance()->extractFeatures("D:\\HSE\\experiments\\images\\lfw-deepfunneled\\DNN_results\\save_dir\\Aaron_Guiel\\Aaron_Guiel_0001.bmp", features);
	for (int i = 0; i < 10; ++i)
		std::cout << features[i] << ' ';
	std::cout << features[255] << endl;

	exit(0);*/
#ifdef NEED_NON_METRIC_SPACE_LIB
	initLibrary("logfile.txt");
	//testNonMetricSpace();
#endif
	//fill_image();
	//return;

	const float FALSE_ACCEPT_RATE = (DB_USED == USE_TEST_DB) ? 0.02f : 0.01f;//(DB_USED==USE_FERET)?0.01:0.01;
	//loadImage("D:\\HSE\\Users\\Andrey\\Documents\\images\\yales\\hard\\db\\subject04\\subject04.normal.jpg");

	vector<FaceImage*> dbImages;
	vector<FaceImage*> testImages;

#if DB_USED != USE_TEST_DB
	
#if DB_USED == USE_LFW_PUBFIG83
	loadImagesFromOneDir(DB, dbImages, testImages);
#else
#if 1
	MapOfFaces totalImages;
	loadFaces(totalImages);
	getTrainingAndTestImages(totalImages, dbImages, testImages,false);
#else
	loadImages(DB, dbImages);
	loadImages(TEST, testImages);
#endif
#endif
	//runFaceRecognitionTest(dbImages, testImages);

	
	/*map<string, int> classes;
	for (int i = 0; i < dbImages.size(); ++i){
		if (classes.find(dbImages[i]->personName) == classes.end())
			classes[dbImages[i]->personName] = classes.size();
	}
	ofstream dbLabels("train.txt");
	for (int i = 0; i<dbImages.size(); ++i)
		dbLabels << dbImages[i]->fileName << ' ' << classes[dbImages[i]->personName]<<endl;
	dbLabels.close();
	ofstream testLabels("val.txt");
	for (int i = 0; i<testImages.size(); ++i)
		testLabels << testImages[i]->fileName << ' ' << classes[testImages[i]->personName] << endl;
	testLabels.close();*/

	/*std::ofstream of("features_train.txt");
	for (int i = 0; i < dbImages.size(); ++i){
		of << dbImages[i]->fileName << ' ' << dbImages[i]->personName << " " << FEATURES_COUNT << "\n";
		for (int j = 0; j < FEATURES_COUNT; j++)
		{
			of << dbImages[i]->getFeatures()[j] << ' ';
		}
		of << "\n";
	}
	of.close();

	of.open("features_test.txt");
	for (int i = 0; i < testImages.size(); ++i){
		of << testImages[i]->fileName << ' ' << testImages[i]->personName << " " << FEATURES_COUNT << "\n";
		for (int j = 0; j < FEATURES_COUNT; j++)
		{
			of << testImages[i]->getFeatures()[j] << ' ';
		}
		of << "\n";
	}
	of.close();*/
	

	//test_bf(dbImages, testImages);
	/*ofstream dbLabels("training_labels.txt");
	for(int i=0;i<dbImages.size();++i)
		dbLabels<<dbImages[i]->personName<<endl;
	dbLabels.close();
	ofstream testLabels("test_labels.txt");
	for (int i = 0; i<testImages.size(); ++i)
		testLabels << testImages[i]->personName << endl;
	testLabels.close();*/
#else
	cout<<"start load\n";
	load_model_and_test_images(dbImages,testImages);
	cout<<"loaded\n";
#endif
	cluster_database(dbImages);
	cout << "dbSize=" << dbImages.size() << " testSize=" << testImages.size() << endl;
	RecognitionThreadPool recognitionThreadPool(dbImages, FALSE_ACCEPT_RATE);
	cout<<"thread pool loaded\n";
	recognitionThreadPool.setImageCountToCheck(100);

	FaceImage* bestImage=0;
    float bestDist=100000,tmpDist;

	std::vector<FaceImage*>::iterator iter;

    int errorsCount=0;

    LARGE_INTEGER freq, start, end; 
	float delta_milliseconds;

	QueryPerformanceFrequency(&freq); 

	
	std::ofstream resFile("res.txt");
	resFile.imbue(std::locale());

	RecognitionThreadPool::nnMethod=NNMethod::Simple;
	runOneTest("Brute force", recognitionThreadPool, testImages, resFile);

	//RecognitionThreadPool::nnMethod=SVM_CLASSIFIER;
	//runOneTest("SVM", recognitionThreadPool, testImages, resFile);

#ifndef USE_PARALLEL_DEM
	DirectedEnumeration<FaceImage*> directedEnumeration(dbImages,0.01,0);
#endif

	//char c;
	//std::cerr<"\ninput\n";
	//std::cin>>c;
	int high=dbImages.size()-dbImages.size()/NUM_OF_THREADS*(NUM_OF_THREADS-1);
	//for (double ratio = 0.1; ratio <= /*0.5*/1.001; ratio += 0.1)
	for (double ratio = 0.05; ratio <= /*0.5*/1.001; ratio += 0.05)
	//for (double ratio = 0.025; ratio <= /*0.5*/1.001; ratio += 0.025)
	{
		int imageCountToCheck=(int)(ratio*high);
		recognitionThreadPool.setImageCountToCheck(imageCountToCheck);
		resFile<<imageCountToCheck<<' '<<ratio<<std::endl;
		std::cout<<imageCountToCheck<<' '<<ratio<<std::endl;
#ifdef NEED_KD_TREE
		RecognitionThreadPool::nnMethod=KD_TREE;
		if(imageCountToCheck!=0)
			runOneTest("BBF", recognitionThreadPool, testImages, resFile);
		//return;
#endif

#ifdef NEED_FLANN
		RecognitionThreadPool::nnMethod=FLANN;
		runOneTest("FLANN", recognitionThreadPool, testImages, resFile);
		//return;
#endif

#ifdef NEED_NON_METRIC_SPACE_LIB
		RecognitionThreadPool::nnMethod = NON_METRIC_SPACE_LIB;
		runOneTest("NON_METRIC_SPACE_LIB", recognitionThreadPool, testImages, resFile);
		//return;
#endif
		
#ifdef NEED_SMALL_WORLD
		RecognitionThreadPool::nnMethod = SMALL_WORLD;
		runOneTest("SMALL_WORLD", recognitionThreadPool, testImages, resFile);
		//return;
#endif

#ifdef NEED_DEM
	RecognitionThreadPool::nnMethod=DEM;
#ifdef USE_PARALLEL_DEM
	//std::cout<<"high="<<high<<"\n";
	runOneTest("DEM", recognitionThreadPool, testImages, resFile);
#else
	directedEnumeration.setImageCountToCheck(imageCountToCheck);
    QueryPerformanceCounter(&start); 
    errorsCount=0;
    float avgCheckedPercent=0;

	for(iter=testImages.begin();iter!=testImages.end();++iter){
		FaceImage* test=*iter;
        int ind=directedEnumeration.recognize(test);
		bestImage=dbImages[ind];
        if(test->personName!=bestImage->personName){
            ++errorsCount;
            //std::cout<<test->fileName<<" "<<faceImages[ind]->personName<<" "<<faceImages[ind]->distance(test)<<" "<<faceImages[ind]->distanceToTest<<std::endl;
        }
        //avgCheckedPercent+=directedEnumeration.getCheckedPercent();
    }
    QueryPerformanceCounter(&end); 
	delta_milliseconds = (float)(end.QuadPart - start.QuadPart) / freq.QuadPart*1000000.0; 

    std::cout<<"DEM error="<<100.*errorsCount/testImages.size()<<
            //" avgChecked="<<avgCheckedPercent/testImages.size()<<
			" time="<<delta_milliseconds<<" us, rel="<<
		delta_milliseconds/testImages.size()<<" us"<<std::endl;
	resFile<<'\t'<<100.*errorsCount/testImages.size()<<'\t'<<delta_milliseconds/testImages.size();
	resFile<<std::endl;
#endif
#endif
	}
	resFile.close();
}


#if 0
//#define TRAIN_SVM
#include "svm.h"

#include <opencv/highgui.h>
#include <opencv/ml.h>
static void print_null(const char *s) { }
void test_bf(const vector<FaceImage*>& dbImages, const vector<FaceImage*>& testImages){
	using namespace cv;
	Histos<float> distance;
	const int num_of_features = POINTS_IN_H*POINTS_IN_W;
	float distances[num_of_features];
	//cout << "diff=" << testImages[0]->ptr_histos_diff() << " "<<FEATURES_COUNT << endl;

	int num_of_positives = 0, num_of_negatives = 0;
	for (int i = 0; i < dbImages.size(); ++i){
		for (int j = 0; j < dbImages.size(); ++j){
			if (dbImages[i]->personName == dbImages[j]->personName)
				++num_of_positives;
			else
				++num_of_negatives;
		}
	}

	int num_of_train_data = num_of_positives + num_of_negatives;

	cout << num_of_train_data << endl;

#ifdef TRAIN_SVM

	svm_parameter param;
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 1.0 / num_of_features;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	svm_set_print_string_function(print_null);

	svm_problem prob;
	prob.l = num_of_train_data;
	prob.y = new double[prob.l];
	prob.x = new svm_node*[prob.l];
	svm_node* x_space = new svm_node[num_of_train_data*(num_of_features + 1)];
#else
	CvMat* labelsMat = cvCreateMat(num_of_train_data, 1, CV_32FC1);
	CvMat* trainingDataMat = cvCreateMat(num_of_train_data, num_of_features, CV_32FC1);
#endif
	cout << "start distance calculation\n";

	int ind = 0, ind1 = 0;
	for (int i = 0; i < dbImages.size(); ++i){
		const float* dbFeatures1 = dbImages[i]->getFeatures();
		for (int j = 0; j < dbImages.size(); ++j){
			const float* dbFeatures2 = dbImages[j]->getFeatures();
			distance(dbFeatures2, dbFeatures1, FEATURES_COUNT, distances);
			if (dbImages[i]->personName == dbImages[j]->personName){
#ifdef TRAIN_SVM
				prob.y[ind] = 1;
#else
				CV_MAT_ELEM(*labelsMat, float, ind, 0) = 1;
#endif
			}
			else{
#ifdef TRAIN_SVM
				prob.y[ind] = 0;
#else
				CV_MAT_ELEM(*labelsMat, float, ind, 0) = 0;
#endif
			}
#ifdef TRAIN_SVM
			prob.x[ind] = &x_space[ind1];

			for (int k = 0; k<num_of_features; ++k){
				prob.x[ind][k].index = k + 1;
				prob.x[ind][k].value = distances[k];
				++ind1;
			}
			x_space[ind1++].index = -1;
#else
			for (int fi = 0; fi < num_of_features; ++fi){
				CV_MAT_ELEM(*trainingDataMat, float, ind, fi) = distances[fi];
			}
#endif
			++ind;
		}
	}
	cout << "start train\n";

#ifdef TRAIN_SVM
	svm_model *model = svm_train(&prob, &param);
	delete[] x_space;
	delete[] prob.x;
	delete[] prob.y;
#else
	CvMat* var_type = cvCreateMat(num_of_features + 1, 1, CV_8U);
	cvSet(var_type, cvScalarAll(CV_VAR_ORDERED));
	cvSetReal1D(var_type, num_of_features, CV_VAR_CATEGORICAL);
	CvBoost boost;
	boost.train(trainingDataMat, CV_ROW_SAMPLE, labelsMat, 0, 0, var_type, 0, CvBoostParams(CvBoost::DISCRETE, 10, 0.95, 1, false, 0));
	cvReleaseMat(&trainingDataMat);
	cvReleaseMat(&labelsMat);
#endif
	cout << "end train\n";

	int error_count = 0;
	double prob_estimates[2];
#ifdef TRAIN_SVM
	svm_node x[num_of_features + 1];
	x[num_of_features].index = -1;
#else
	CvMat* weak_responses = cvCreateMat(1, boost.get_weak_predictors()->total, CV_32FC1);
	CvMat* queryMat = cvCreateMat(1, num_of_features, CV_32FC1);
#endif

	clock_t start_time = clock();
	for (int i = 0; i<testImages.size(); ++i){
		const float* testFeatures = testImages[i]->getFeatures();
		float minDist = FLT_MAX;
		int best_ind = -1;
		double max_sum = -DBL_MAX;
		for (int j = 0; j < dbImages.size(); ++j){
			const float* dbFeatures = dbImages[j]->getFeatures();
#if 0
			float f = distance(dbFeatures, testFeatures, FEATURES_COUNT);
			if (f < minDist){
				minDist = f;
				best_ind = j;
			}
#else
			distance(dbFeatures, testFeatures, FEATURES_COUNT, distances);
#ifdef TRAIN_SVM
			for (int k = 0; k<num_of_features; ++k){
				x[k].index = k + 1;
				x[k].value = distances[k];
			}
			//std::cout << "pred=" << svm_predict(model, x) << '\n';
			float predict_label = svm_predict_probability(model, x, prob_estimates);
			std::cout << "res=" << predict_label << '\n';
			for (int i = 0; i<2; ++i){
				std::cout << i << '(' << prob_estimates[i] << ") ";
			}
			std::cout << "\n";
#else
			for (int fi = 0; fi < num_of_features; ++fi){
				CV_MAT_ELEM(*queryMat, float, 0, fi) = distances[fi];
			}
			boost.predict(queryMat, 0, weak_responses);
			double sum = cvSum(weak_responses).val[0];
			//cout << sum << endl;
			if (max_sum < sum)
			{
				max_sum = sum;
				best_ind = j;
			}
#endif
			//cout << testImages[i]->personName << " " << dbImages[j]->personName << endl;
#endif
		}
		if (dbImages[best_ind]->personName != testImages[i]->personName){
			++error_count;
			//cout << best_ind << " " << dbImages[best_ind]->personName << " " << dbImages[i]->personName << endl;
		}
	}
	clock_t end_time = clock();
	cout << "total=" << ((double)(end_time - start_time)) / (testImages.size()*CLOCKS_PER_SEC) << "error=" << (100.0 * error_count / testImages.size()) << endl;

#ifdef TRAIN_SVM
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
#else
	cvReleaseMat(&var_type);
	cvReleaseMat(&queryMat);
	cvReleaseMat(&weak_responses);
#endif
}
#endif
