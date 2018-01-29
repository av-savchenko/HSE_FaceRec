#include "FaceImage.h"
#include "db.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>
#include <chrono>
using namespace std;


#define print_endl cout<<endl;


typedef std::map<std::string, std::vector<std::vector<FaceImage*>>> MapOfVideos;

const std::string VIDEO_DIR = BASE_DIR +
#if DB_USED==USE_LFW
"YTF_cropped_10th";
#else
//"gallery_media";
"probe";
#endif

const std::string VIDEO_FEATURES_FILE = VIDEO_DIR +
#ifdef USE_RGB_DNN
//"_vgg_dnn_features.txt";
"_vgg2_dnn_features.txt";
//"_res101_dnn_features.txt";
//"_ydwen_dnn_features.txt";
#else
"_lcnn_dnn_features.txt";
#endif

#if 0 //vgg L2, vgg2
const double DISTANCE_WEIGHT = 1000;
#elif 1 //lcnn, caffe center face, resnet-101
const double DISTANCE_WEIGHT = 100;
#elif 1 //mobilenet, resnet L2 & chisq, vgg chisq, LBPH/HOG
const double DISTANCE_WEIGHT = 10000;
#else 
const double DISTANCE_WEIGHT = 100000;
#endif
const double FEAT_COUNT_TO_N_RATIO = (FEATURES_COUNT - 1) / DISTANCE_WEIGHT;

void loadVideosFromDir(MapOfVideos& videoMap) {
	WIN32_FIND_DATA ffd, ffd1, ffd2;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	HANDLE hFindDir = INVALID_HANDLE_VALUE;
	HANDLE hFindFile = INVALID_HANDLE_VALUE;
	int num_of_persons_processed = 0;

	std::cout << "start load" << std::endl;

	hFind = FindFirstFile((VIDEO_DIR + "\\*").c_str(), &ffd);
	if (INVALID_HANDLE_VALUE == hFind) {
		std::cout << "no dirs. Return." << std::endl;
		return;
	}
	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			std::string dirName = ffd.cFileName;
			if (dirName[0] == '.')
				continue;
			std::string videoDirName = VIDEO_DIR + "\\" + dirName + "\\";
			hFindDir = FindFirstFile((videoDirName + "*").c_str(), &ffd1);
			if (INVALID_HANDLE_VALUE == hFindDir) {
				std::cout << "no video dirs." << std::endl;
			}
			std::vector<std::vector<FaceImage*>> currentVideos;
			do
			{
				if (ffd1.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					if (ffd1.cFileName[0] == '.')
						continue;
					
					std::string fullDirName = videoDirName + ffd1.cFileName + "\\";
					hFindFile = FindFirstFile((fullDirName + "*").c_str(), &ffd2);
					if (INVALID_HANDLE_VALUE == hFindFile) {
						std::cout << "no files." << std::endl;
					}
					std::vector<FaceImage*> currentDirFaces;
					int face_counter = 0;
					do
					{
						if (((ffd2.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) &&
							((ffd2.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN) == 0))
						{
							currentDirFaces.push_back(new FaceImage((fullDirName + ffd2.cFileName).c_str(), dirName));
							//break;
						}
					} while (FindNextFile(hFindFile, &ffd2) != 0);
					currentVideos.push_back(currentDirFaces);
				}	
			} while (FindNextFile(hFindDir, &ffd1) != 0);

			videoMap.insert(std::make_pair(dirName, currentVideos));
#ifdef USE_DNN_FEATURES
			if (++num_of_persons_processed % 10 == 0)
				std::cout << "first stage " << num_of_persons_processed << '\n';
#endif
		}
	} while (FindNextFile(hFind, &ffd) != 0);

#ifdef USE_DNN_FEATURES
	ofstream of(VIDEO_FEATURES_FILE);
	if (of) {
		cout << "begin write file\n";
		for (MapOfVideos::iterator iter = videoMap.begin(); iter != videoMap.end(); ++iter) {
			vector<vector<FaceImage*>>& person_videos = iter->second;
			of << iter->first << endl;
			of << person_videos.size() << endl;
			for(int video_ind=0; video_ind<person_videos.size();++video_ind){
				of << person_videos[video_ind].size() << endl;
				for (FaceImage* face : person_videos[video_ind]) {
					of << face->fileName << endl;
					for (int i = 0; i < FEATURES_COUNT; ++i)
						of << face->getFeatures()[i] << ' ';
					of << endl;

					std::vector<float>& featureVector = face->getFeatureVector();
					float sum = 0;
					for (int i = 0; i < FEATURES_COUNT; ++i)
						sum += featureVector[i] * featureVector[i];
					sum = sqrt(sum);
					for (int i = 0; i < FEATURES_COUNT; ++i)
						featureVector[i] /= sum;
				}
			}
		}
		cout << "end write file\n";
		of.close();
	}
#endif
}
void loadVideos(MapOfVideos& dbVideos, std::string video_file= VIDEO_FEATURES_FILE) {
	int total_images = 0, total_videos = 0;
#if defined(USE_DNN_FEATURES)
	ifstream ifs(video_file);
	if (!ifs)
#endif
	{
		cout << "loading images from dir "<< VIDEO_DIR <<"\n";
		loadVideosFromDir(dbVideos);
		return;
	}
#if defined(USE_DNN_FEATURES)
	while (ifs) {
		/*if (dbVideos.size() > 100)
			break;*/
		std::string fileName, personName, feat_str;
		if (!getline(ifs, personName))
			break;
		personName.erase(0, personName.find_first_not_of(" \t\n\r\f\v\r\n"));
		int videos_count;
		ifs >> videos_count;
		dbVideos.insert(std::make_pair(personName, vector<vector<FaceImage*>>()));
		vector<vector<FaceImage*>>& person_videos = dbVideos[personName];
		person_videos.resize(videos_count);

		for (int i = 0; i < videos_count; ++i) {
			int frames_count;
			ifs >> frames_count;
			if (!getline(ifs, fileName))
				break;
#if 1

			for (int j = 0; j < frames_count; ++j) {
				if (!getline(ifs, fileName))
					break;
				if (!getline(ifs, feat_str))
					break;
				istringstream iss(feat_str);
				vector<float> features(FEATURES_COUNT);
				for (int i = 0; i < FEATURES_COUNT; ++i) {
					iss >>features[i];
				}
				person_videos[i].push_back(new FaceImage(fileName, personName, features));
				++total_images;

				//cout << fileName << ' ' << personName << ' ' << features[0] << ' ' << features[FEATURES_COUNT - 1] << '\n';
			}
#else
			vector<float> features(FEATURES_COUNT);

			for (int j = 0; j < frames_count; ++j) {
				if (!getline(ifs, fileName))
					break;
				if (!getline(ifs, feat_str))
					break;
				istringstream iss(feat_str);
				for (int i = 0; i < FEATURES_COUNT; ++i) {
					float f;
					iss >> f;
					features[i] += f;
				}

				//cout << fileName << ' ' << personName << ' ' << features[0] << ' ' << features[FEATURES_COUNT - 1] << '\n';
			}
			for (int i = 0; i < FEATURES_COUNT; ++i) {
				features[i] /= frames_count;
			}
			person_videos[i].push_back(new FaceImage(fileName, personName, features));
			++total_images;

#endif
		}
		total_videos += videos_count;
		dbVideos.insert(std::make_pair(personName, person_videos));
	}
	ifs.close();
	cout << "total size=" << dbVideos.size() << " totalVideos=" << total_videos << " totalImages=" << total_images;
	print_endl;
#endif
}




unordered_map<string, unordered_map<string, float> > model_distances, model_probabs;
typedef unordered_map<FaceImage*, unordered_map<FaceImage*, float>> distance_map;
void compute_model_distances(MapOfFaces& totalImages, const vector<string>& commonNames, distance_map* dm=0) {
	double avg_same_dist = 0;
	int same_count = 0;
	for (string name : commonNames) {
		for (string otherName : commonNames) {
			float classDistance = 10000000.f;
			if (name == otherName) {
				classDistance = 0;
				int size = totalImages[name].size();
				if (size > 1) {
					for (FaceImage* face : totalImages[name]) {
						for (FaceImage* otherFace : totalImages[name]) {
							float dist = (dm!=0)?(*dm)[otherFace][face]:otherFace->distance(face);
							classDistance += dist;
						}
					}
					classDistance /= size*(size - 1);
					avg_same_dist += classDistance;
					++same_count;
				}
			}
			else {
				for (FaceImage* face : totalImages[name]) {
					for (FaceImage* otherFace : totalImages[otherName]) {
						float dist = otherFace->distance(face);
						if (dist < classDistance)
							classDistance = dist;
					}
				}
			}
			model_distances[name].insert(make_pair(otherName, classDistance));
		}
	}
	if (same_count > 0)
		avg_same_dist /= same_count;

	//normalize
	for (string name : commonNames) {
		double max_val = 0, sum = 0;
		if (model_distances[name][name] == 0)
			model_distances[name][name] = avg_same_dist;
		for (string otherName : commonNames) {
			double prob = exp(-DISTANCE_WEIGHT* model_distances[name][otherName]);
			model_probabs[name][otherName] = prob;
			sum += prob;
			if (max_val < prob)
				max_val = prob;
		}
		max_val = sum;
		if (max_val > 0) {
			for (string otherName : commonNames) {
				model_probabs[name][otherName] /= max_val;
			}
		}
	}
}
template <typename T, template <typename, typename, typename...> class MapT, typename... AddParams>
inline void sortDistances(vector<pair<string, T> >& distanceToClass, const MapT<string, T, AddParams...>& classDistances) {
	transform(classDistances.begin(), classDistances.end(), distanceToClass.begin(),
		[](const pair<std::string, T>& p) {
		return make_pair(p.first, p.second);
	}
	);

	sort(distanceToClass.begin(), distanceToClass.begin() + classDistances.size(),
		[](const pair<string, T>& lhs, const pair<string, T>& rhs) {
		return lhs.second < rhs.second; }
	);
}

int processVideo_NN(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName, bool use_MAP=false) {
	int frames_count = videoClassDistances.size();
	int class_count = videoClassDistances[0].size();
	unordered_map<string, double> avg_frames, current_frame;
	vector<pair<string, double>> distanceToClass(class_count);

	if (!use_MAP) {
		avg_frames = videoClassDistances[0];
		for (int i = 1; i < frames_count; ++i) {
			for (auto person : videoClassDistances[i]) {
				avg_frames[person.first] += person.second;
			}
		}
	}
	else {
		for (auto person : videoClassDistances[0])
			avg_frames.insert(make_pair(person.first, 0));

		for (int i = 0; i < frames_count; ++i) {
			double sum = 0;
			for (auto person : videoClassDistances[i]) {
				sum += exp(-DISTANCE_WEIGHT*person.second);
			}
			for (auto person : videoClassDistances[i]) {
				avg_frames[person.first] -= exp(-DISTANCE_WEIGHT*person.second) / sum;
			}
		}
	}
	sortDistances(distanceToClass, avg_frames);
	int res = 0;
	for (; res < distanceToClass.size() && distanceToClass[res].first != correctClassName; ++res)
		;
	return res;
}
static int regularizer_weight = 8; //2;
//1;
//28;
//6;
//40;

static int analyzedModelsCount = 64;// 128;// 5;
							 //-1 = avg_frame_dists.size();
int processVideo_ML_dist(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName) {
	int frames_count = videoClassDistances.size();
	int class_count = videoClassDistances[0].size();
	double log_weight = 1.0 / (class_count)*regularizer_weight;
	unordered_map<string, double> avg_frames, current_frame, avg_frame_dists;

	avg_frame_dists = videoClassDistances[0];
	int init_count = frames_count;
					//min(5, frames_count);
	for (int i = 1; i < init_count; ++i) {
		for (auto& person : videoClassDistances[i])
			avg_frame_dists[person.first] += person.second;
	}
	for (auto& frame : avg_frame_dists)
		frame.second /= init_count;

#if 0
	videoClassDistances.clear();
	videoClassDistances.push_back(avg_frame_dists);
	frames_count = 1;
#endif

	if(analyzedModelsCount==-1)
		analyzedModelsCount= avg_frame_dists.size();
	
	set<string> positions;
	//avg_frames.clear();

	vector<pair<string, double>> distanceToClass(class_count);
	sortDistances(distanceToClass, avg_frame_dists);
	for (int i = 0; i < analyzedModelsCount; ++i) {
		positions.insert(distanceToClass[i].first);
	}

	for (auto& person : videoClassDistances[0])
		avg_frame_dists[person.first] = 0;
	for (int i = 0; i < frames_count; ++i) {
		double sum = 0, sum1 = 0;
		for (string name : positions) {
			auto& model_row = model_distances[name];
			double log_prob = 0;
			for (auto& person : videoClassDistances[i]) {
				if (person.first == name)
					continue;
				double expected_dist = model_row[person.first];
				double cur_dist = person.second;
				double tmp = (expected_dist - cur_dist)*(expected_dist - cur_dist) / (4 * expected_dist);
				log_prob += tmp;
			}
			current_frame[name] = exp(-DISTANCE_WEIGHT*(videoClassDistances[i][name] + log_prob*log_weight));
			sum += current_frame[name];
		}
		for (string name : positions) {
			avg_frames[name] -= current_frame[name] / sum;
		}
	}

	sortDistances(distanceToClass, avg_frames);
	int res = 0;
	for (; res < distanceToClass.size() && distanceToClass[res].first != correctClassName; ++res)
		;
	return res;
}



enum class NN_Method { SIMPLE_NN=0, MAP, ML_DIST };
void test_recognition_method(NN_Method method, vector<FaceImage*>& faceImages, MapOfVideos& videos) {
	int errorsCount = 0, totalVideos = 0, totalFrames = 0;
	unordered_map<string, int> class_errors;
	cout << "start" << flush;
	auto t1 = chrono::high_resolution_clock::now();
	for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter) {
		for (vector<FaceImage*>& video : iter->second) {
			vector<unordered_map<string, double>> videoClassDistances(video.size());
			for (int ind = 0; ind < video.size(); ++ind)
			{
				unordered_map<string, double>& frameDistances = videoClassDistances[ind];
				for (int j = 0; j < faceImages.size(); ++j) {
					float tmpDist = video[ind]->distance(faceImages[j]);
					bool classNameExists = frameDistances.find(faceImages[j]->personName) != frameDistances.end();
					if (!classNameExists || (classNameExists && frameDistances[faceImages[j]->personName] > tmpDist))
						frameDistances[faceImages[j]->personName] = tmpDist;
				}
			}
			int pos = -1;
			switch (method) {
			case NN_Method::SIMPLE_NN:
				pos = processVideo_NN(videoClassDistances, iter->first);
				break;
			case NN_Method::MAP:
				pos = processVideo_NN(videoClassDistances, iter->first,true);
				break;
			case NN_Method::ML_DIST:
				pos = processVideo_ML_dist(videoClassDistances, iter->first);
				break;
			}
			if (pos != 0) {
				cout << "\rorig=" << std::setw(35) << iter->first << " pos=" << std::setw(4) << pos << flush;
				//cout << "orig="  << iter->first << " pos=" << pos << " file="<< video[0]->fileName <<'\n';
				++errorsCount;
				++class_errors[iter->first];
			}
			++totalVideos;
			totalFrames += video.size();
			/*if (totalVideos > 100)
			break;*/
		}
	}
	auto t2 = chrono::high_resolution_clock::now();
	float total = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	double recall = 0;
	for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter) {
		if (class_errors.find(iter->first) == class_errors.end())
			recall += 100;
		else
			recall += 100 - 100.0*class_errors[iter->first] / iter->second.size();
	}

	cout << endl<<"method "<< (int)method<<" video error rate=" << (100.0*errorsCount / totalVideos) << " (" << errorsCount << " out of "
		<< totalVideos << ") recall=" << (recall / videos.size()) << " avg time=" << (total / totalFrames) << "(ms) prev time=" << (total / totalVideos) << "\n";

}
//#define USE_SVM
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;
void test_complex_classifier(vector<FaceImage*>& faceImages, MapOfVideos& videos, unordered_map<string, int>& class2no) {
	cv::Ptr<cv::ml::SVM> svmClassifier;
	int num_of_cont_features = FEATURES_COUNT;
	Mat labelsMat(faceImages.size(), 1, CV_32S);
	Mat trainingDataMat(faceImages.size(), num_of_cont_features, CV_32FC1);
	for (int i = 0; i<faceImages.size(); ++i) {
		for (int fi = 0; fi<num_of_cont_features; ++fi) {
			trainingDataMat.at<float>(i, fi) = faceImages[i]->getFeatures()[fi];
		}
		labelsMat.at<int>(i, 0) = class2no[faceImages[i]->personName];
	}

	// Set up SVM's parameters
	svmClassifier = SVM::create();
	svmClassifier->setType(SVM::C_SVC);
	svmClassifier->setKernel(SVM::LINEAR);
	//svmClassifier->setKernel(SVM::RBF);
	svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-8));
	svmClassifier->setC(10);

	// Train the SVM
	svmClassifier->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	cout << "start" << flush;
	int errorsCount = 0, totalVideos = 0, totalFrames = 0;
	unordered_map<string, int> class_errors;
	auto t1 = chrono::high_resolution_clock::now();
	for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter) {
		for (vector<FaceImage*>& video : iter->second) {
			vector<unordered_map<string, double>> videoClassDistances(video.size());
			unordered_map<string, float> classDistances;
			for (int ind = 0; ind < video.size(); ++ind)
			{
				Mat queryMat(1, FEATURES_COUNT, CV_32FC1);
				for (int fi = 0; fi<FEATURES_COUNT; ++fi) {
					queryMat.at<float>(0, fi) = video[ind]->getFeatures()[fi];
				}

				float response = svmClassifier->predict(queryMat);
				for (auto& class_no : class2no) {
					if (fabs(class_no.second - response) < 0.1)
						classDistances[class_no.first] += 0;
					else
						classDistances[class_no.first] += 1;
				}
			}
			string bestClass = "";
			float bestDist = 100000;
			for (auto& class_dist : classDistances) {
				if (class_dist.second < bestDist) {
					bestDist = class_dist.second;
					bestClass = class_dist.first;
				}
			}
			if (bestClass != iter->first) {
				cout << "\rorig=" << std::setw(15) << iter->first << " bestClass=" << std::setw(30) << bestClass << flush;
				++errorsCount;
				++class_errors[iter->first];
			}
			++totalVideos;
			totalFrames += video.size();
		}
	}
	auto t2 = chrono::high_resolution_clock::now();
	float total = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	double recall = 0;
	for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter) {
		if (class_errors.find(iter->first) == class_errors.end())
			recall += 100;
		else
			recall += 100 - 100.0*class_errors[iter->first] / iter->second.size();
	}

	cout << endl<<"SVM video error rate=" << (100.0*errorsCount / totalVideos) << " (" << errorsCount << " out of "
		<< totalVideos << ") recall=" << (recall / videos.size()) << " avg time=" << (total / totalFrames) << "(ms) prev time=" << (total / totalVideos) << "\n";

}
void testVideoRecognition() {
	MapOfFaces totalImages;
#if 1
	loadFaces(totalImages);
#else
	MapOfVideos training_videos;
	loadVideos(training_videos, BASE_DIR + "gallery_media_vgg2_dnn_features.txt");
	for (MapOfVideos::iterator iter = training_videos.begin(); iter != training_videos.end(); ++iter) {
		vector<FaceImage*>& faces = totalImages[iter->first];
		for (vector<FaceImage*>& video : iter->second) {
			faces.push_back(video[0]);
		}
	}
#endif
	vector<string> dbNames;
	dbNames.reserve(totalImages.size());
	for (auto keyValue : totalImages) {
		dbNames.push_back(keyValue.first);
	}

	MapOfVideos videos;
	loadVideos(videos);

	vector<string> videoNames;
	videoNames.reserve(videos.size());
	for (auto keyValue : videos) {
		videoNames.push_back(keyValue.first);
	}

	vector<string> commonNames(videos.size());
	sort(videoNames.begin(), videoNames.end());
	sort(dbNames.begin(), dbNames.end());
	auto it = std::set_intersection(videoNames.begin(), videoNames.end(), dbNames.begin(), dbNames.end(), commonNames.begin());
	commonNames.resize(it - commonNames.begin());
	
	cout << "still names size=" << dbNames.size() << " video names size=" << videoNames.size() << " common names size=" << commonNames.size();
	print_endl;

	vector<string> listToRemove;
	std::set_symmetric_difference(videoNames.begin(), videoNames.end(), dbNames.begin(), dbNames.end(), back_inserter(listToRemove));
	for (string needToRemove : listToRemove) {
		totalImages.erase(needToRemove);
		videos.erase(needToRemove);
	}

	compute_model_distances(totalImages, commonNames);

	vector<FaceImage*> faceImages;
	unordered_map<string, int> class2no;
	int cur_class_no = 0;
	for (MapOfFaces::iterator iter = totalImages.begin(); iter != totalImages.end(); ++iter) {
		class2no[iter->first] = ++cur_class_no;
		for (FaceImage* face : iter->second)
			faceImages.push_back(face);
	}
	test_recognition_method(NN_Method::SIMPLE_NN, faceImages, videos);
	test_recognition_method(NN_Method::MAP, faceImages, videos);
	test_complex_classifier(faceImages, videos, class2no);
	vector<int> regs({ 8 });// 8, 80, 100, 120, 64, 40, 28, 4, 16});
	for (int reg : regs) {
	//for (int reg = 1; reg <= 8; reg += 1) {
	//for (int reg = 8; reg >= 2; reg -= 2) {
		regularizer_weight = reg;
		cout << "regularizer_weight=" << regularizer_weight << endl;
		//for (int M = 1; M <= 256; M *= 2) 
		{
			analyzedModelsCount = 16;// M;
			cout << "analyzedModelsCount=" << analyzedModelsCount << endl;
			test_recognition_method(NN_Method::ML_DIST, faceImages, videos);
		}
	}
}

void testRecognitionOfMultipleImages() {
	//srand(13);
	MapOfFaces totalImages, faceImages, testImages;
	loadFaces(totalImages);
	
	distance_map dist_map;
	for (auto& person1 : totalImages) {
		for (auto& face1 : person1.second) {
			for (auto& person2 : totalImages) {
				for (auto& face2 : person2.second) {
					dist_map[face1][face2] = face1->distance(face2);
				}
			}
		}
	}
	cout << " distance map computed\n";

	vector<string> class_names;
	class_names.reserve(totalImages.size());
	for (auto& image : totalImages)
	class_names.push_back(image.first);

	const int FRAMES_COUNT = 1;

	const int TESTS = 10;
	double totalTestsErrorRate = 0, errorRateVar = 0;
	for (int testCount = 0; testCount < TESTS; ++testCount) {
		int errorsCount = 0;
		getTrainingAndTestImages(totalImages, faceImages, testImages);
		compute_model_distances(faceImages, class_names,&dist_map);

		float bestDist, tmpDist;
		string bestClass;
		vector<unordered_map<string, double>> videoClassDistances;
		int test_count = 0;
		for (auto& testPersonImages : testImages) {
			for (int i = 0; i < testPersonImages.second.size(); ++i, ++test_count) {
				videoClassDistances.clear();
				for (int frame = 0; frame < FRAMES_COUNT; ++frame) {
					int ind = (frame == 0) ? i : rand() % testPersonImages.second.size();
					videoClassDistances.push_back(unordered_map<string, double>());
					unordered_map<string, double>& classDistances = videoClassDistances.back();

					for (auto& dbPersonImages : faceImages) {
						bestDist = 100000;
						for (int j = 0; j < dbPersonImages.second.size(); ++j) {
							//tmpDist = testPersonImages.second[ind]->distance(dbPersonImages.second[j]);
							tmpDist = dist_map[testPersonImages.second[ind]][dbPersonImages.second[j]];
							if (tmpDist < bestDist) {
								bestDist = tmpDist;
							}
						}
						classDistances.insert(make_pair(dbPersonImages.first, bestDist));
					}
				}
				int res = processVideo_ML_dist(videoClassDistances, testPersonImages.first);
				if (res != 0)
					++errorsCount;
			}
		}


		double errorRate = 100.*errorsCount / test_count;
		std::cout << "test=" << testCount << " test_count=" << test_count << " error=" << errorRate << std::endl;
		totalTestsErrorRate += errorRate;
		errorRateVar += errorRate * errorRate;

	}
	totalTestsErrorRate /= TESTS;
	errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS));
	std::cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar << endl;
}