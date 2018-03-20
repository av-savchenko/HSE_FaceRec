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
#include <sstream>
#include <algorithm>
#include <numeric>
using namespace std;


#define print_endl cout<<endl;


typedef std::map<std::string, std::vector<std::vector<FaceImage*>>> MapOfVideos;

const std::string VIDEO_DIR = BASE_DIR +
#if DB_USED==USE_LFW
"YTF_cropped_10th";
#else
//"gallery_media";
"probe_equal";
#endif

const std::string VIDEO_FEATURES_FILE = VIDEO_DIR + 
#if DB_USED!=USE_LFW
"_equal"+
#endif
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
#elif 0 //lcnn, caffe center face, resnet-101
const double DISTANCE_WEIGHT = 100;
#elif 1 //mobilenet, resnet L2 & chisq, vgg chisq, LBPH/HOG
const double DISTANCE_WEIGHT = 10000;
#else 
const double DISTANCE_WEIGHT = 100000;
#endif

//#define AGGREGATE_VIDEOS

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
		/*if (dbVideos.size() > 10)
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
#ifdef USE_MEDIA_ID
			vector<float> avg_features(FEATURES_COUNT);
			string prev_media_id = "";
			int num_of_frames_in_media = 0;
#endif
#ifdef AGGREGATE_VIDEOS
			vector<float> features(FEATURES_COUNT);

			for (int j = 0; j < frames_count; ++j) {
				if (!getline(ifs, fileName))
					break;
				if (!getline(ifs, feat_str))
					break;
				istringstream iss(feat_str);
				for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
					float f;
					iss >> f;
					features[i] += f;
				}

			}
			for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
				features[ind] /= frames_count;
			}
			person_videos[i].push_back(new FaceImage(fileName, personName, features));
			++total_images;

#else
			for (int j = 0; j < frames_count; ++j) {
				if (!getline(ifs, fileName))
					break;
				if (!getline(ifs, feat_str))
					break;
				istringstream iss(feat_str);
				vector<float> features(FEATURES_COUNT);
				for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
					iss >> features[ind];
				}
#ifdef USE_MEDIA_ID
				size_t slash_ind = fileName.rfind("\\");
				size_t dot_ind = fileName.rfind(".");
				size_t underscore_ind = fileName.rfind("_");
				string media_id = "";
				if (slash_ind != string::npos) {
					size_t end_ind = dot_ind;
					if (underscore_ind != string::npos && underscore_ind > slash_ind)
						end_ind = underscore_ind;
					media_id = fileName.substr(slash_ind + 1, end_ind - slash_ind - 1);
				}
				//cout << fileName << ' ' << media_id << '\n';
				//cout << i<<' '<<fileName << ' ' << media_id << ' ' << prev_media_id << ' ' << num_of_frames_in_media << '\n';
				if (prev_media_id != media_id) {
					if (num_of_frames_in_media > 0) {
						for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
							avg_features[ind] /= num_of_frames_in_media;
						}
						person_videos[i].push_back(new FaceImage(fileName, personName, avg_features));
						for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
							avg_features[ind] = 0;
						}
						num_of_frames_in_media = 0;
						++total_images;
					}
					prev_media_id = media_id;
				}
				++num_of_frames_in_media;
				for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
					avg_features[ind] += features[ind];
				}
#else
				person_videos[i].push_back(new FaceImage(fileName, personName, features));
				++total_images;
#endif
			}
#ifdef USE_MEDIA_ID
			string media_id = prev_media_id;
			//cout << "end "<<fileName << ' ' << media_id << ' ' << prev_media_id << ' ' << num_of_frames_in_media << '\n';
			if (num_of_frames_in_media > 0) {
				for (int ind = 0; ind < FEATURES_COUNT; ++ind) {
					avg_features[ind] /= num_of_frames_in_media;
				}
				person_videos[i].push_back(new FaceImage(fileName, personName, avg_features));
				++total_images;
			}
#endif
#endif
		}
		/*for (int i = 0; i < videos_count; ++i){
			cout << i << ' ' << person_videos[i].size() << '\n';
		}*/
		total_videos += videos_count;
		dbVideos.insert(std::make_pair(personName, person_videos));
	}
	ifs.close();
	cout << "total size=" << dbVideos.size() << " totalVideos=" << total_videos << " totalImages=" << total_images;
	print_endl;
#endif
}

static unordered_map<string, string> closestFaces;




class Classifier {
public:
	Classifier():pDbImages(0) {}
	virtual ~Classifier(){}

	virtual void train(MapOfFaces& pDb, const vector<string>& commonNames) { pDbImages = &pDb; }
	virtual int get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName) = 0;
	string get_name() { return name; }
private:
	string name;

protected:
	void set_name(string n) { name = n; }
	MapOfFaces* pDbImages;
};

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

//=================================
enum class FusionMethod{MEAN_DIST=0, MAP, MAX_POSTERIOR};
class BruteForceClassifier :public Classifier {
public:
	BruteForceClassifier(FusionMethod fusion = FusionMethod::MEAN_DIST, int dist_weight=DISTANCE_WEIGHT);
	virtual int get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName);

	virtual int processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName);
protected:
	const FusionMethod fusion_method;
	const int distance_weight;
};
BruteForceClassifier::BruteForceClassifier(FusionMethod fusion, int dist_weight):fusion_method(fusion), distance_weight(dist_weight)
{
	ostringstream os;
	os << "BF "<< int(fusion_method);
	set_name(os.str());
}
int BruteForceClassifier::get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName) 
{
	MapOfFaces& faceImages= *pDbImages;
	vector<unordered_map<string, double>> videoClassDistances(video.size());
	for (int ind = 0; ind < video.size(); ++ind)
	{
		unordered_map<string, double>& frameDistances = videoClassDistances[ind];
		for (MapOfFaces::iterator iter = faceImages.begin(); iter != faceImages.end(); ++iter) {
			for (FaceImage* face : iter->second) {
				float tmpDist = video[ind]->distance(face);
				bool classNameExists = frameDistances.find(face->personName) != frameDistances.end();
				if (!classNameExists || (classNameExists && frameDistances[face->personName] > tmpDist)) {
					frameDistances[face->personName] = tmpDist;
					closestFaces[face->personName] = face->fileName;
				}
			}
		}
	}
	return processVideo(videoClassDistances, correctClassName);
}

int BruteForceClassifier::processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName) {
	int frames_count = videoClassDistances.size();
	int class_count = videoClassDistances[0].size();
	unordered_map<string, double> avg_frames, current_frame;
	vector<pair<string, double>> distanceToClass(class_count);

	if (fusion_method == FusionMethod::MEAN_DIST) {
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
			double sum = 0, min_dist=1000000;
			for (auto person : videoClassDistances[i]) {
				sum += exp(-distance_weight*person.second);
				if (person.second<min_dist)
					min_dist = person.second;
			}
			for (auto person : videoClassDistances[i]) {
#if 0
					double probab = exp(-distance_weight*(person.second - min_dist));
					if (fusion_method == FusionMethod::MAP)
						avg_frames[person.first] -= probab;
					else if (fusion_method == FusionMethod::MAX_POSTERIOR) {
						if (avg_frames[person.first] > -probab)
							avg_frames[person.first] = -probab;
					}
				}
#else
				
				double probab = exp(-distance_weight*person.second) / sum;
				if (fusion_method == FusionMethod::MAP)
					avg_frames[person.first] -= probab;
				else if (fusion_method == FusionMethod::MAX_POSTERIOR) {
					/*if (avg_frames[person.first] > -probab)
						avg_frames[person.first] = -probab;*/
					avg_frames[person.first] += probab*(1 - avg_frames[person.first]);

				}
#endif
			}
		}
	}
	if (fusion_method == FusionMethod::MAX_POSTERIOR) {
		for (auto& class_dist : avg_frames) {
			class_dist.second *= -1;
		}
	}
	int res = 0;
#if 0
	sortDistances(distanceToClass, avg_frames);
	/*cout << "use_MAP="<<use_MAP << " size="<<distanceToClass.size()<<endl;
	for (int i = 0; i < min(10,(int)distanceToClass.size()); ++i) {
	cout << distanceToClass[i].first << '\t' << distanceToClass[i].second <<endl;
	}
	cout << endl;*/

	for (; res < distanceToClass.size() && distanceToClass[res].first != correctClassName; ++res)
		;
	if (distanceToClass.empty())
		res = 1;
#else
	string bestClass = "";
	float bestDist = numeric_limits<float>::max();
	for (auto& class_dist : avg_frames) {
		if (class_dist.second < bestDist) {
			bestDist = class_dist.second;
			bestClass = class_dist.first;
		}
	}
	res = (bestClass!= correctClassName);
#endif
	return res;
}

//=================================
class MLDistClassifier :public BruteForceClassifier {
public:
	//typedef unordered_map<FaceImage*, unordered_map<FaceImage*, float>> distance_map;
	typedef vector<vector<float>> distance_map;
	typedef unordered_map<FaceImage*, int> image_map;
	MLDistClassifier(double reg_weight = 8, int model_count = -1, distance_map* dm = 0, image_map* im=0);
	virtual void train(MapOfFaces& pDb, const vector<string>& commonNames);

	virtual int processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName);
private:
	int max_features;
	bool use_MAP;
	double regularizer_weight;// = 8; 2, 1, 28, 6, 40;
	int analyzed_models_count;// = 64;// 128;// 5;
										//-1 = avg_frame_dists.size();

	distance_map* dist_map;
	image_map* img_map;
	unordered_map<string, unordered_map<string, float> > model_distances;// , model_probabs;
};
MLDistClassifier::MLDistClassifier(double reg_weight, int model_count, distance_map* dm, image_map* im) : regularizer_weight(reg_weight), analyzed_models_count(model_count), dist_map(dm), img_map(im) {
	ostringstream os;
	os << "ML distances reg=" << regularizer_weight<<" analyzed_models_count="<< analyzed_models_count;
	set_name(os.str()); 
}

void MLDistClassifier::train(MapOfFaces& totalImages, const vector<string>& commonNames)
{ 
	BruteForceClassifier::train(totalImages, commonNames);
	//compute model distances
	model_distances.clear();
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
						int face_ind = (img_map != 0)?(*img_map)[face]:0;
						for (FaceImage* otherFace : totalImages[name]) {
							int other_ind = (img_map != 0) ? (*img_map)[otherFace] : 0;
							float dist = (dist_map != 0) ? (*dist_map)[other_ind][face_ind] : otherFace->distance(face);
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
					int face_ind = (img_map != 0) ? (*img_map)[face] : 0;
					for (FaceImage* otherFace : totalImages[otherName]) {
						int other_ind = (img_map != 0) ? (*img_map)[otherFace] : 0;
						float dist = (dist_map != 0) ? (*dist_map)[other_ind][face_ind] : otherFace->distance(face);
						if (dist < classDistance)
							classDistance = dist;
					}
				}
			}
			model_distances[name][otherName]= classDistance;
		}
	}
	if (same_count > 0)
		avg_same_dist /= same_count;

	//normalize
	for (string name : commonNames) {
		double max_val = 0, sum = 0;
		if (model_distances[name][name] == 0)
			model_distances[name][name] = avg_same_dist;
		/*for (string otherName : commonNames) {
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
		}*/
	}
}

int MLDistClassifier::processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName) {
	int frames_count = videoClassDistances.size();
	int class_count = videoClassDistances[0].size();
	double log_weight = regularizer_weight / (class_count);
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
	int models_count = analyzed_models_count;
	if (models_count == -1)
		models_count = avg_frame_dists.size();

	set<string> positions;
	//avg_frames.clear();

	vector<pair<string, double>> distanceToClass(class_count);
	sortDistances(distanceToClass, avg_frame_dists);
	bool correctlyRecognized=true;
	for (int i = 0; i < models_count; ++i) {
		positions.insert(distanceToClass[i].first);
		/*if (i!=0 && distanceToClass[i].first == correctClassName)
			correctlyRecognized = false;*/
	}
	if (!correctlyRecognized) {
		cout << "incorrect recognition of " << correctClassName << '\n';
		for (int i = 0; i < models_count; ++i) {
			cout << distanceToClass[i].first << ' ' << distanceToClass[i].second << endl;
		}
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
			if (!correctlyRecognized)
				cout << name << '\t' << videoClassDistances[i][name]<<'\t'<<log_prob << '\t' << log_weight<<'\t'<< (videoClassDistances[i][name] + log_prob*log_weight)<<endl;
			current_frame[name] = exp(-DISTANCE_WEIGHT*(videoClassDistances[i][name] + log_prob*log_weight));
			sum += current_frame[name];
		}
		for (string name : positions) {
			avg_frames[name] -= current_frame[name] / sum;
		}
	}

	sortDistances(distanceToClass, avg_frames);

	if (!correctlyRecognized) 
	{
		cout << "after regularization\n";
		for (int i = 0; i < models_count; ++i) {
			cout << distanceToClass[i].first << ' ' << distanceToClass[i].second << '\t' << closestFaces[distanceToClass[i].first] << endl;
		}
	}
	int res = 0;
	for (; res < distanceToClass.size() && distanceToClass[res].first != correctClassName; ++res)
		;
	return res;
}
//=================================
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

static int opencv_num_of_features = FEATURES_COUNT;
class SVMClassifier :public Classifier {
public:
	SVMClassifier();
	void train(MapOfFaces& totalImages, const vector<string>& commonNames);
	int get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName);

private:
	cv::Ptr<cv::ml::SVM> opencvClassifier;
	unordered_map<string, int> class2no;
};

SVMClassifier::SVMClassifier()
{
	set_name("SVM");
	// Set up SVM's parameters
	opencvClassifier = SVM::create();
	opencvClassifier->setType(SVM::C_SVC);
	opencvClassifier->setKernel(SVM::LINEAR);
	//opencvClassifier->setKernel(SVM::RBF);
	opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-8));
	opencvClassifier->setC(10);
}
void SVMClassifier::train(MapOfFaces& faceImages, const vector<string>& commonNames) {
	class2no.clear();
	int cur_class_no = 0;
	for (MapOfFaces::iterator iter = faceImages.begin(); iter != faceImages.end(); ++iter) {
		class2no[iter->first] = ++cur_class_no;
	}

	int db_size = 0;
	for (MapOfFaces::iterator iter = faceImages.begin(); iter != faceImages.end(); ++iter) {
		for (FaceImage* face : iter->second) {
			++db_size;
		}
	}
	Mat labelsMat(db_size, 1, CV_32S);
	Mat trainingDataMat(db_size, opencv_num_of_features, CV_32FC1);

	int ind = 0;
	for (MapOfFaces::iterator iter = faceImages.begin(); iter != faceImages.end(); ++iter) {
		for (FaceImage* face : iter->second) {
			for (int fi = 0; fi < opencv_num_of_features; ++fi) {
				trainingDataMat.at<float>(ind, fi) = face->getFeatures()[fi];
			}
			labelsMat.at<int>(ind, 0) = class2no[face->personName];
			++ind;
		}
	}
	opencvClassifier->train(trainingDataMat, ROW_SAMPLE, labelsMat);
}
int SVMClassifier::get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName)
{
	vector<unordered_map<string, double>> videoClassDistances(video.size());
	unordered_map<string, float> classDistances;
	for (int ind = 0; ind < video.size(); ++ind)
	{
		Mat queryMat(1, FEATURES_COUNT, CV_32FC1);
		for (int fi = 0; fi < FEATURES_COUNT; ++fi) {
			queryMat.at<float>(0, fi) = video[ind]->getFeatures()[fi];
		}

		float response = opencvClassifier->predict(queryMat);
		for (auto& class_no : class2no) {
			if (fabs(class_no.second - response) < 0.1)
				classDistances[class_no.first] += 0;
			else
				classDistances[class_no.first] += 1;
		}
	}
	/*string bestClass = "";
	float bestDist = 100000;
	for (auto& class_dist : classDistances) {
		if (class_dist.second < bestDist) {
			bestDist = class_dist.second;
			bestClass = class_dist.first;
		}
	}*/
	vector<pair<string, float>> distanceToClass(class2no.size());
	sortDistances(distanceToClass, classDistances);
	
	int res = 0;
	for (; res < distanceToClass.size() && distanceToClass[res].first != correctClassName; ++res)
		;
	return res;
}
//=================================
#define COMPUTE_PCA
int max_components = 256;
void train_pca(MapOfFaces& totalImages, PCA& pca) {
#ifdef COMPUTE_PCA
	int total_images_count = 0;
	for (auto& person : totalImages) {
		total_images_count += person.second.size();
	}
	Mat mat_features(total_images_count, FEATURES_COUNT, CV_32F);
	int ind = 0;
	for (auto& person : totalImages) {
		for (const FaceImage* face : person.second) {
			for (int j = 0; j < FEATURES_COUNT; ++j) {
				mat_features.at<float>(ind, j) =
					face->getFeatures()[j];
			}
			++ind;
		}
	}
	cout << "before pca train " << mat_features.rows << ' ' << mat_features.cols << ' ' << mat_features.type() << "\n";
	pca(mat_features, Mat(), CV_PCA_DATA_AS_ROW, max_components);
	cout << " end training pca\n";
#endif
}
class SequentialClassifier :public BruteForceClassifier {
public:
	SequentialClassifier(PCA& pca, FusionMethod fusion = FusionMethod::MEAN_DIST, double th=0.7, int feat_count = 32, int dist_weight = DISTANCE_WEIGHT);
	void train(MapOfFaces& totalImages, const vector<string>& commonNames);
	virtual int get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName);

	virtual int processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName);
private:
	double threshold;
	int reduced_features_count;

	cv::PCA& pca;
	std::vector<FaceImage*> new_database;
	std::vector<int> class_indices;
	int class_count;
	
	void recognize_frame(const float* test_features, vector<double>& distances, vector<int>& end_feature_indices, vector<int>& classes_to_check);
};
SequentialClassifier::SequentialClassifier(PCA& p, FusionMethod fusion, double th, int feat_count, int dist_weight) : BruteForceClassifier(fusion, dist_weight), pca(p), threshold(1.0 / th), reduced_features_count(feat_count)
{
	ostringstream os;
	os << "seq "<<int(fusion)<<" threshold=" << threshold << " feat_count=" << reduced_features_count<<" dw="<<distance_weight;
	set_name(os.str()); 
}
void SequentialClassifier::train(MapOfFaces& totalImages, const vector<string>& commonNames) {
	int total_images_count = 0;
	for (auto& person : totalImages) {
		total_images_count += person.second.size();
	}
#ifdef COMPUTE_PCA
	Mat mat_features(total_images_count, FEATURES_COUNT, CV_32F);
	int ind = 0;
	for (auto& person : totalImages) {
		for (const FaceImage* face : person.second) {
			for (int j = 0; j < FEATURES_COUNT; ++j) {
				mat_features.at<float>(ind, j) =
					face->getFeatures()[j];
			}
			++ind;
		}
	}
	/*cout << "before pca train "<<mat_features.rows<<' '<<mat_features.cols<<' '<<mat_features.type()<<"\n";
	pca(mat_features, Mat(), CV_PCA_DATA_AS_ROW, 0);
	*/
	Mat mat_projection_result = pca.project(mat_features);
	//cout << "after pca train " << mat_projection_result.rows << ' ' << mat_projection_result.cols << ' ' << mat_projection_result.type() << "\n";
#endif
	int images_no = 0;
	class_count = 0;
	new_database.resize(total_images_count);
	class_indices.resize(total_images_count);
	for (auto& person : totalImages) {
		for (FaceImage* face : person.second) {
#ifdef COMPUTE_PCA
			std::vector<float> features(FEATURES_COUNT);
			for (int j = 0; j < mat_projection_result.cols; ++j) {
				//db_features[i1*featuresCount + j] =
				features[j] = mat_projection_result.at<float>(images_no, j);
			}
#else
			std::vector<float>& features = face->getFeatureVector();
#endif
#ifdef USE_DNN_FEATURES
			new_database[images_no]=new FaceImage(face->fileName,face->personName,features,false);
#else
			new_database[images_no] = 0;
#endif
			class_indices[images_no] = class_count;
			
			++images_no;
		}
		++class_count;
	}
}

inline float get_distance(const float* lhs, const float* rhs, int start, int end) {
	float d = 0;
	for (int fi = start; fi < end; ++fi) {
#if DISTANCE==EUC
		d += (lhs[fi] - rhs[fi])*(lhs[fi] - rhs[fi]);
#endif
	}
	
	return d;
}
void SequentialClassifier::recognize_frame(const float* test_features, vector<double>& distances, vector<int>& end_feature_indices, vector<int>& classes_to_check) {
	distances.resize(new_database.size());
	end_feature_indices.resize(new_database.size());
	vector<double> class_min_distances(class_count);
	int last_feature =
		//FEATURES_COUNT;
		256;
	//max_components;
	
	int cur_features = 0;
	for (; cur_features<last_feature; cur_features += reduced_features_count) {
		double bestDist = 100000;
		fill(class_min_distances.begin(), class_min_distances.end(), numeric_limits<float>::max());
		for (int j = 0; j < new_database.size();++j) {
			if (!classes_to_check[class_indices[j]])
				continue;
			distances[j] += get_distance(test_features, new_database[j]->getFeatures(), cur_features, cur_features+reduced_features_count);
			end_feature_indices[j] += reduced_features_count;

			if (distances[j]<class_min_distances[class_indices[j]])
				class_min_distances[class_indices[j]] = distances[j];
			if (distances[j] < bestDist) {
				bestDist = distances[j];
			}
		}

		int num_of_variants = 0;
		double dist_threshold = bestDist*threshold;

		for (int c = 0; c<classes_to_check.size(); ++c) {
			if (classes_to_check[c]) {
				if (class_min_distances[c]>dist_threshold)
					classes_to_check[c] = 0;
				else
					++num_of_variants;
			}
		}
		if (num_of_variants == 1)
			break;
	}

}
int SequentialClassifier::get_correct_class_pos(const vector<FaceImage*>& video, string correctClassName)
{
	int frames_count = video.size();
	//pca transform
	vector<vector<float>> test_features(frames_count);
#ifdef COMPUTE_PCA
	Mat queryMat(frames_count, FEATURES_COUNT, CV_32FC1);
	for (int ind = 0; ind < frames_count; ++ind)
	{
		for (int fi = 0; fi < FEATURES_COUNT; ++fi) {
			queryMat.at<float>(ind, fi) = video[ind]->getFeatures()[fi];
		}
	}
	Mat pcaMat = pca.project(queryMat);
#endif
	for (int ind = 0; ind < frames_count; ++ind)
	{
		test_features[ind].resize(max_components);
		for (int fi = 0; fi < max_components; ++fi) {
#ifdef COMPUTE_PCA
			test_features[ind][fi] = pcaMat.at<float>(ind, fi);
#else
			test_features[ind][fi] = video[ind]->getFeatures()[fi];
#endif
		}
	}

	vector<unordered_map<string, double>> videoClassDistances(video.size());

	if (reduced_features_count > 0) {
		vector<vector<double>> distances(frames_count);
		vector<vector<int>> end_feature_indices(frames_count);
		vector<int> classes_to_check(class_count),total_classes_to_check(class_count);

		for (int ind = 0; ind < frames_count; ++ind)
		{
			fill(classes_to_check.begin(), classes_to_check.end(), 1);
			recognize_frame(&test_features[ind][0], distances[ind], end_feature_indices[ind], classes_to_check);
			for (int c = 0; c < class_count; ++c)
				total_classes_to_check[c] += classes_to_check[c];
		}
		for (int ind = 0; ind < frames_count; ++ind)
		{
			unordered_map<string, double>& frameDistances = videoClassDistances[ind];
			for (int j = 0; j < new_database.size(); ++j) {
				if (!total_classes_to_check[class_indices[j]])
					continue;

				distances[ind][j] += get_distance(&test_features[ind][0], new_database[j]->getFeatures(),
					end_feature_indices[ind][j], max_components);
				distances[ind][j] /= max_components;

				string class_name = new_database[j]->personName;
				bool classNameExists = frameDistances.find(class_name) != frameDistances.end();
				if (!classNameExists || (classNameExists && frameDistances[class_name] > distances[ind][j])) {
					frameDistances[class_name] = distances[ind][j];
				}
			}
		}
	}
	else{
		for (int ind = 0; ind < frames_count; ++ind)
		{
			unordered_map<string, double>& frameDistances = videoClassDistances[ind];
			for (int j = 0; j < new_database.size(); ++j) {

				float dist = get_distance(&test_features[ind][0], new_database[j]->getFeatures(),
					0, -reduced_features_count)/(-reduced_features_count);

				string class_name = new_database[j]->personName;
				bool classNameExists = frameDistances.find(class_name) != frameDistances.end();
				if (!classNameExists || (classNameExists && frameDistances[class_name] > dist)) {
					frameDistances[class_name] = dist;
				}
			}
		}
	}
	return processVideo(videoClassDistances,correctClassName);
}

int SequentialClassifier::processVideo(vector<unordered_map<string, double>>& videoClassDistances, string correctClassName) {
	return BruteForceClassifier::processVideo(videoClassDistances,correctClassName);
}
//=================================




//#define TRAIN_RATE 0.8
#ifdef TRAIN_RATE
#define TEST_COUNT 3
#else
#define TEST_COUNT 1
#endif

void test_recognition_method(Classifier* classifier, MapOfFaces& totalImages, MapOfVideos& videos, vector<string>& commonNames) {
#ifdef TRAIN_RATE
	srand(17);
	MapOfFaces faceImages, testImages;
#else
	MapOfFaces& faceImages = totalImages;
#endif
	float total_accuracy = 0, total_time=0;
	for (int t = 0; t < TEST_COUNT; ++t) {
#ifdef TRAIN_RATE
		getTrainingAndTestImages(totalImages, faceImages, testImages, true, TRAIN_RATE);
#endif
		classifier->train(faceImages, commonNames);

		int errorsCount = 0, totalVideos = 0, totalFrames = 0;
		unordered_map<string, int> class_errors;
		cout << "start" << flush;
		auto t1 = chrono::high_resolution_clock::now();
		for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter) {
			for (vector<FaceImage*>& video : iter->second) {
				int pos = classifier->get_correct_class_pos(video, iter->first);
				if (pos != 0) {
					cout << "\rorig=" << std::setw(35) << iter->first << " pos=" << std::setw(4) << pos << flush;
					//cout << "orig="  << iter->first << " pos=" << pos << " file="<< video[0]->fileName <<'\n';
					++errorsCount;
					++class_errors[iter->first];
				}
				++totalVideos;
				totalFrames += video.size();
			}
			/*if (totalVideos > 10)
				break;*/
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

		cout << endl << "method " << classifier->get_name() << " video error rate=" << (100.0*errorsCount / totalVideos) << " (" << errorsCount << " out of "
			<< totalVideos << ") recall=" << (recall / videos.size()) << " avg time=" << (total / totalFrames) << "(ms) prev time=" << (total / totalVideos) << "\n";
		total_accuracy += 100.0*(totalVideos - errorsCount) / totalVideos;
		total_time += total / totalFrames;
	}
	cout << "method " << classifier->get_name()<<" average accuracy=" << total_accuracy/TEST_COUNT <<
		" total time (ms)="<< total_time/TEST_COUNT << "\n\n";
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
	


#if 0
	compute_model_distances(totalImages, commonNames);
	/*vector<FaceImage*> faceImages;
	for (MapOfFaces::iterator iter = totalImages.begin(); iter != totalImages.end(); ++iter) {
		for (FaceImage* face : iter->second)
			faceImages.push_back(face);
	}*/
	string className = "942";
	vector<FaceImage*> video (1);
	vector<FaceImage*>& orig_video = videos[className][12];
	video[0]=orig_video[0];
	//for (int k = 0; k < orig_video.size(); ++k)
	{
		//video[0] = orig_video[k];
		/*cout << k << endl;
		video.clear();
		video.push_back(orig_video[k]);*/
		
		//video.push_back(orig_video[0]); video.push_back(orig_video[10]); video.push_back(orig_video.back());
		//video.erase(video.begin() + k, video.end());
		//video.erase(video.begin(),video.begin() + video.size() -1);
		//video = orig_video;
		int pos = recognize(NN_Method::SIMPLE_NN, totalImages, video, className);
		cout << "nn pos=" << pos << endl;
		/*pos = recognize(NN_Method::MAP, totalImages, video, className);
		cout << "nn pos=" << pos << endl;*/
		analyzed_models_count = 16;
		pos = recognize(NN_Method::ML_DIST, totalImages, video, className);
		cout << "proposed pos=" << pos << endl;
	}
	return;
#endif

	vector<Classifier*> classifiers;

	classifiers.push_back(new BruteForceClassifier(FusionMethod::MEAN_DIST));
#ifndef AGGREGATE_VIDEOS
	classifiers.push_back(new BruteForceClassifier(FusionMethod::MAP)); 
	//classifiers.push_back(new BruteForceClassifier(FusionMethod::MAX_POSTERIOR));
#endif
#if DISTANCE==EUC
	classifiers.push_back(new SVMClassifier());
#endif
#if 1
	//test_complex_classifier(totalImages, videos, class2no);
	vector<int> regs({ 8, 4 });// 8, 80, 100, 120, 64, 40, 28, 4, 16});
	for (int reg : regs) {
	//for (double reg = 3.5; reg <= 5.5; reg += 0.5) {
		//for (int M = 2; M <= 16; M *= 2) 
		int M = 16;
		{
			classifiers.push_back(new MLDistClassifier(reg, M));
		}
	}
#endif
#if 0
	PCA pca;
	train_pca(totalImages, pca);
	//for (double th = 0.5; th <= 0.9; th += 0.1)
	double th = 0.7;
	{
		//brute force of first 32/256 principal components
		//classifiers.push_back(new SequentialClassifier(pca, FusionMethod::MEAN_DIST, th, -32));
		//classifiers.push_back(new SequentialClassifier(pca, FusionMethod::MEAN_DIST, th, -256));

		classifiers.push_back(new SequentialClassifier(pca, FusionMethod::MEAN_DIST, th, 32));

#ifndef AGGREGATE_VIDEOS
		//for (int dist_weight = 300; dist_weight <= 300000; dist_weight *= 10) 
		int dist_weight = 3000;// 30000;
		{
			classifiers.push_back(new SequentialClassifier(pca, FusionMethod::MAP, th, 32, dist_weight));
			classifiers.push_back(new SequentialClassifier(pca, FusionMethod::MAX_POSTERIOR, th, 32, dist_weight));
		}
#endif
	}
#endif
	for (Classifier* classifier : classifiers) {
		test_recognition_method(classifier, totalImages, videos, commonNames);
	}
	for (Classifier* classifier : classifiers) {
		delete classifier;
	}
}



void testRecognitionOfMultipleImages() {
	//srand(13);
	MapOfFaces totalImages, faceImages, testImages;
	loadFaces(totalImages);
#if 1
	ifstream ifs(BASE_DIR + "lfw_ytf_classes.txt");
	vector<string> valid_classes;
	while (ifs) {
		string class_name;
		if (!getline(ifs, class_name))
			break;
		valid_classes.push_back(class_name);
	}
	ifs.close();
	vector<string> dbNames;
	for (auto keyValue : totalImages) {
		dbNames.push_back(keyValue.first);
	}
	std::sort(valid_classes.begin(), valid_classes.end());
	std::sort(dbNames.begin(), dbNames.end());

	vector<string> listToRemove;
	std::set_difference(dbNames.begin(), dbNames.end(), valid_classes.begin(), valid_classes.end(), back_inserter(listToRemove));
	for (string needToRemove : listToRemove) {
		totalImages.erase(needToRemove);
	}
	cout << "after removal of non YTF faces:" << totalImages.size() << endl;
#elif 0
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
	}
	ofstream ofs(BASE_DIR + "lfw_ytf_classes.txt");
	for (auto& person : totalImages)
		ofs << person.first << endl;
	ofs.close();
#endif
	
	MLDistClassifier::image_map img_map;
	int ind = 0;
	for (auto& person : totalImages) {
		for (auto& face : person.second) {
			img_map.insert(make_pair(face, ind));
			++ind;
		}
	}
	MLDistClassifier::distance_map dist_map;
	int num_faces = img_map.size();
	dist_map.resize(num_faces);
	for(int i=0;i<num_faces;++i)
		dist_map[i].resize(num_faces);
	int ind1 = 0;
	for (auto& person1 : totalImages) {
		for (auto& face1 : person1.second) {
			int ind2 = 0;
			for (auto& person2 : totalImages) {
				for (auto& face2 : person2.second) {
					dist_map[ind1][ind2] = face1->distance(face2);
					++ind2;
				}
			}
			++ind1;
		}
	}
	cout << " distance map computed\n";

	vector<string> class_names;
	class_names.reserve(totalImages.size());
	for (auto& image : totalImages)
	class_names.push_back(image.first);

	vector<BruteForceClassifier*> classifiers;
	classifiers.push_back(new BruteForceClassifier(FusionMethod::MEAN_DIST));
	for (int reg = 30; reg <= 45; reg += 5)
	//int reg = 60;
		classifiers.push_back(new MLDistClassifier(reg, 16, &dist_map, &img_map));
	int num_of_classifiers = classifiers.size();
	
	const int FRAMES_COUNT = 1;
	const int TESTS = 10;
	vector<double> totalTestsErrorRates(num_of_classifiers), errorRateVars(num_of_classifiers);


	for (int testCount = 0; testCount < TESTS; ++testCount) {
		vector<int> errorsCount (num_of_classifiers);
		getTrainingAndTestImages(totalImages, faceImages, testImages);
		for (int clsId=0; clsId<num_of_classifiers;++clsId)
			classifiers[clsId]->train(faceImages, class_names);

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
							tmpDist = dist_map[img_map[testPersonImages.second[ind]]][img_map[dbPersonImages.second[j]]];
							if (tmpDist < bestDist) {
								bestDist = tmpDist;
							}
						}
						classDistances.insert(make_pair(dbPersonImages.first, bestDist));
					}
				}
				for (int clsId = 0; clsId < num_of_classifiers; ++clsId) {
					int res = classifiers[clsId]->processVideo(videoClassDistances, testPersonImages.first);
					if (res != 0)
						++errorsCount[clsId];
				}
			}
		}


		for (int clsId = 0; clsId < num_of_classifiers; ++clsId) {
			double errorRate = 100.*errorsCount[clsId] / test_count;
			std::cout << classifiers[clsId]->get_name()<< ": test=" << testCount << " test_count=" << test_count << " error=" << errorRate << std::endl;
			totalTestsErrorRates[clsId] += errorRate;
			errorRateVars[clsId] += errorRate * errorRate;
		}

	}
	for (int clsId = 0; clsId < num_of_classifiers; ++clsId) {
		totalTestsErrorRates[clsId] /= TESTS;
		errorRateVars[clsId] = sqrt((errorRateVars[clsId] - totalTestsErrorRates[clsId] * totalTestsErrorRates[clsId] * TESTS) / (TESTS));
		std::cout << classifiers[clsId]->get_name() << ": Avg error=" << totalTestsErrorRates[clsId] << " Sigma=" << errorRateVars[clsId] << endl;
	}
	for (int clsId = 0; clsId<num_of_classifiers; ++clsId)
		delete classifiers[clsId];
}