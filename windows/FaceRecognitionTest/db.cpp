#include "db.h"

#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include "windows.h"

static void loadImages(std::string db_path, std::string test_path, MapOfFaces & faceImagesMap);
void removeSingleImages(MapOfFaces& dbImages) {
	vector<string> listToRemove;
	for (MapOfFaces::iterator iter = dbImages.begin(); iter != dbImages.end(); ++iter) {
		int currentFaceCount = iter->second.size();
		if (currentFaceCount <= 1)
			listToRemove.push_back(iter->first);
	}
	listToRemove.clear();
	for (string needToRemove : listToRemove)
		dbImages.erase(needToRemove);
	cout << "total size=" << dbImages.size() << " removed=" << listToRemove.size() << endl;
}

void getTrainingAndTestImages(const MapOfFaces& totalImages, std::vector<FaceImage*>& faceImages, std::vector<FaceImage*>& testImages, bool randomize, float fraction)
{
	const int INDICES_COUNT = 400;
	int indices[INDICES_COUNT];
	for (int i = 0; i<INDICES_COUNT; ++i)
		indices[i] = i;
	if (randomize)
		std::random_shuffle(indices, indices + INDICES_COUNT);

	faceImages.clear();
	testImages.clear();

	for (MapOfFaces::const_iterator iter = totalImages.begin(); iter != totalImages.end(); ++iter) {
		int currentFaceCount = iter->second.size();
		//cout << currentFaceCount << endl;
		float size_f = currentFaceCount*fraction;
		int db_size = (int)size_f;
		if (rand() & 1)
			db_size = (int)ceil(size_f);
		if (db_size == currentFaceCount)
			db_size = currentFaceCount - 1;
		if (db_size == 0)
			db_size = 1;

		//std::vector<FaceImage*> dbFaces(iter->second.begin(),iter->second.begin()+db_size);
		//std::vector<FaceImage*> testFaces(iter->second.begin()+db_size,iter->second.end());
		int ind = 0;
		for (int i = 0; i < INDICES_COUNT; ++i){
			if (indices[i] < currentFaceCount) {
				if (ind < db_size) {
					faceImages.push_back(iter->second[indices[i]]);
				}
				else
					testImages.push_back(iter->second[indices[i]]);
				++ind;
			}
		}
	}
}
void getTrainingAndTestImages(const MapOfFaces& totalImages, MapOfFaces& faceImages, MapOfFaces& testImages, bool randomize, float fraction){
	const int INDICES_COUNT = 400;
	int indices[INDICES_COUNT];
	for (int i = 0; i<INDICES_COUNT; ++i)
		indices[i] = i;
	if (randomize)
		std::random_shuffle(indices, indices + INDICES_COUNT);

	faceImages.clear();
	testImages.clear();

	for (MapOfFaces::const_iterator iter = totalImages.begin(); iter != totalImages.end(); ++iter) {
		string class_name = iter->first;
		int currentFaceCount = iter->second.size();
		//cout << currentFaceCount << endl;
		float size_f = currentFaceCount*fraction;
		int db_size = (int)size_f;
		if (rand() & 1)
			db_size = (int)ceil(size_f);
		if (db_size == currentFaceCount)
			db_size = currentFaceCount - 1;
		if (db_size == 0)
			db_size = 1;
		faceImages.insert(std::make_pair(class_name, std::vector<FaceImage*>()));
		if (db_size < currentFaceCount)
			testImages.insert(std::make_pair(class_name, std::vector<FaceImage*>()));
		//std::vector<FaceImage*> dbFaces(iter->second.begin(),iter->second.begin()+db_size);
		//std::vector<FaceImage*> testFaces(iter->second.begin()+db_size,iter->second.end());
		int ind = 0;
		for (int i = 0; i < INDICES_COUNT; ++i) {
			if (indices[i] < currentFaceCount) {
				if (ind < db_size) {
					faceImages[class_name].push_back(iter->second[indices[i]]);
				}
				else
					testImages[class_name].push_back(iter->second[indices[i]]);
				++ind;
			}
		}
	}
}
void loadFaces(MapOfFaces& dbImages){
	bool read_image_files = true;
#if defined(USE_DNN_FEATURES)
#if 0
	ifstream ifs("dnn_vgg_features.bin");
	if (ifs){
		while (ifs){
			FaceImage* face = FaceImage::readFaceImage(ifs);
			if (dbImages.find(face->person_name) == dbImages.end()){
				dbImages.insert(std::make_pair(face->person_name, std::vector<FaceImage*>()));
			}
			std::vector<FaceImage*>& currentDirFaces = dbImages[face->person_name];
			currentDirFaces.push_back(face);
		}
		ifs.close();
		read_image_files = false;
		cout << "total size=" << dbImages.size() << endl;
	}
#else
	ifstream ifs(FEATURES_FILE_NAME);
	//ofstream ofs("dnn_vgg_features.bin");
	if (ifs){
		int total_count = 0;
#ifdef USE_MEDIA_ID
		vector<float> avg_features(FEATURES_COUNT);
		string prev_media_id, prev_person_name,prev_file_name;
		int num_of_frames_in_media = 0;
#endif
		std::string file_name, person_name, feat_str;
		while (ifs) {
			if (!getline(ifs, file_name))
				break;
			if (!getline(ifs, person_name))
				break;
			if (!getline(ifs, feat_str))
				break;
			//cout << file_name << ' ' << person_name << '\n';
			istringstream iss(feat_str);
			vector<float> features(FEATURES_COUNT);
			for (int i = 0; i < FEATURES_COUNT; ++i)
				iss >> features[i];

			if (dbImages.find(person_name) == dbImages.end()) {
				dbImages.insert(std::make_pair(person_name, std::vector<FaceImage*>()));
			}

#ifdef USE_MEDIA_ID
			size_t slash_ind = file_name.rfind("\\");
			size_t dot_ind = file_name.rfind(".");
			size_t underscore_ind = file_name.rfind("_");
			string media_id = "";
			if (slash_ind != string::npos) {
				size_t end_ind = dot_ind;
				if (underscore_ind != string::npos && underscore_ind > slash_ind)
					end_ind = underscore_ind;
				media_id = file_name.substr(slash_ind + 1, end_ind - slash_ind - 1);
			}
			//cout << prev_file_name << ' ' << media_id << ' '<< prev_media_id<<' '<< num_of_frames_in_media<<'\n';
			if (prev_person_name != person_name || prev_media_id != media_id) {
				if (num_of_frames_in_media > 0) {
					for (int i = 0; i < FEATURES_COUNT; ++i) {
						avg_features[i] /= num_of_frames_in_media;
					}
					std::vector<FaceImage*>& currentDirFaces = dbImages[prev_person_name];
					currentDirFaces.push_back(new FaceImage(prev_file_name, prev_person_name, avg_features));
					for (int i = 0; i < FEATURES_COUNT; ++i) {
						avg_features[i] = 0;
					}
					num_of_frames_in_media = 0;
					++total_count;
				}
				prev_media_id=media_id;
				prev_person_name = person_name;
				prev_file_name = file_name;
			}
			
			++num_of_frames_in_media;
			for (int i = 0; i < FEATURES_COUNT; ++i) {
				avg_features[i] += features[i];
			}
#else
			std::vector<FaceImage*>& currentDirFaces = dbImages[person_name];
			currentDirFaces.push_back(new FaceImage(file_name, person_name, features));
			//currentDirFaces.back()->writeFaceImage(ofs);
			++total_count;
#endif
			/*if (total_count >= 3000)
				break;*/
		}
		ifs.close();
#ifdef USE_MEDIA_ID
		//cout << " end "<<prev_file_name << ' ' << media_id << ' ' << prev_media_id << ' ' << num_of_frames_in_media << '\n';
		if (num_of_frames_in_media > 0) {
			for (int i = 0; i < FEATURES_COUNT; ++i) {
				avg_features[i] /= num_of_frames_in_media;
			}
			std::vector<FaceImage*>& currentDirFaces = dbImages[prev_person_name];
			currentDirFaces.push_back(new FaceImage(prev_file_name, prev_person_name, avg_features));
			++total_count;
		}
#endif
		read_image_files = false;
		cout << "total size=" << dbImages.size() << " totalImages=" << total_count<<endl;
		removeSingleImages(dbImages);
	}
	//ofs.close();
#endif
#endif
	if (read_image_files){
		loadImages(DB, TEST, dbImages);
		removeSingleImages(dbImages);
#if defined(USE_DNN_FEATURES)
		ofstream of(FEATURES_FILE_NAME);
		if (of){
			cout << "begin write file\n";
			for (MapOfFaces::iterator iter = dbImages.begin(); iter != dbImages.end(); ++iter){
				for (FaceImage* face : iter->second){
					of << face->fileName << endl;
					of << face->personName << endl;
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
			cout << "end write file\n";
			of.close();
		}
#endif
	}
}


static void loadImages(std::string db_path, std::string test_path, MapOfFaces & faceImagesMap){
	WIN32_FIND_DATA ffd, ffd1;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	HANDLE hFindFile = INVALID_HANDLE_VALUE;
	int num_of_persons_processed = 0;

	std::cout << "start load " << db_path<<std::endl;

	hFind = FindFirstFile((db_path + "\\*").c_str(), &ffd);
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
			std::string fullDirName = db_path + "\\" + dirName + "\\";
			hFindFile = FindFirstFile((fullDirName + "*").c_str(), &ffd1);
			if (INVALID_HANDLE_VALUE == hFindFile) {
				std::cout << "no files." << std::endl;
			}
			std::vector<FaceImage*> currentDirFaces;
			do
			{
				string filePath = fullDirName + ffd1.cFileName;
				if ((ffd1.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 &&
					(ffd1.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN) == 0)
				{
					currentDirFaces.push_back(new FaceImage(filePath.c_str(), dirName));
					//currentDirFaces.push_back(new FaceImage(filePath.c_str(), dirName,POINTS_IN_W,POINTS_IN_H,0,ImageTransform::FLIP));
					//currentDirFaces.push_back(new FaceImage(filePath.c_str(), dirName, POINTS_IN_W, POINTS_IN_H, 0, ImageTransform::NORMALIZE));
				}
			} while (FindNextFile(hFindFile, &ffd1) != 0);

			faceImagesMap.insert(std::make_pair(dirName, currentDirFaces));
#ifdef USE_DNN_FEATURES
			if (++num_of_persons_processed % 10 == 0)
				std::cout << "first stage " << num_of_persons_processed << '\n';
#endif
		}
	} while (FindNextFile(hFind, &ffd) != 0);

	if (test_path != ""){
		num_of_persons_processed = 0;
		hFind = FindFirstFile((test_path + "\\*").c_str(), &ffd);
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
				std::string fullDirName = test_path + "\\" + dirName + "\\";
				hFindFile = FindFirstFile((fullDirName + "*").c_str(), &ffd1);
				if (INVALID_HANDLE_VALUE == hFindFile) {
					std::cout << "no files." << std::endl;
				}
				std::vector<FaceImage*>& currentDirFaces = faceImagesMap[dirName];
				do
				{
					if ((ffd1.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
					{
						currentDirFaces.push_back(new FaceImage((fullDirName + ffd1.cFileName).c_str(), dirName));
					}
				} while (FindNextFile(hFindFile, &ffd1) != 0);
#ifdef USE_DNN_FEATURES
				if (++num_of_persons_processed % 10 == 0)
					std::cout << "second stage " << num_of_persons_processed << '\n';
#endif
			}
		} while (FindNextFile(hFind, &ffd) != 0);
	}
	std::cout << "end load" << std::endl;

}