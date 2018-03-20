#ifndef __DB_H__
#define __DB_H__

#include "FaceImage.h"

#include <string>
#include <map>
#include <vector>

#define USE_ESSEX 1
#define USE_FERET 2
#define USE_YALE 3
#define USE_ATT 4
#define USE_ATT_YALE 5
#define USE_STUDENTS 6
#define USE_JAFFE 7
#define USE_LFW 8
#define USE_PUBFIG83 9
#define USE_IJBA 10
#define USE_TEST_DB 11

//#define DB_USED USE_ESSEX
//#define DB_USED USE_FERET
//#define DB_USED USE_YALE
//#define DB_USED USE_ATT
//#define DB_USED USE_ATT_YALE
//#define DB_USED USE_STUDENTS
//#define DB_USED USE_JAFFE
//#define DB_USED USE_LFW
//#define DB_USED USE_PUBFIG83
#define DB_USED USE_IJBA
//#define DB_USED USE_TEST_DB

const std::string IMAGE_DIR = "D:\\datasets\\";

#if DB_USED == USE_ESSEX
const std::string DB= IMAGE_DIR+"Essex\\reduced_db";
//const std::string DB= IMAGE_DIR+"Essex\\db";
const std::string TEST= IMAGE_DIR+"Essex\\test_faces";
//const std::string TEST= "";
//const std::string DB = IMAGE_DIR+"Essex\\db";
//const std::string DB = IMAGE_DIR+"Essex\\test_faces";
const float FRACTION = 0.4;
const std::string FEATURES_FILE_NAME = IMAGE_DIR+"Essex\\dnn_features.txt";
#elif DB_USED == USE_FERET
//const std::string DB = IMAGE_DIR+"feret\\db";
//const std::string TEST = IMAGE_DIR+"feret\\test";
const std::string DB = IMAGE_DIR+"feret\\fa";
const std::string TEST = IMAGE_DIR+"feret\\fb";
//const std::string DB = IMAGE_DIR+"feret\\probes\\fa";//probes
//const std::string TEST = IMAGE_DIR+"feret\\probes\\dup2";
//const std::string DB = IMAGE_DIR+"feret\\full_db";
//const std::string TEST = "";
const float FRACTION = 0.5;
const std::string FEATURES_FILE_NAME = 
#ifdef USE_RGB_DNN
										IMAGE_DIR+"feret\\dnn_vgg_features.txt";
#else
										IMAGE_DIR+"feret\\dnn_features.txt";
#endif
#elif DB_USED == USE_YALE
//const std::string DB = IMAGE_DIR+"yales\\hard\\db";
//const std::string TEST = IMAGE_DIR+"yales\\faces";
const std::string DB = IMAGE_DIR+"yales\\faces";
const std::string TEST = "";
const std::string FEATURES_FILE_NAME = IMAGE_DIR+"yales\\dnn_features.txt";
const float FRACTION = 0.1;
#elif DB_USED == USE_ATT
//const std::string DB = IMAGE_DIR+"att_faces\\db";
//const std::string TEST = IMAGE_DIR+"att_faces\\test";
const std::string DB = IMAGE_DIR+"att_faces\\normal_db";
const std::string TEST = "";
const float FRACTION = 0.1;
const std::string FEATURES_FILE_NAME =
#ifdef USE_RGB_DNN
IMAGE_DIR+"att_faces\\dnn_vgg__features.txt";
#else
IMAGE_DIR+"att_faces\\dnn_features.txt";
#endif
#elif DB_USED == USE_ATT_YALE
const std::string DB = IMAGE_DIR+"att_yale";
const std::string TEST = "";
const float FRACTION = 0.2f;
#elif DB_USED == USE_STUDENTS
const std::string DB = IMAGE_DIR+"students\\db";
const std::string TEST = IMAGE_DIR+"students\\test";
const float FRACTION = 0.05;
const std::string FEATURES_FILE_NAME = IMAGE_DIR+"students\\dnn_features.txt";
#elif DB_USED == USE_JAFFE
const std::string DB = IMAGE_DIR+"jaffe\\faces";
const std::string TEST = "";
const float FRACTION = 0.05;
const std::string FEATURES_FILE_NAME = IMAGE_DIR+"jaffe\\dnn_features.txt";
#elif DB_USED == USE_LFW
const std::string BASE_DIR = IMAGE_DIR+"lfw_ytf\\";
//const std::string DB = IMAGE_DIR+"lfw-deepfunneled\\db";
//const std::string TEST = IMAGE_DIR+"lfw-deepfunneled\\test";
const std::string DB = BASE_DIR + 
//"lfw_faces";
"lfw_cropped";
const std::string TEST = "";
const float FRACTION = 0.0;
const std::string FEATURES_FILE_NAME = BASE_DIR +"lfw_"+"faces_"+
#ifdef USE_RGB_DNN
//"vgg_features.txt";
"vgg2_features.txt";
//"res101_features.txt";
//"ydwen_features.txt"; 
#else
"dnn_features.txt";
#endif
#elif DB_USED == USE_PUBFIG83
//const std::string TEST = IMAGE_DIR+"pubfig83\\db";
//const std::string DB = IMAGE_DIR+"pubfig83\\test";
const std::string DB = IMAGE_DIR+"pubfig83\\saved_color";
const std::string TEST = "";
const float FRACTION = 0.4;
const std::string FEATURES_FILE_NAME =
#ifdef USE_RGB_DNN
IMAGE_DIR+"pubfig83\\dnn_vgg_features.txt";
#else
IMAGE_DIR+"pubfig83\\dnn_features.txt";
#endif
#elif DB_USED == USE_IJBA
//#define USE_MEDIA_ID
const std::string BASE_DIR = IMAGE_DIR +
"ijba\\1N_sets\\split2\\new\\";
//"ijba\\1N_images\\";

const std::string DB = BASE_DIR +"gallery_equal";
const std::string TEST = "";
const float FRACTION = 0.5;
const std::string FEATURES_FILE_NAME = BASE_DIR +"gallery_"+"equal_"+
#ifdef USE_RGB_DNN
//"dnn_vgg_features.txt";
"dnn_vgg2_features.txt";
//"dnn_vgg2_features_flipped.txt";
//"dnn_res101_features.txt";
//"dnn_ydwen_features.txt"; 
#else
"dnn_lcnn_features.txt";
#endif
#elif DB_USED == USE_TEST_DB
const std::string DB = "";
const std::string TEST = "";
const float FRACTION = 0.5;
#endif


typedef std::map<std::string, std::vector<FaceImage*> > MapOfFaces;
void loadFaces(MapOfFaces& totalImages);
void getTrainingAndTestImages(const MapOfFaces& totalImages, std::vector<FaceImage*>& faceImages, std::vector<FaceImage*>& testImages,bool randomize=true, float fraction=FRACTION);
void getTrainingAndTestImages(const MapOfFaces& totalImages, MapOfFaces& faceImages, MapOfFaces& testImages, bool randomize = true, float fraction = FRACTION);

#endif //__DB_H__