#ifndef DB_H
#define DB_H

//#define USE_LCNN
#ifndef USE_LCNN
#define USE_VGG
#endif

#define USE_LFW
//#define USE_CASIA
//#define USE_CALTECH

static const char* PCA_CASIA_FEATURES_FILE_NAME ="casia_dnn_features_my_best_pca.txt";

static const char* PCA_FEATURES_FILE_NAME =
#ifdef USE_LCNN
#if defined (USE_LFW) && defined(USE_CASIA)
        PCA_CASIA_FEATURES_FILE_NAME;
#elif defined (USE_LFW)
        "lfw_dnn_features_my_best_pca.txt";
#else
        "pf_dnn_features_my_best_pca.txt";
#endif

#else
#ifdef USE_CALTECH
#ifdef USE_VGG
        //"dnn_101_vgg19_last_mean.txt";
        "dnn_101_vgg19_img_resized_fc6.txt";
        //"birds_vgg19_fc6.txt";
        //"dogs_dnn_vgg19_features_mean_fc7.txt";
#else
        "dnn_101_googlenet.txt";
        //"dnn_256_googlenet.txt";
        //"birds_googlenet.txt";
        //"dogs_dnn_googlenet.txt";
#endif
#else
        "lfw_dnn_vgg_features_noscale_pca.txt";
        //"dnn_vgg_features_all_mean.txt";
#endif
#endif

static const char* FEATURES_FILE_NAME =
#ifdef USE_PCA
#ifdef USE_LCNN
//"lfw\\dnn_features_my_best.txt";
//"pubfig83\\dnn_features_best_256.txt";
//"feret\\dnn_features.txt";
//"casia\\dnn_features_my.txt";
#else
//"lfw\\dnn_vgg_features_mean.txt";
//"casia\\dnn_vgg_features_mean.txt";
"lfw\\dnn_101_googlenet.txt";
#endif
#else
PCA_FEATURES_FILE_NAME;
#endif


const double FRACTION =
#ifdef USE_CALTECH
        0.03;//0.1;
#elif defined(USE_LFW)
        0.5;
#else
        0.5;//0.05;
#endif
const int FEATURES_COUNT =
#ifdef USE_LCNN
256;
#else
#ifdef USE_VGG
        4096;
#else
        1024;
#endif
#endif

#endif // DB_H
