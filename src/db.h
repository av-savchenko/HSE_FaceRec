#ifndef __DB_H__
#define __DB_H__

#include <string>

#define USE_ESSEX 1
#define USE_FERET 2
#define USE_YALE 3
#define USE_ATT 4
#define USE_STUDENTS 5
#define USE_JAFFE 6
#define USE_TEST_DB 7

#define DB_USED USE_ESSEX
//#define DB_USED USE_FERET
//#define DB_USED USE_YALE
//#define DB_USED USE_ATT
//#define DB_USED USE_STUDENTS
//#define DB_USED USE_JAFFE
//#define DB_USED USE_TEST_DB

#if DB_USED == USE_ESSEX
//const std::string DB= "C:\\Users\\Andrey\\Documents\\images\\Essex\\reduced_db";
const std::string DB= "C:\\Users\\Andrey\\Documents\\images\\Essex\\db";
const std::string TEST= "C:\\Users\\Andrey\\Documents\\images\\Essex\\test_faces";
//const std::string TEST= "";
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\Essex\\db";
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\Essex\\test_faces";
const float FRACTION = 0.4;
#elif DB_USED == USE_FERET
const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\feret\\db";
const std::string TEST = "C:\\Users\\Andrey\\Documents\\images\\feret\\test";
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\feret\\test";
//const std::string TEST = "C:\\Users\\Andrey\\Documents\\images\\feret\\db";
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\feret\\full_db";
//const std::string TEST = "";
const float FRACTION = 0.2;
#elif DB_USED == USE_YALE
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\yales\\hard\\db";
//const std::string TEST = "C:\\Users\\Andrey\\Documents\\images\\yales\\faces";
const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\yales\\faces";
const std::string TEST = "";
const float FRACTION = 0.1;
#elif DB_USED == USE_ATT
//const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\att_faces\\db";
//const std::string TEST = "C:\\Users\\Andrey\\Documents\\images\\att_faces\\normal_db";
const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\att_faces\\normal_db";
const std::string TEST = "";
const float FRACTION = 0.2;
#elif DB_USED == USE_STUDENTS
const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\students\\db";
const std::string TEST = "C:\\Users\\Andrey\\Documents\\images\\students\\test";
const float FRACTION = 0.4;
#elif DB_USED == USE_JAFFE
const std::string DB = "C:\\Users\\Andrey\\Documents\\images\\jaffe\\faces";
const std::string TEST = "";
const float FRACTION = 0.25;
#elif DB_USED == USE_TEST_DB
const std::string DB = "";
const std::string TEST = "";
const float FRACTION = 0.5;
#endif


#endif //__DB_H__
