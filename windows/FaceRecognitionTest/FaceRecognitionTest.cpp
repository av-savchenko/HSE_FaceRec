// FaceRecognitionTest.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

#include "RecognitionTesting.h"
#include "DEMTesting.h"

#include <windows.h>

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <locale>
#include <cmath>

using namespace std;

class my_numpunct: public std::numpunct<char> { 
    std::string do_grouping() const { return "\3"; } 
};  

void testVideoRecognition();
void testRecognitionOfMultipleImages();


int _tmain(int argc, _TCHAR* argv[])
{
	std::locale nl(std::locale(), new my_numpunct);  
    std::cout.imbue(nl); 
    std::cout.setf(std::ios::fixed); 
	std::cout.precision(5);

	//srand(time(0));
	srand(54321);
#if 0
	testClassification();
#elif 0
	testSIFT();
#elif 0
	testRecognitionOfMultipleImages();
#elif 1
	testVideoRecognition();
#elif 1
	testRecognition();
#elif 0
	doClustering();
#elif 0
	extern int main_ch();
	main_detect();
	//main_ch();
#else
	testDEM();
#endif

    return 0;
}
