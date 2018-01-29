#ifndef __TEST_DB__
#define __TEST_DB__

#include <vector>
class FaceImage;

void fill_image();
void load_model_and_test_images(std::vector<FaceImage*>& dbImages, std::vector<FaceImage*>& testImages);

#endif //__TEST_DB__