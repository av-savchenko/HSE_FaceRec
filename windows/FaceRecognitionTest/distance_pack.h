#ifndef __DISTANCE_PACK__
#define __DISTANCE_PACK__

#include <cmath>

//#define USE_PACK

#ifndef USE_PACK
inline float pack(float x, float y){
	return y;
}
inline void unpack(float packed, float *x, float *y){
	*x = *y = packed;
}
inline float unpack_dist(float packed){
	return packed;
}
inline float unpack_var(float packed){
	return packed;
}
#else
const int precision = 4096 * 256;
inline float pack(float x, float y){
	x = std::floor(x*(precision - 1));
	y = std::floor(y*(precision - 1));
	return x*precision + y;
}
inline void unpack(float packed, float *x, float *y){
	*x = std::floor(packed / precision) / (precision - 1);
	*y = std::fmod(packed, precision) / (precision - 1);
}
inline float unpack_dist(float packed){
	return std::floor(packed / precision) / (precision - 1);
}
inline float unpack_var(float packed){
	return std::fmod(packed, precision) / (precision - 1);
}
#endif
#endif //__DISTANCE_PACK__