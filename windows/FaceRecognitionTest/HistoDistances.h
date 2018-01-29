#ifndef __HISTO_DISTANCES_H__
#define __HISTO_DISTANCES_H__
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

#include "FaceImage.h"

#include "distances.h"
#include <numeric>


//#define CURRENT_DISTANCE cvflann::L2
//#define CURRENT_DISTANCE cvflann::L1

#ifdef USE_DNN_FEATURES

template<class T>
struct DnnDistance
{
	typedef cvflann::True is_kdtree_distance;
	typedef cvflann::True is_vector_space_distance;

	typedef T ElementType;
	typedef typename cvflann::Accumulator<T>::Type ResultType;

	template <typename Iterator1, typename Iterator2>
	ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
	{
		ResultType result = ResultType();
		ResultType diff;
		Iterator1 last = a + size;

		/* Process 4 items with each loop for efficiency. */
		while (a < last) {
			result += accum_dist(*a,*b,0);
			++a;
			++b;

			if ((worst_dist>0) && (result>worst_dist)) {
				return result;
			}
		}
		return result;
	}
	template <typename U, typename V>
	inline ResultType accum_dist(const U& a, const V& b, int) const
	{
		ResultType res = ResultType();
#if DISTANCE==MANHATTEN
		res = cvflann::abs(a - b);
#elif DISTANCE==EUC
		res = (a - b)*(a - b);
#elif DISTANCE==CHI_SQUARE
		if ((a + b)>0)
			res = (a - b)*(a - b) / (a + b);
#endif
		return res;
	}
};
#define CURRENT_DISTANCE DnnDistance
#else

template<class T>
struct Histos
{
	typedef cvflann::True is_kdtree_distance;
	typedef cvflann::True is_vector_space_distance;

	typedef T ElementType;
	typedef typename cvflann::Accumulator<T>::Type ResultType;

	ElementType operator()(const ElementType* x, const ElementType* y, size_t qty) const {
		const ElementType *lhs, *rhs;
		ElementType distances[POINTS_IN_H*POINTS_IN_W];
		operator()(x, y, qty, distances);
		ElementType result = std::accumulate(distances, distances + POINTS_IN_H*POINTS_IN_W, ElementType());
		return result;
	}

	void operator()(const ElementType* x, const ElementType* y, size_t qty, ElementType* distances) const {
		const ElementType *lhs, *rhs;
		for (int i = 0; i < POINTS_IN_H; ++i)
		{
			size_t iMin = i >= DELTA ? i - DELTA : 0;
			size_t iMax = i + DELTA;
			if (iMax >= POINTS_IN_H)
				iMax = POINTS_IN_H - 1;
			for (int j = 0; j < POINTS_IN_W; ++j){
				size_t jMin = j >= DELTA ? j - DELTA : 0;
				size_t jMax = j + DELTA;
				if (jMax >= POINTS_IN_W)
					jMax = POINTS_IN_W - 1;
				ElementType minSum = FLT_MAX;
				size_t i1 = i, j1 = j;
				//for(i1=iMin;i1<=iMax;++i1)
				//    for(j1=jMin;j1<=jMax;++j1)
				lhs = y + (i1*POINTS_IN_W + j1)*HISTO_SIZE;
				for (size_t i2 = iMin; i2 <= iMax; ++i2){
					for (size_t j2 = jMin; j2 <= jMax; ++j2){
						rhs = x + (i2*POINTS_IN_W + j2)*HISTO_SIZE;
						ElementType curSum = 0;
						const ElementType* llhs = lhs;
						for (size_t ind = 0; ind < HISTO_SIZE; ++ind){
							float d1 = *llhs;
							float d2 = *(rhs + ind);
							float kd1 = *(llhs + FEATURES_COUNT);
							float kd2 = *(rhs + ind + FEATURES_COUNT);
							float kd1_f = *(llhs + 2 * FEATURES_COUNT);
							float kd2_f = *(rhs + ind + 2 * FEATURES_COUNT);
							//kd1 = d1;
							//kd2 = d2;
							//std::cout << ((((char*)&kernel_histos[0][0][0][0]) - ((char*)&histos[0][0][0][0]))) << std::endl;
							curSum += accum_dist(d1, kd1, kd1_f, d2, kd2, kd2_f);
							++llhs;
						}
						if (curSum<0)
							curSum = 0;
						if (minSum>curSum){
							minSum = curSum;
						}
					}
				}
				*distances++ = fast_sqrt(minSum);
				//result+=minSum;
			}
		}
	}

	template <typename Iterator1, typename Iterator2>
	ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
	{
		ResultType result = ResultType();
		Iterator2 lhs, lhs2, lhs3;
		Iterator1 rhs;

		for (int i = 0; i<POINTS_IN_H; ++i)
		{
			size_t iMin = i >= DELTA ? i - DELTA : 0;
			size_t iMax = i + DELTA;
			if (iMax >= POINTS_IN_H)
				iMax = POINTS_IN_H - 1;
			for (int j = 0; j<POINTS_IN_W; ++j){
				size_t jMin = j >= DELTA ? j - DELTA : 0;
				size_t jMax = j + DELTA;
				if (jMax >= POINTS_IN_W)
					jMax = POINTS_IN_W - 1;
				ResultType minSum = FLT_MAX;
				size_t i1 = i, j1 = j;
				//for(i1=iMin;i1<=iMax;++i1)
				//    for(j1=jMin;j1<=jMax;++j1)
				lhs = b;
				lhs += (i1*POINTS_IN_W + j1)*HISTO_SIZE;

				lhs2 = b;
				lhs2 += (i1*POINTS_IN_W + j1)*HISTO_SIZE + FEATURES_COUNT;
				lhs3 = b;
				lhs3 += (i1*POINTS_IN_W + j1)*HISTO_SIZE + 2 * FEATURES_COUNT;

				for (size_t i2 = iMin; i2 <= iMax; ++i2){
					for (size_t j2 = jMin; j2 <= jMax; ++j2){
						rhs = a + (i2*POINTS_IN_W + j2)*HISTO_SIZE;
						ResultType curSum = 0;
						Iterator2 llhs = lhs, llhs2 = lhs2, llhs3 = lhs3;
						for (size_t ind = 0; ind<HISTO_SIZE; ++ind){
							ResultType d1 = *llhs++;
							ResultType d2 = *(rhs + ind);
							float kd1 = *llhs2++;
							float kd2 = *(rhs + ind + FEATURES_COUNT);
							float kd1_f = *llhs3++;
							float kd2_f = *(rhs + ind + 2 * FEATURES_COUNT);
							//kd1 = d1;
							//kd2 = d2;
							//std::cout << ((((char*)&kernel_histos[0][0][0][0]) - ((char*)&histos[0][0][0][0]))) << std::endl;
							curSum += accum_dist(d1, kd1, kd1_f, d2, kd2, kd2_f);
							//curSum+=accum_dist(d1,d2,1);
						}
						if (curSum<0)
							curSum = 0;
						if (minSum>curSum){
							minSum = curSum;
						}
					}
				}
				result += fast_sqrt(minSum);
				//result+=minSum;
				if ((worst_dist>0) && (result>worst_dist)) {
					return result;
				}
			}
		}
		/*
		ResultType sum;
		Iterator1 last = a + size;
		while (a < last) {
		sum=ResultType();
		for(int i=0;i<HISTO_SIZE;++i){
		sum += (ResultType)abs(*a - *b);
		++a;
		++b;
		}
		result += fast_sqrt(sum);

		if ((worst_dist>0)&&(result>worst_dist)) {
		return result;
		}
		}*/
		return result;
	}

	/**
	* Partial distance, used by the kd-tree.
	*/
	template <typename U, typename V>
	inline ResultType accum_dist(const U& a, const V& b, int) const
	{
		return  accum_dist(a, a, a, b, b, b);
	}

	template <typename U, typename V>
	inline ResultType accum_dist(const U& a, const U& ka, const U& kd1_f, const V& b, const V& kb, const V& kd2_f) const
	{
		ResultType res = ResultType();
#if DISTANCE==MANHATTEN
		res = cvflann::abs(a - b);
#elif DISTANCE==EUC
		res = (a - b)*(a - b);
#elif DISTANCE==CHI_SQUARE
		if ((a + b)>0)
			res = (a - b)*(a - b) / (a + b);
#elif DISTANCE==PNN || DISTANCE==KL
#if 1//!defined(NEED_FLANN) && 0
		if (kd1_f == a){
			if (a> 0 && b>0)
				res += a*std::log(a / b);
		}
		else
			res += a*(kd1_f - kd2_f);
#else
		if (a> 0 && b>0)
			res += a*std::log(a / b);
#endif
#elif DISTANCE==SIMPLIFIED_HOMOGENEITY
		float summary = (a + b) / 2;
		if (summary>0){
			float inv_sum = 2.f / summary;
#ifndef NEED_FLANN
			res += kd1_f*inv_sum*a*(ka*ka - summary*summary);
			res += kd2_f*inv_sum*b*(kb*kb - summary*summary);
#else
			if (ka>0){
				res += a*(ka*ka - summary*summary) / (2 * ka*summary);
			}
			if (kb>0){
				res += b*(kb*kb - summary*summary) / (2 * kb*summary);
			}
#endif
		}
#endif
		return res;
	}
};
#define CURRENT_DISTANCE Histos
#endif

#endif //__HISTO_DISTANCES_H__