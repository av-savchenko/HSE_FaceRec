#include "stdafx.h"

#include "RecognitionThreadPool.h"
#include "distance_pack.h"

#include <set>
#include <list>
using namespace std;


NNMethod RecognitionThreadPool::nnMethod = NNMethod::Simple;

RecognitionThreadPool::RecognitionThreadPool( vector<FaceImage*>& dbImages, float falseAcceptRate)
{
	image_dists=NULL;

	InitializeCriticalSection(&csTasksCount);
	jobCompletedEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	threadPool.SetThreadpoolMin(NUM_OF_THREADS);
	threadPool.SetThreadpoolMax(NUM_OF_THREADS);


	int dbSize=dbImages.size();
	float threshold=0;

#if 0
	if(DirectedEnumeration::MAX_ROW_SIZE<dbSize*NUM_OF_THREADS){
		vector<float> otherClassesDists(dbSize);
		float* dist_vector=new float[dbSize];

		int facesInOneThread=dbSize/NUM_OF_THREADS;
		int size=facesInOneThread;
		int thread_num=-1;
		int min_index=0;

		int image_dists_size=dbSize*(dbSize-(NUM_OF_THREADS-1)*facesInOneThread);
		image_dists=new DirectedEnumeration::ImageDist[image_dists_size];
		memset(image_dists,0,image_dists_size*sizeof(DirectedEnumeration::ImageDist));

		DirectedEnumeration::ImageDist * current_thread_image_dists;

		for(int i=0;i<dbSize;++i){
			if(thread_num<NUM_OF_THREADS-1 && i%facesInOneThread==0){
				++thread_num;
				if(thread_num==NUM_OF_THREADS-1){
					size=dbSize-thread_num*facesInOneThread;
				}
				min_index=thread_num*facesInOneThread;
				current_thread_image_dists=&image_dists[min_index*facesInOneThread];
				//std::cerr<<i<<" "<<facesInOneThread<<" "<<min_index<<" "<<size<<"\t"<<current_thread_image_dists<<std::endl;
			}

			float otherMinDist=FLT_MAX;
			for(int j=0;j<dbSize;++j){
				dist_vector[j] =
#ifdef NEED_DEM
					dbImages[j]->distance(dbImages[i]);
#else
					0;
#endif
				if(j>=min_index && j<min_index+size){
					current_thread_image_dists[(i-min_index)*size+j-min_index]=
						DirectedEnumeration::ImageDist(dist_vector[j], j-min_index);
				}
				if(dbImages[i]->personName!=dbImages[j]->personName && dist_vector[j]<otherMinDist)
					otherMinDist=dist_vector[j];
			}
			otherClassesDists[i]=otherMinDist;

			//std::sort(&image_dists[i*facesInOneThread],&image_dists[i*facesInOneThread+size]);
		}
		delete[] dist_vector;
		threshold=DirectedEnumeration::getThreshold(otherClassesDists,falseAcceptRate);

		size=facesInOneThread;
		for(thread_num=0;thread_num<NUM_OF_THREADS;++thread_num){
			if(thread_num==NUM_OF_THREADS-1){
				size=dbSize-thread_num*facesInOneThread;
			}
			int min_index=thread_num*facesInOneThread;
			vector<FaceImage*> facesChunk(size);
			for(int j=0;j<size;++j){
				facesChunk[j]=dbImages[min_index+j];
			}
			//std::cerr<<min_index<<" "<<size<<"\t"<<&image_dists[min_index*facesInOneThread]<<std::endl;
			facesData.push_back(new FacesData(facesChunk, threshold, 
				&image_dists[min_index*facesInOneThread], size));
		}
	}
	else
#endif
	{
		float* dist_matrix=NULL;
#if defined (NEED_DEM) || defined(NEED_SMALL_WORLD)
		dist_matrix=new float[dbSize*dbSize];
		char* dist_matrix_file = "dist_matrix.txt";
		ifstream ifs(dist_matrix_file);
		if (ifs){
			vector<float> otherClassesDists(dbSize);

			for (int i = 0; i < dbSize; ++i){
				float otherMinDist = FLT_MAX;
				for (int j = 0; j < dbSize; ++j){
					//dist_matrix[i*dbSize + j] = dbImages[j]->distance(dbImages[i]);
					ifs >> dist_matrix[i*dbSize + j];
					if (dbImages[i]->personName != dbImages[j]->personName && unpack_var(dist_matrix[i*dbSize + j]) < otherMinDist)
						otherMinDist = unpack_var(dist_matrix[i*dbSize + j]);
				}
				otherClassesDists[i] = otherMinDist;
			}
			threshold = DirectedEnumeration<FaceImage*>::getThreshold(otherClassesDists, falseAcceptRate);
		}
		else{
			memset(dist_matrix, 0, sizeof(float)*dbSize*dbSize);
			{
				ofstream ofs(dist_matrix_file);
				vector<float> otherClassesDists(dbSize);
				float var;

				float x, y;

				for (int i = 0; i < dbSize; ++i){
					float otherMinDist = FLT_MAX;
					for (int j = 0; j < dbSize; ++j){
						dist_matrix[i*dbSize + j] = dbImages[j]->distance(dbImages[i], &var);
						if (dbImages[i]->personName != dbImages[j]->personName && dist_matrix[i*dbSize + j] < otherMinDist)
							otherMinDist = dist_matrix[i*dbSize + j];

						//std::cout << dist_matrix[i*dbSize + j]<<' '<<var << '\n';
						dist_matrix[i*dbSize + j] = pack(var,dist_matrix[i*dbSize + j]);
						/*std::cout << "packed: "<<dist_matrix[i*dbSize + j] << '\n';
						float x, y;
						unpack(dist_matrix[i*dbSize + j], &x, &y);
						std::cout << "unpacked:" << x << ' ' << y << '\n';*/
						
						ofs << dist_matrix[i*dbSize + j] << ' ';
					}
					ofs << endl;
					//std::exit(0);
					otherClassesDists[i] = otherMinDist;
				}
				threshold = DirectedEnumeration<FaceImage*>::getThreshold(otherClassesDists, falseAcceptRate);
			}
		}
#endif
		vector<int> indices[NUM_OF_THREADS];
		divideDatabasesIntoClasses(NUM_OF_THREADS,dbSize, dist_matrix, indices);

		for(int i=0;i<NUM_OF_THREADS;++i){
			int size=indices[i].size();
			vector<FaceImage*> facesChunk(size);
			for(int j=0;j<size;++j){
				facesChunk[j]=dbImages[indices[i][j]];
			}
			//std::cout << "hi\n";
			facesData.push_back(new FacesData(facesChunk, threshold, dist_matrix, dbSize, &indices[i]));
		}

		delete[] dist_matrix;
	}
}
RecognitionThreadPool::~RecognitionThreadPool(){

	DeleteCriticalSection(&csTasksCount);
	CloseHandle(jobCompletedEvent);

	for(vector<FacesData*>::iterator iter=facesData.begin();iter!=facesData.end();++iter)
		delete (*iter);
	
	if(image_dists!=NULL){
		delete[] image_dists;
	}
}

void RecognitionThreadPool::setImageCountToCheck(int imageCountToCheck)
{
	for(vector<FacesData*>::iterator iter=facesData.begin();iter!=facesData.end();++iter)
		(*iter)->setImageCountToCheck(imageCountToCheck);
}

void RecognitionThreadPool::init(){
	for(vector<FacesData*>::iterator iter=facesData.begin();iter!=facesData.end();++iter)
		(*iter)->init();
}
float RecognitionThreadPool::getAverageCheckedPercent(){
	float avgCheckedPercent=0;
	for(vector<FacesData*>::iterator iter=facesData.begin();iter!=facesData.end();++iter)
		avgCheckedPercent+=(*iter)->avgCheckedPercent;
	return avgCheckedPercent/facesData.size();
}
FaceImage* RecognitionThreadPool::recognizeTestImage(FaceImage* test)
{
	vector<FacesData*>::iterator iter;
	for(iter=facesData.begin();iter!=facesData.end();++iter)
		(*iter)->setTestImage(test);

	closestImage=0;
	closestDistance=1000000;
	
	DirectedEnumeration<FaceImage*>::terminateSearch = 0;
	SmallWorld<FaceImage*>::terminateSearch = 0;

#if NUM_OF_THREADS==1
	for(iter=facesData.begin();iter!=facesData.end();++iter){
		(*(*iter))();
		if((*iter)->closestDistance<closestDistance){
			closestDistance=(*iter)->closestDistance;
			closestImage=(*iter)->closestImage;
		}
	}
#else

	EnterCriticalSection(&csTasksCount);
	ResetEvent(jobCompletedEvent);
	//tasksCount=facesData.size();

	for(iter=facesData.begin();iter!=facesData.end();++iter)
		threadPool.QueueUserWorkItem(processTask,this, (LPVOID)*iter);
	LeaveCriticalSection(&csTasksCount);
	threadPool.WaitForAll(); 
#endif
	
	return closestImage;
}
inline void RecognitionThreadPool::taskCompleted(FacesData* faces)
{
	EnterCriticalSection(&csTasksCount);

	if(faces->closestDistance<closestDistance){
		closestDistance=faces->closestDistance;
		closestImage=faces->closestImage;
	}
	
	LeaveCriticalSection(&csTasksCount);
}
DWORD WINAPI RecognitionThreadPool::processTask(LPVOID param1, LPVOID param2)
{
	FacesData* faces=(FacesData*)param2;

	(*faces)();

	RecognitionThreadPool* recognitionThreadPool=(RecognitionThreadPool*)param1;
	recognitionThreadPool->taskCompleted(faces);
	return 0;
}


#define SEQUENCE 1
#define RANDOM 2
#define K_MEDOIDS 3
#define COMPLETE_LINK 4
#define GAAC 5
#define EQUAL_CLUSTERS 6

#define DIVISION_ALGO SEQUENCE
//#define DIVISION_ALGO RANDOM
//#define DIVISION_ALGO K_MEDOIDS
//#define DIVISION_ALGO COMPLETE_LINK
//#define DIVISION_ALGO GAAC
//#define DIVISION_ALGO EQUAL_CLUSTERS

//#define USE_CLOSEST
#ifdef USE_CLOSEST
const float INIT_DIST=10000000;
#define CMP <
#else
const float INIT_DIST=0;
#define CMP >
#endif

void RecognitionThreadPool::divideDatabasesIntoClasses(int numOfClasses, int dbSize, float* distMatrix, vector<int>* indices)
{
#if DIVISION_ALGO==SEQUENCE
	int facesInOneThread=dbSize/numOfClasses;
	for(int i=0;i<numOfClasses;++i){
		int size=(i<numOfClasses-1)?facesInOneThread:
			dbSize-i*facesInOneThread;
		indices[i].resize(size);
		for(int j=0;j<size;++j)
			indices[i][j]=i*facesInOneThread+j;
	}
#elif DIVISION_ALGO==RANDOM
	for(int i=0;i<dbSize;++i){
		int classInd=rand()%numOfClasses;
		indices[classInd].push_back(i);
	}
#elif DIVISION_ALGO==K_MEDOIDS
	int* medoidIndices=new int[numOfClasses];
	for(int c=0;c<numOfClasses;++c){
		//medoidIndices[c]=rand()%dbSize;
		medoidIndices[c]=dbSize/numOfClasses*c;
	}
	for(int t=0;t<50;++t){
		for(int c=0;c<numOfClasses;++c)
			indices[c].clear();
		//expectation
		for(int i=0;i<dbSize;++i){
			int bestCluster=-1;
			float minDist=100000;
			for(int c=0;c<numOfClasses;++c){
				int ind=i*dbSize+medoidIndices[c];
				if(distMatrix[ind]<minDist){
					minDist=distMatrix[ind];
					bestCluster=c;
				}
			}
			indices[bestCluster].push_back(i);
		}

		//maximization
		for(int c=0;c<numOfClasses;++c){
			float minDistSum=10000000;
			int bestInd=-1;

			int clusterSize=indices[c].size();
			for(int i=0;i<clusterSize;++i){
				float sum=0;
				for(int i1=0;i1<clusterSize;++i1){
					sum+=distMatrix[indices[c][i]*dbSize+indices[c][i1]];
				}
				sum/=clusterSize;
				if(sum<minDistSum){
					minDistSum=sum;
					bestInd=i;
				}
			}
			medoidIndices[c]=bestInd;
		}
	}
	delete[] medoidIndices;
#elif DIVISION_ALGO==COMPLETE_LINK
	int* classIndices=new int[dbSize];
	for(int i=0;i<dbSize;++i)
		classIndices[i]=i;
	
	for(int iter=0;iter<dbSize-numOfClasses;++iter){
		float minDist=INIT_DIST;
		int iMin=-1,i1Min=-1;
		for(int i=0;i<dbSize;++i)
			if(classIndices[i]==i)
				for(int i1=0;i1<dbSize;++i1)
					if(i1!=i && classIndices[i1]==i1 && distMatrix[i*dbSize+i1] CMP minDist){
						minDist=distMatrix[i*dbSize+i1];
						iMin=i;
						i1Min=i1;
					}

		//sequenceOfUnions.push_back(make_pair(iMin,i1Min));
		for(int i=0;i<dbSize;++i)
			if(classIndices[i]==i1Min)
				classIndices[i]=iMin;
		//std::cout<<"i Min="<<iMin<<' '<<i1Min<<'\n';
		for(int i=0;i<dbSize;++i)
			if(classIndices[i]==i){
				if(distMatrix[iMin*dbSize+i] CMP distMatrix[i1Min*dbSize+i])
					distMatrix[iMin*dbSize+i]=distMatrix[i1Min*dbSize+i];
				if(distMatrix[i*dbSize+iMin] CMP distMatrix[i*dbSize+i1Min])
					distMatrix[i*dbSize+iMin]=distMatrix[i*dbSize+i1Min];
			}
	}

	set<int> prevChecked;
	for(int c=0;c<numOfClasses;++c){
		int newIndex=-1;
		for(int i=0;i<dbSize;++i){
			if(newIndex==-1 && prevChecked.find(classIndices[i])==prevChecked.end()){
				newIndex=classIndices[i];
			}
			if(classIndices[i]==newIndex){
				indices[c].push_back(i);
			}
		}
		prevChecked.insert(newIndex);
		//std::cout<<indices[c].size()<<'\n';
	}
	delete[] classIndices;
#elif DIVISION_ALGO==GAAC
	vector<pair<float, list<int> > > clusters;
	//vector<pair<int,int> > sequenceOfUnions;
	for(int i=0;i<dbSize;++i)
		clusters.push_back(make_pair(0, list<int>(1, i)));
	
	for(int iter=0;iter<dbSize-numOfClasses;++iter){
		float minDist=INIT_DIST;
		int iMin=-1,i1Min=-1;
		int clustersCount=clusters.size();
		for(int i=0;i<clustersCount;++i)
			for(int i1=0;i1<clustersCount;++i1)
				if(i1!=i){
					float totalDist=clusters[i].first+clusters[i1].first;
					for(list<int>::const_iterator iter1=clusters[i].second.begin();iter1!=clusters[i].second.end();++iter1)
						for(list<int>::const_iterator iter2=clusters[i1].second.begin();iter2!=clusters[i1].second.end();++iter2)
							totalDist+=distMatrix[(*iter1) * dbSize + (*iter2)];
					//totalDist/=(clusters[i].second.size()+clusters[i1].second.size())*(clusters[i].second.size()+clusters[i1].second.size()-1);
					if(totalDist CMP minDist){
						minDist=totalDist;
						iMin=i;
						i1Min=i1;
					}
				}

		clusters[iMin].second.splice(clusters[iMin].second.end(),clusters[i1Min].second);
		clusters[iMin].first=minDist;//*(clusters[iMin].second.size()+clusters[i1Min].second.size())*(clusters[iMin].second.size()+clusters[i1Min].second.size()-1);
		clusters.erase(clusters.begin()+i1Min);
	}

	for(int c=0;c<numOfClasses;++c){
		indices[c].assign(clusters[c].second.begin(), clusters[c].second.end());
	}
#elif DIVISION_ALGO==EQUAL_CLUSTERS
	int* isIncludedInClass=new int[dbSize];
	for(int i=0;i<dbSize;++i)
		isIncludedInClass[i]=0;

	int facesInOneThread=dbSize/numOfClasses;

	for(int c=0;c<numOfClasses-1;++c){
		indices[c].resize(facesInOneThread);

		for(int i=0;i<dbSize;++i)
			if(!isIncludedInClass[i]){
				indices[c][0]=i;
				isIncludedInClass[i]=1;
				break;
			}
		
		for(int curSize=1;curSize<facesInOneThread;++curSize){
			float minDist=INIT_DIST;
			int iMin=-1;
			for(int i=0;i<dbSize;++i)
				if(!isIncludedInClass[i]){
					float avgDist=0;
					for(int item=0;item<curSize;++item)
						avgDist+=distMatrix[item*dbSize+i];
					avgDist/=curSize;
					if(avgDist CMP minDist){
						minDist=avgDist;
						iMin=i;
					}
				}
			indices[c][curSize]=iMin;
			isIncludedInClass[iMin]=1;
		}

	}
	
	indices[numOfClasses-1].reserve(facesInOneThread);
	for(int i=0;i<dbSize;++i)
		if(!isIncludedInClass[i]){
			indices[numOfClasses-1].push_back(i);
		}

	delete[] isIncludedInClass;
#endif
	for(int c=0;c<numOfClasses;++c){
		std::cout<<indices[c].size()<<'\n';
	}
}
