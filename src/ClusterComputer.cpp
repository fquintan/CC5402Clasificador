#include "ClusterComputer.hpp"
#include "Utils.cpp"

ClusterComputer::ClusterComputer(int k): nClusters(k){}
ClusterComputer::~ClusterComputer(){}

void ClusterComputer::compute(cv::Mat &descriptors, cv::Mat &labels, cv::Mat &centers){
	int attempts = 5;

	cv::kmeans(descriptors, nClusters, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, cv::KMEANS_PP_CENTERS, centers);

}


