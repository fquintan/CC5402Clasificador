#include "ClusterComputer.hpp"

ClusterComputer::ClusterComputer(int k): nClusters(k){}
ClusterComputer::~ClusterComputer(){}


void ClusterComputer::compute(std::vector<cv::Mat> descriptors, cv::Mat &labels, cv::Mat &centers){
	// First pack all descriptors into a single matrix
	cv::Mat descriptorContainer;
	int nDescriptors = descriptors.size();
	for (int i = 0; i < nDescriptors; ++i)
	{
		descriptorContainer.push_back(descriptors[i]);
	}

	// int clusterCount = 15;
	int attempts = 5;

	cv::kmeans(descriptorContainer, nClusters, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, cv::KMEANS_PP_CENTERS, centers);

}
