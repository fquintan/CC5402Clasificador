#include "ClusterComputer.hpp"

ClusterComputer::ClusterComputer(){}
ClusterComputer::~ClusterComputer(){}


cv::Mat ClusterComputer::compute(std::vector<cv::Mat> descriptors){
	    // TODO: Extract clustering into its own class
	cv::Mat descriptorContainer;
	int nDescriptors = descriptors.size();
	for (int i = 0; i < nDescriptors; ++i)
	{
		descriptorContainer.push_back(descriptors[i]);
	}

	int clusterCount = 15;
	cv::Mat labels;
	int attempts = 5;
	cv::Mat centers;
	cv::kmeans(descriptorContainer, clusterCount, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, cv::KMEANS_PP_CENTERS, centers);
	return centers;
}
