#include "FeatureExtractor.hpp"
#include "Utils.cpp"



FeatureExtractor::FeatureExtractor(){}
FeatureExtractor::~FeatureExtractor(){}

cv::Mat FeatureExtractor::extractDescriptors(std::string img_file){

    const cv::Mat input = cv::imread(img_file, 0); //Load as grayscale
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 500;
	cv::SurfFeatureDetector detector( minHessian );
	std::vector<cv::KeyPoint> keypoints;

	detector.detect(input, keypoints);

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptors;

	extractor.compute(input, keypoints, descriptors);
    
    return descriptors;
    // TODO: Extract clustering into its own class
	// int clusterCount = 15;
	// cv::Mat labels;
	// int attempts = 5;
	// cv::Mat centers;
	// cv::kmeans(descriptors, clusterCount, labels,
	// 	cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
	// 	attempts, cv::KMEANS_PP_CENTERS, centers);
}

std::vector<cv::Mat> FeatureExtractor::extractDescriptors(std::vector<std::string> img_files){
	std::vector<cv::Mat> descriptors(img_files.size());
	for(std::string img_file : img_files){
		descriptors.push_back(extractDescriptors(img_file));
	}
	return descriptors;
}
