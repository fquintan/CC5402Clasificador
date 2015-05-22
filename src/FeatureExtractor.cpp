#include "FeatureExtractor.hpp"
#include "Utils.cpp"



FeatureExtractor::FeatureExtractor(){}
FeatureExtractor::~FeatureExtractor(){}

cv::Mat FeatureExtractor::extractDescriptors(std::string img_file){

    const cv::Mat input = cv::imread(img_file, 0); //Load as grayscale
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 500;
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;

	detector.detect(input, keypoints);

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SiftDescriptorExtractor extractor;
	cv::Mat descriptors;

	extractor.compute(input, keypoints, descriptors);
    
    return descriptors;

}

std::vector<cv::Mat> FeatureExtractor::extractDescriptors(std::vector<std::string> img_files){
	std::vector<cv::Mat> descriptors;
	for(std::string img_file : img_files){
		cv::Mat current = extractDescriptors(img_file);
		descriptors.push_back(current);
	}
		
	return descriptors;
}
