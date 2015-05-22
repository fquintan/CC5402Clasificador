#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <string>


class FeatureExtractor{
public:
	FeatureExtractor();
	~FeatureExtractor();
	cv::Mat extractDescriptors(std::string img_file);
	std::vector<cv::Mat> extractDescriptors(std::vector<std::string> img_files);
};