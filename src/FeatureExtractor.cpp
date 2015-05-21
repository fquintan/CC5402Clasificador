#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //Thanks to Alessandro
#include <iostream>
#include <string>


void printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;

        switch (type) {
        case cv::Param::BOOLEAN:
            typeText = "bool";
            break;
        case cv::Param::INT:
            typeText = "int";
            break;
        case cv::Param::REAL:
            typeText = "real (double)";
            break;
        case cv::Param::STRING:
            typeText = "string";
            break;
        case cv::Param::MAT:
            typeText = "Mat";
            break;
        case cv::Param::ALGORITHM:
            typeText = "Algorithm";
            break;
        case cv::Param::MAT_VECTOR:
            typeText = "Mat vector";
            break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread("/home/felipe/Documents/BusquedaContenido/tarea3/CC5402Clasificador/imagenes/cat_test/000011.jpg", 0); //Load as grayscale
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	cv::SurfFeatureDetector detector( minHessian );

	std::vector<cv::KeyPoint> keypoints;

	detector.detect(input, keypoints);

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptors;
	printf("Detector params\n");
	printParams(&detector);
	printf("Extractor params\n");
	printParams(&extractor);

	extractor.compute(input, keypoints, descriptors);
    
	std::cout << "Matrix type: " << type2str(descriptors.type()) << std::endl; 
	int rows = descriptors.rows;
	int cols = descriptors.cols;
	printf("Rows: %d\n", rows);
	printf("Columns: %d\n", cols);

	int clusterCount = 15;
	cv::Mat labels;
	int attempts = 5;
	cv::Mat centers;
	cv::kmeans(descriptors, clusterCount, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, cv::KMEANS_PP_CENTERS, centers);

	rows = centers.rows;
	cols = centers.cols;
	printf("Rows: %d\n", rows);
	printf("Columns: %d\n", cols);

    return 0;
}

