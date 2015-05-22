#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>

int main(int argc, char **argv){
	FeatureExtractor extractor;

	std::string dir = "/home/felipe/Documents/BusquedaContenido/tarea3/CC5402Clasificador/imagenes/cat_test2";
	std::vector<std::string> files = Utils::getAllFilenames(dir);
	std::vector<cv::Mat> descriptors = extractor.extractDescriptors(files);

	int nDescriptors = 0;
	for (int i = 0; i < descriptors.size(); ++i)
	{
		nDescriptors += descriptors[i].rows;
	}
	std::cout << "Number of descriptors: " << nDescriptors << std::endl;
	std::cout << "Dimension of descriptors: " << descriptors[0].cols << std::endl;

	ClusterComputer clusterComputer;

	cv::Mat centers = clusterComputer.compute(descriptors);

	std::cout << "Number of centers: " << centers.rows << std::endl;
	std::cout << "Dimension of centers: " << centers.cols << std::endl;

}