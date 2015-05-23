#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>

int main(int argc, char **argv){
	FeatureExtractor extractor;

	std::string dir = "/home/felipe/Documents/BusquedaContenido/tarea3/CC5402Clasificador/imagenes/cat_test2";
	std::vector<std::string> files = Utils::getAllFilenames(dir);
	std::vector<cv::Mat> descriptors = extractor.extractDescriptors(files);

	int i;
	int nDescriptors = 0;
	int nFiles = files.size();
	for (i = 0; i < nFiles; i++){
		nDescriptors += descriptors[i].rows;
	}
	std::cout << "Number of descriptors: " << nDescriptors << std::endl;
	std::cout << "Dimension of descriptors: " << descriptors[0].cols << std::endl;

	int nClusters = 15;
	ClusterComputer clusterComputer(nClusters);
	cv::Mat labels;
	cv::Mat centers;

	clusterComputer.compute(descriptors, labels, centers);

	std::ofstream outputFile;
	std::string outputName = "descriptores.txt";
	outputFile.open(outputName);
	std::cout << "Saving descriptors in file: " << outputName << std::endl;

	int c = 0;
	for (i = 0; i < nFiles; i++){
		std::vector<float> BOVWDescriptor(nClusters);
		std::vector<int> counter(nClusters);
		nDescriptors = descriptors[i].rows;
		for (int j = 0; j < nDescriptors; j++){
			counter[labels.at<int>(c++,0)]++;
		}
		for (int j = 0; j < nClusters; j++){
			BOVWDescriptor[j] = counter[j] / (float) nDescriptors;
		}
		// Save descriptor to file
		outputFile << Utils::vectorToString(BOVWDescriptor) << std::endl;
	}
	outputFile.close();

}