#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>

void usage(){
	std::cout << "Modo de uso: extractor <inputDir> <output>" << std::endl;
	std::cout << "inputDir: Directorio donde se encuentran las imagenes de entrenamiento" << std::endl;
	std::cout << "output: Nombre del archivo para guardar los descriptores" << std::endl;

}

int main(int argc, char **argv){
	if (argc < 3){
		usage();
		return 1;
	}
	std::string inputDir = argv[1];
	std::string outputName = argv[2];

	std::vector<std::string> files = Utils::getAllFilenames(inputDir);
	std::cout << "Abriendo " << inputDir << std::endl;
	std::cout << "Se encontraron " << files.size() << " archivos" << std::endl;

	FeatureExtractor extractor;
	std::vector<cv::Mat> descriptors = extractor.extractDescriptors(files);

	int i;
	int nDescriptors = 0;
	int nFiles = files.size();
	for (i = 0; i < nFiles; i++){
		nDescriptors += descriptors[i].rows;
	}
	std::cout << "Descriptores locales calculados: " << nDescriptors << std::endl;
	std::cout << "Descriptores de dimension: " << descriptors[0].cols << std::endl;

	int nClusters = 15;
	ClusterComputer clusterComputer(nClusters);
	cv::Mat labels;
	cv::Mat centers;

	std::cout << "Calculando clusters" << std::endl;
	clusterComputer.compute(descriptors, labels, centers);

	std::ofstream outputFile;
	outputFile.open(outputName);
	std::cout << "Calculando descriptores BOVW" << std::endl;
	std::cout << "Guardando descriptores en el archivo: " << outputName << std::endl;

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