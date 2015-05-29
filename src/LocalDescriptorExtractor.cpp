#include "FeatureExtractor.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>

int main(int argc, char **argv){

	std::vector<std::string> train_image_dirs (3);
	train_image_dirs[0] = "imagenes/car_train";
	train_image_dirs[1] = "imagenes/cat_train";
	train_image_dirs[2] = "imagenes/sheep_train";
	
	std::vector<std::string> val_image_dirs (3);
	val_image_dirs[0] = "imagenes/car_val";
	val_image_dirs[1] = "imagenes/cat_val";
	val_image_dirs[2] = "imagenes/sheep_val";

	std::vector<std::string> test_image_dirs (3);
	test_image_dirs[0] = "imagenes/car_test";
	test_image_dirs[1] = "imagenes/cat_test";
	test_image_dirs[2] = "imagenes/sheep_test";

	std::vector<std::string> class_name (3);
	class_name[0] = "car";
	class_name[1] = "cat";
	class_name[2] = "sheep";

	FeatureExtractor extractor;
	std::vector<std::string> files;
	std::vector<cv::Mat> directory_descriptors;

	std::string out_name = "imagenes/descriptores_locales/train_";
	std::ofstream outputFile;


	std::cout << "Calculando descriptores para conjunto de entrenamiento" << std::endl;

	int nFiles;
	for (int i = 0; i < 3; ++i){
		std::string filename = out_name + class_name[i];
		outputFile.open(filename);
		files = Utils::getAllFilenames(train_image_dirs[i]);
		nFiles = files.size();
		directory_descriptors = extractor.extractDescriptors(files);
		for (int j = 0; j < nFiles; ++j){
			int nDescriptors = directory_descriptors[j].rows;
			outputFile << files[j] << std::endl;
			outputFile << nDescriptors << std::endl; 
			outputFile << Utils::matToString(directory_descriptors[j]);
		}
		outputFile.close();
	}

	std::cout << "Calculando descriptores para conjunto de test" << std::endl;
	out_name = "imagenes/descriptores_locales/test_";

	for (int i = 0; i < 3; ++i){
		std::string filename = out_name + class_name[i];
		outputFile.open(filename);
		files = Utils::getAllFilenames(test_image_dirs[i]);
		nFiles = files.size();
		directory_descriptors = extractor.extractDescriptors(files);
		for (int j = 0; j < nFiles; ++j){
			int nDescriptors = directory_descriptors[j].rows;
			outputFile << files[j] << std::endl;
			outputFile << nDescriptors << std::endl; 
			outputFile << Utils::matToString(directory_descriptors[j]);
		}
		outputFile.close();
	}

	std::cout << "Calculando descriptores para conjunto de validacion" << std::endl;
	out_name = "imagenes/descriptores_locales/val_";

	for (int i = 0; i < 3; ++i){
		std::string filename = out_name + class_name[i];
		outputFile.open(filename);
		files = Utils::getAllFilenames(val_image_dirs[i]);
		nFiles = files.size();
		directory_descriptors = extractor.extractDescriptors(files);
		for (int j = 0; j < nFiles; ++j){
			int nDescriptors = directory_descriptors[j].rows;
			outputFile << files[j] << std::endl;
			outputFile << nDescriptors << std::endl; 
			outputFile << Utils::matToString(directory_descriptors[j]);
		}
		outputFile.close();
	}
	return 0;

}
