#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>
#include <opencv2/ml/ml.hpp>
#include <math.h>  
#include <unordered_map>
#include <sstream>
#include <string>
#include <limits>

void usage(){
	std::cout << "Modo de uso: extractor <inputDir1> <inputDir2> <output>" << std::endl;
	std::cout << "inputDir: Directorio donde se encuentran las imagenes de entrenamiento" << std::endl;
	std::cout << "output: Nombre del archivo para guardar los descriptores" << std::endl;

}

int nearestCluster(std::string descriptorAsString, cv::Mat &centers){
	// std::cout << "Entered nearestCluster" << std::endl;

	std::vector<std::string> descriptorAsVector = Utils::split(descriptorAsString, ',');

	int nDimensions = centers.cols;
	int nClusters = centers.rows;
	float distance = 0.0;
	float diff;
	float coordinate;
	float min_distance = std::numeric_limits<float>::max();
	int label = 0;
	int i;
	for (i = 0; i < nClusters; ++i){
		distance = 0.0;
		for (int d = 0; d < nDimensions; ++d){
			coordinate = std::stof(descriptorAsVector[d]);
			diff = coordinate - centers.at<float>(i,d);
			distance += diff * diff;
		}
		if (distance < min_distance){
			min_distance = distance;
			label = i;
		}
	}
	// std::cout << "Exited nearestCluster" << std::endl;

	return label;
}



int main(int argc, char **argv){
	std::vector<float> train_local_descriptors;
	std::vector<std::string> train_local_descriptor_files(3);
	train_local_descriptor_files[0] = "imagenes/descriptores_locales/train_car";
	train_local_descriptor_files[1] = "imagenes/descriptores_locales/train_cat";
	train_local_descriptor_files[2] = "imagenes/descriptores_locales/train_bird";

	int count = 0;
	int dimension = 128;
	int max_files = 160;
	int file_counter;

	for (int i = 0; i < 3; ++i){
		/* code */
		std::ifstream infile(train_local_descriptor_files[i]);
		file_counter = 0;
		std::string line;
		std::vector<std::string> vectorAsString;
		while (std::getline(infile, line)){
			if (Utils::stringEndsWith(line, "jpg")){
				file_counter++;
				std::getline(infile, line);
				if(file_counter >= max_files){
					break;
				}
				continue;		
			}
			vectorAsString.clear();
			vectorAsString = Utils::split(line, ',');
			for(int j = 0; j < dimension; j++){
				train_local_descriptors.push_back(std::stof(vectorAsString[j]));
			}
			count++;
		    // process pair (a,b)
		}
		infile.close();
	}

	std::cout << "Read " << train_local_descriptors.size() << " local descriptors from file" << std::endl;

	cv::Mat descriptor_matrix = cv::Mat(count, dimension, CV_32FC1);
	memcpy(descriptor_matrix.data, train_local_descriptors.data(), train_local_descriptors.size()*sizeof(float));

	std::cout << "Starting k-means clustering" << std::endl;

	int nClusters = std::stoi(argv[1]);
	ClusterComputer clusterComputer(nClusters);
	cv::Mat labels;
	cv::Mat centers;
	clusterComputer.compute(descriptor_matrix, labels, centers);
	std::cout << "Cumputed clusters" << std::endl;
	std::cout << "Number of centers: " << centers.rows << std::endl;
	std::cout << "Centers dimension: " << centers.cols << std::endl;



	std::vector<std::string> local_descriptor_files (9);
	// Inicializar
	local_descriptor_files[0] = "imagenes/descriptores_locales/train_car";
	local_descriptor_files[1] = "imagenes/descriptores_locales/train_cat";
	local_descriptor_files[2] = "imagenes/descriptores_locales/train_bird";
	local_descriptor_files[3] = "imagenes/descriptores_locales/test_car";
	local_descriptor_files[4] = "imagenes/descriptores_locales/test_cat";
	local_descriptor_files[5] = "imagenes/descriptores_locales/test_bird";
	local_descriptor_files[7] = "imagenes/descriptores_locales/val_cat";
	local_descriptor_files[6] = "imagenes/descriptores_locales/val_car";
	local_descriptor_files[8] = "imagenes/descriptores_locales/val_bird";


	std::unordered_map<std::string, std::vector<int>> document_word_frequency;
	std::unordered_map<std::string, std::vector<float>> tf;

	std::unordered_map<std::string, int> document_word_counter;
	std::vector<int> word_frequency(nClusters, 0);
	int word_total = 0;
	std::vector<std::vector<std::string>> filenames(9);


	for (int i = 0; i < 9; ++i){
		std::cout << "Aproximating words for  " << local_descriptor_files[i] << std::endl;

		std::ifstream infile(local_descriptor_files[i]);
		
		std::string line;
		std::vector<std::string> vectorAsString;
		std::string currentFile = "";
		// std::cout << "File opened" << std::endl;
		
		while (std::getline(infile, line)){

			// std::cout << "Read line: " << line << std::endl;
			if (Utils::stringEndsWith(line, "jpg")){

				// std::cout << "Line ended with jpg" << std::endl;

				if(currentFile != ""){
				// Finished last file
					// std::cout << "Finished a file" << std::endl;
					for (int j = 0; j < nClusters; ++j){
						if(document_word_frequency[currentFile][j] > 0){
							word_frequency[j]++;
						}
						word_total += document_word_frequency[currentFile][j];
						tf[currentFile][j] = document_word_frequency[currentFile][j] / (float) document_word_counter[currentFile];
						// std::cout << "Updated frequency for last file" << std::endl;
					}
				}
				currentFile = line;
				filenames[i].push_back(currentFile);

				// std::cout << "Pushed filename into vector" << std::endl;
				document_word_frequency[currentFile] = *(new std::vector<int> (nClusters, 0));
				// std::cout << "Created vector for frequency" << std::endl;
				tf[currentFile] = *(new std::vector<float> (nClusters, 0.0));
				// std::cout << "Created vector for tf" << std::endl;
				std::getline(infile, line);
				document_word_counter[currentFile] = std::stoi(line);
				continue;		
			}

			int word = nearestCluster(line, centers);
			document_word_frequency[currentFile][word] = document_word_frequency[currentFile][word]+1;
		}
		for (int j = 0; j < nClusters; ++j){
			if(document_word_frequency[currentFile][j] > 0){
				word_frequency[j]++;
			}
			word_total += document_word_frequency[currentFile][j];
			tf[currentFile][j] = document_word_frequency[currentFile][j] / (float) document_word_counter[currentFile];
		}

	}

	std::cout << "Computing idf values"  << std::endl;

	std::vector<float> idf(nClusters, 0);
	for (int i = 0; i < nClusters; ++i){
		idf[i] = log(word_total / (float) word_frequency[i]);
	}

	std::string output_directory = "imagenes/BOVW_descriptors_" + std::to_string(nClusters) + "/";
	std::vector<std::string> output_filenames(9);
	output_filenames[0] = output_directory + "train_car";
	output_filenames[1] = output_directory + "train_cat";
	output_filenames[2] = output_directory + "train_bird";
	output_filenames[3] = output_directory + "test_car";
	output_filenames[4] = output_directory + "test_cat";
	output_filenames[5] = output_directory + "test_bird";
	output_filenames[6] = output_directory + "val_car";
	output_filenames[7] = output_directory + "val_cat";
	output_filenames[8] = output_directory + "val_bird";

	std::ofstream outputFile;
	std::vector<float> BOVW_descriptor(nClusters, 0.0);
	for (int i = 0; i < 9; ++i){
		std::cout << "Computing BOVW descriptors for " << output_filenames[i] << std::endl;

		outputFile.open(output_filenames[i]);
		int file_size = filenames[i].size();
		for (int j = 0; j < file_size; ++j){
			outputFile << filenames[i][j] << std::endl;
			for (int k = 0; k < nClusters; ++k){
				BOVW_descriptor[k] = tf[filenames[i][j]][k] * idf[k];
			}
			outputFile << Utils::vectorToString(BOVW_descriptor) << std::endl;
		}
		outputFile.close();
	}

}