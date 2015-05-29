#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>
#include <opencv2/ml/ml.hpp>
#include <math.h>  


void usage(){
	std::cout << "Modo de uso: extractor <inputDir1> <inputDir2> <output>" << std::endl;
	std::cout << "inputDir: Directorio donde se encuentran las imagenes de entrenamiento" << std::endl;
	std::cout << "output: Nombre del archivo para guardar los descriptores" << std::endl;

}

void nearestCluster(cv::Mat &descriptors, cv::Mat &centers, std::vector<int> &labels){
	int nDescriptors = descriptors.cols;
	int nClusters = centers.cols;
	// assert descriptors.rows = centers.rows;
	// assert nDescriptors <= labels.size();
	int nDimensions = centers.rows;
	float distance = 0.0;
	float diff;
	std::vector<float> min_distances(1000.0, nDescriptors);
	for (int i = 0; i < nDescriptors; i++){
		for (int j = 0; j < nClusters; j++){
			// Compute distance to center
			distance = 0.0;
			for (int d = 0; d < nDimensions; d++){
				diff = (descriptors.at<float>(i,d) - centers.at<float>(j,d));
				distance +=  diff * diff;
			}
			if (distance < min_distances[i]){
				min_distances[i] = distance;
				labels[i] = j;
			}
		}
	}
}


int main(int argc, char **argv){
	if (argc < 2){
		usage();
		return 1;
	}
	std::vector<std::string> train_image_dirs (3);
	train_image_dirs[0] = "imagenes2/car_train";
	train_image_dirs[1] = "imagenes2/cat_train";
	train_image_dirs[2] = "imagenes2/sheep_train";
	
	std::vector<std::string> image_dirs (6);
	image_dirs[0] = "imagenes2/car_val";
	image_dirs[1] = "imagenes2/car_test";
	image_dirs[2] = "imagenes2/cat_val";
	image_dirs[3] = "imagenes2/cat_test";
	image_dirs[4] = "imagenes2/sheep_val";
	image_dirs[5] = "imagenes2/sheep_test";

	std::vector<std::string> image_dirs_ids (6);
	image_dirs_ids[0] = "3";
	image_dirs_ids[1] = "4";
	image_dirs_ids[2] = "5";
	image_dirs_ids[3] = "6";
	image_dirs_ids[4] = "7";
	image_dirs_ids[5] = "8";

	std::string outputName = argv[1];

	// Opening training image directories and finding image files for each class 
	std::vector<std::string> files = Utils::getAllFilenames(train_image_dirs[0]);
	std::cout << "Abriendo " << train_image_dirs[0] << std::endl;
	std::cout << "Se encontraron " << files.size() << " archivos" << std::endl;
	std::vector<int> files_index (3);
	files_index[0] = files.size() - 1;

	std::vector<std::string> aux = Utils::getAllFilenames(train_image_dirs[1]);
	std::cout << "Abriendo " << train_image_dirs[1] << std::endl;
	std::cout << "Se encontraron " << aux.size() << " archivos" << std::endl;
	files.insert(files.end(), aux.begin(), aux.end());
	files_index[1] = files.size() - 1;

	aux = Utils::getAllFilenames(train_image_dirs[2]);
	std::cout << "Abriendo " << train_image_dirs[2] << std::endl;
	std::cout << "Se encontraron " << aux.size() << " archivos" << std::endl;
	files.insert(files.end(), aux.begin(), aux.end());
	files_index[2] = files.size() - 1;

	FeatureExtractor extractor;
	std::vector<cv::Mat> descriptors = extractor.extractDescriptors(files);

	int i;
	int c = 0;
	int nDescriptors = 0;
	int nFiles = files.size();
	std::vector<int> descriptor_class_index (3);
	for (i = 0; i < nFiles; i++){
		nDescriptors += descriptors[i].rows;
		if(i == files_index[c]){
			descriptor_class_index[c] = nDescriptors - 1;
			c++;
		}
	}
	std::cout << "Descriptores locales calculados: " << nDescriptors << std::endl;

	int nClusters = 100;
	ClusterComputer clusterComputer(nClusters);
	cv::Mat labels;
	cv::Mat centers;

	std::cout << "Calculando clusters" << std::endl;
	clusterComputer.compute(descriptors, labels, centers);

	std::ofstream outputFile;
	outputFile.open(outputName);
	std::cout << "Aproximando descriptores locales a palabras visuales" << std::endl;
	
	std::vector<std::vector<int>> dir_labels(6);
	std::vector<std::vector<int>> descriptor_indexes(6);
	cv::Mat container;
	// Aproximate each local descriptor to its nearest "word" 
	for (int i = 0; i < image_dirs.size(); ++i){
		std::cout << "Holi" << std::endl;
		std::string current_dir = image_dirs[i];
		std::cout << "Abriendo " << current_dir  << std::endl;
		std::vector<std::string> img_files = Utils::getAllFilenames(current_dir);
		std::cout << "Se encontraron " << img_files.size() << " archivos" << std::endl;
		
		descriptor_indexes[i].reserve(img_files.size());
		descriptors.clear();
		container.release();

		std::vector<cv::Mat> descriptors = extractor.extractDescriptors(files);

		for (int j = 0; j < descriptors.size(); ++j){
			descriptor_indexes[i][j] = descriptors[j].rows - 1;
		}
		std::cout << "Uniendo matrices" << std::endl;
		// cv::Mat container;
		int nMats = descriptors.size();
		for (int a = 0; a < nMats; ++a){
			std::cout << "Matriz " << a << std::endl;
			container.push_back(descriptors[a]);
		}
		// container = Utils::joinMats(descriptors);
		std::cout << "Reservando memoria para etiquetas" << std::endl;

		dir_labels[i].reserve(container.rows);
		std::cout << "Aproximando palabras" << std::endl;

		nearestCluster(container, centers, dir_labels[i]);
		std::cout << "Finalizo la aproximacion de palabras" << std::endl;

	}

	std::cout << "Calculando frecuencias tf-idf" << std::endl;
	// Compute frequencies of every word for tf-idf
	std::vector<int> frequencies(0, nClusters);
	int total = nFiles;
	for (int i = 0; i < nDescriptors; ++i){
		frequencies[labels.at<int>(i,0)]++;
	}
	for (int i = 0; i < 6; ++i){
		int current_dir_size = dir_labels[i].size();
		total += current_dir_size;
		for (int j = 0; j < current_dir_size; ++j){
			frequencies[dir_labels[i][j]]++;
		}
	}
	std::vector<float> idf(nClusters);
	for (int i = 0; i < nClusters; ++i){
		idf[i] = log((float)total /frequencies[i]);
	} 

	std::cout << "Calculando descriptores BOVW" << std::endl;

	c = 0;
	std::vector<float> BOVWDescriptor(nClusters);
	std::vector<int> counter(0, nClusters);
	int file_index = 0;
	for (i = 0; i < nFiles; i++){
		nDescriptors = descriptors[i].rows;
		// First reset counter
		std::fill(counter.begin(), counter.end(), 0);
		for (int j = 0; j < nDescriptors; j++){
			counter[labels.at<int>(c++,0)]++;
		}
		for (int j = 0; j < nClusters; j++){
			BOVWDescriptor[j] = (counter[j] / (float) nDescriptors) / idf[j];
		}
		// Save descriptor to file
		outputFile << file_index << " " <<  Utils::vectorToString(BOVWDescriptor) << std::endl;
		if (i == files_index[file_index]){
			file_index++;
		}
	}

	c = 0;
	int file_count = 0;
	for (int i = 0; i < 6; ++i){
		int current_dir_size = dir_labels[i].size();
		std::fill(counter.begin(), counter.end(), 0);
		for (int j = 0; j < current_dir_size; ++j){
			// get index corresponding to the end of the current file
			int file_end = descriptor_indexes[i][file_count++];
			// count all words in the current image file
			for (int k = 0; k < file_end; ++k){
				counter[dir_labels[i][c++]]++;				
			}
			// compute tf-idf for current file
			for (int k = 0; k < nClusters; ++k){
				BOVWDescriptor[j] = (counter[j] / (float) nDescriptors) / idf[j];	
			}
			// Save descriptor to file
			outputFile << image_dirs_ids[i] << " " <<  Utils::vectorToString(BOVWDescriptor) << std::endl;
		}
	}

	outputFile.close();

	//////////////////////////////////////////////

/*    // Set up training data
    float labelsa[4] = {1.0, -1.0, -1.0, -1.0};
    cv::Mat labelsMat(4, 1, CV_32FC1, labelsa);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);*/

}