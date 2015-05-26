#include "FeatureExtractor.hpp"
#include "ClusterComputer.hpp"
#include "Utils.cpp"
#include <iostream>
#include <fstream>
#include <opencv2/ml/ml.hpp>


void usage(){
	std::cout << "Modo de uso: extractor <inputDir1> <inputDir2> <output>" << std::endl;
	std::cout << "inputDir: Directorio donde se encuentran las imagenes de entrenamiento" << std::endl;
	std::cout << "output: Nombre del archivo para guardar los descriptores" << std::endl;

}

int main(int argc, char **argv){
	if (argc < 2){
		usage();
		return 1;
	}
	std::vector<std::string> c1_image_dir (3);
	c1_image_dir[0] = "imagenes/car_train2";
	c1_image_dir[1] = "imagenes/car_val";
	c1_image_dir[2] = "imagenes/car_test";
	std::vector<std::string> c2_image_dir (3);
	c2_image_dir[0] = "imagenes/cat_train2";
	c2_image_dir[1] = "imagenes/cat_val";
	c2_image_dir[2] = "imagenes/cat_test";
	std::vector<std::string> c3_image_dir (3);
	c3_image_dir[0] = "imagenes/sheep_train2";
	c3_image_dir[1] = "imagenes/sheep_val";
	c3_image_dir[2] = "imagenes/sheep_test";

	std::string outputName = argv[1];

	// Opening image directories and finding image files for each class 
	std::vector<std::string> files = Utils::getAllFilenames(c1_image_dir[0]);
	std::cout << "Abriendo " << c1_image_dir[0] << std::endl;
	std::cout << "Se encontraron " << files.size() << " archivos" << std::endl;
	std::vector<int> files_index (3);
	files_index[0] = files.size() - 1;

	std::vector<std::string> aux = Utils::getAllFilenames(c2_image_dir[0]);
	std::cout << "Abriendo " << c2_image_dir[0] << std::endl;
	std::cout << "Se encontraron " << aux.size() << " archivos" << std::endl;
	files.insert(files.end(), aux.begin(), aux.end());
	files_index[1] = files.size() - 1;

	aux = Utils::getAllFilenames(c3_image_dir[0]);
	std::cout << "Abriendo " << c3_image_dir[0] << std::endl;
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
	std::cout << "Calculando descriptores BOVW" << std::endl;
	std::cout << "Guardando descriptores en el archivo: " << outputName << std::endl;

	c = 0;
	int file_index = 0;
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
		outputFile << file_index << " " <<  Utils::vectorToString(BOVWDescriptor) << std::endl;
		if (i == files_index[file_index]){
			file_index++;
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