#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "Utils.cpp"
#include <fstream>

int main(int argc, char **argv){
	//////////////////////////////////////////////

	std::vector<std::string> training_data_files { 
		"imagenes2/BOVW_descriptors/train_car",
		"imagenes2/BOVW_descriptors/train_cat",
		"imagenes2/BOVW_descriptors/train_sheep"};

	std::vector<int> training_data_sizes(3);

	int dimension = 100;
	int count = 0;


	std::vector<float> training_data;
	for (int i = 0; i < 3; ++i){
		count = 0;
		std::ifstream infile(training_data_files[i]);
		
		std::string line;
		std::vector<std::string> vectorAsString;
		while (std::getline(infile, line)){
			if (Utils::stringEndsWith(line, "jpg")){
				// std::getline(infile, line);
				continue;		
			}
			vectorAsString.clear();
			vectorAsString = Utils::split(line, ',');
			for(int j = 0; j < dimension; j++){
				training_data.push_back(std::stof(vectorAsString[j]));
			}
			count++;
		}
		infile.close();
		training_data_sizes[i] = count;
	}

	count = training_data_sizes[0] + training_data_sizes[1] + training_data_sizes[2];

	std::cout << "Read " << count << " BOVW descriptors from training files" << std::endl;

	cv::Mat training_matrix = cv::Mat(count, dimension, CV_32FC1);
	memcpy(training_matrix.data, training_data.data(), training_data.size()*sizeof(float));
	
    // Set up training data
    std::vector<cv::Mat> labels(3);
    labels[0].create(count, 1, CV_32FC1);
    labels[1].create(count, 1, CV_32FC1);
    labels[2].create(count, 1, CV_32FC1);
    int c = 0;
    for (int i = 0; i < training_data_sizes[0]; ++i){
    	labels[0].at<float>(c,0) = 1.0;
    	labels[1].at<float>(c,0) = -1.0;
    	labels[2].at<float>(c,0) = -1.0;
    	c++;
    }
    for (int i = 0; i < training_data_sizes[1]; ++i){
    	labels[0].at<float>(c,0) = -1.0;
    	labels[1].at<float>(c,0) = 1.0;
    	labels[2].at<float>(c,0) = -1.0;
    	c++;
    }
    for (int i = 0; i < training_data_sizes[2]; ++i){
    	labels[0].at<float>(c,0) = -1.0;
    	labels[1].at<float>(c,0) = -1.0;
    	labels[2].at<float>(c,0) = 1.0;
    	c++;
    }
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM_cat;
    CvSVM SVM_car;
    CvSVM SVM_sheep;
    SVM_car.train(training_matrix, labels[0], cv::Mat(), cv::Mat(), params);
    SVM_cat.train(training_matrix, labels[1], cv::Mat(), cv::Mat(), params);
    SVM_sheep.train(training_matrix, labels[2], cv::Mat(), cv::Mat(), params);

	std::vector<std::string> validation_data_files { 
		"imagenes2/BOVW_descriptors/val_car",
		"imagenes2/BOVW_descriptors/val_cat",
		"imagenes2/BOVW_descriptors/val_sheep"
	};

	std::vector<float> validation_data;
	std::vector<std::string> validation_filenames;
	count = 0;
	for (int i = 0; i < 3; ++i){
		std::ifstream infile(validation_data_files[i]);
		
		std::string line;
		std::vector<std::string> vectorAsString;
		while (std::getline(infile, line)){
			if (Utils::stringEndsWith(line, "jpg")){
				validation_filenames.push_back(line);
				continue;		
			}
			vectorAsString.clear();
			vectorAsString = Utils::split(line, ',');
			for(int j = 0; j < dimension; j++){
				validation_data.push_back(std::stof(vectorAsString[j]));
			}
			count++;
		}
		infile.close();
	}

	std::cout << "Read " << count << " BOVW descriptors from validation files" << std::endl;

	cv::Mat validation_matrix = cv::Mat(count, dimension, CV_32FC1);
	memcpy(validation_matrix.data, validation_data.data(), validation_data.size()*sizeof(float));

	cv::Mat results(validation_matrix.rows, 1, CV_32F);
	SVM_car.predict(validation_matrix, results);


	std::string label;
	float value;
	for (int i = 0; i < count; ++i){
		float max_value = 1000.0;
		cv::Mat example = validation_matrix.row(i);
		
		value = SVM_car.predict(example, true);
		if(value < max_value){
			max_value = value;
			label = "car";
		}
		value = SVM_cat.predict(example, true);
		if(value < max_value){
			max_value = value;
			label = "cat";
		}
		value = SVM_sheep.predict(example, true);
		if(value < max_value){
			max_value = value;
			label = "sheep";
		}
		std::cout << validation_filenames[i] << " -- " << label << std::endl;
	}


    


}