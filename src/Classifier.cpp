#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "Utils.cpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv){
	//////////////////////////////////////////////

	std::string training_data_directory = argv[2];

	std::vector<std::string> training_data_files (3);
	training_data_files[0] = training_data_directory + "/train_car";
	training_data_files[1] = training_data_directory + "/train_cat";
	training_data_files[2] = training_data_directory + "/train_bird";

	std::vector<int> training_data_sizes(3);

	int dimension = std::stoi(argv[1]);
	int count = 0;
	int max_files = 160;


	std::vector<float> training_data;
	for (int i = 0; i < 3; ++i){
		count = 0;
		std::ifstream infile(training_data_files[i]);
		
		std::string line;
		std::vector<std::string> vectorAsString;
		while (std::getline(infile, line)){
			if (Utils::stringEndsWith(line, "jpg")){
				// std::getline(infile, line);
				if(count >= max_files){
					break;
				}
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
    
	std::vector<std::string> validation_data_files (3);
	validation_data_files[0] = training_data_directory + "/val_car";
	validation_data_files[1] = training_data_directory + "/val_cat";
	validation_data_files[2] = training_data_directory + "/val_bird";

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

	std::vector<double> c_values {0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.7, 1, 5, 10};
	std::vector<double> gamma_values {0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.7, 1, 5};
	float best_c = c_values[0];
	float best_gamma = gamma_values[0];

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    CvSVM SVM_cat;
    CvSVM SVM_car;
    CvSVM SVM_bird;
    int best_result = 0;
	int total = count;
	for (double c : c_values){
		for (double gamma : gamma_values){
			params.gamma = gamma;
			params.C = c;
		    // Train the SVM with the correspondind c and gamma values
		    SVM_car.train(training_matrix, labels[0], cv::Mat(), cv::Mat(), params);
		    SVM_cat.train(training_matrix, labels[1], cv::Mat(), cv::Mat(), params);
		    SVM_bird.train(training_matrix, labels[2], cv::Mat(), cv::Mat(), params);
			
			std::string predicted_label;
			std::string correct_label;
			int correct = 0;
			float value;
			for (int i = 0; i < count; ++i){
				float max_value = 1000.0;
				cv::Mat example = validation_matrix.row(i);
				
				value = SVM_car.predict(example, true);
				if(value < max_value){
					max_value = value;
					predicted_label = "car";
				}
				value = SVM_cat.predict(example, true);
				if(value < max_value){
					max_value = value;
					predicted_label = "cat";
				}
				value = SVM_bird.predict(example, true);
				if(value < max_value){
					max_value = value;
					predicted_label = "bird";
				}
				if(validation_filenames[i].find("car") != -1){
					correct_label = "car";
				}
				else if (validation_filenames[i].find("cat") != -1){
					correct_label = "cat";
				}
				else{
					correct_label = "bird";
				}
				if(correct_label == predicted_label){
					correct++;
				}
			}
			if(correct > best_result){
				best_c = c;
				best_gamma = gamma;
				best_result = correct;
			}

		}
	}

	std::cout << "Best C: " << best_c << std::endl;
	std::cout << "Best Gamma: " << best_gamma << std::endl;
	std::cout << "Accuracy: " << best_result/(float)total << std::endl;

	std::vector<std::string> test_data_files (3);
	test_data_files[0] = training_data_directory + "/test_car";
	test_data_files[1] = training_data_directory + "/test_cat";
	test_data_files[2] = training_data_directory + "/test_bird";

	std::vector<float> test_data;
	std::vector<std::string> test_filenames;
	count = 0;
	for (int i = 0; i < 3; ++i){
		std::ifstream infile(test_data_files[i]);
		
		std::string line;
		std::vector<std::string> vectorAsString;
		while (std::getline(infile, line)){
			if (Utils::stringEndsWith(line, "jpg")){
				test_filenames.push_back(line);
				continue;		
			}
			vectorAsString.clear();
			vectorAsString = Utils::split(line, ',');
			for(int j = 0; j < dimension; j++){
				test_data.push_back(std::stof(vectorAsString[j]));
			}
			count++;
		}
		infile.close();
	}

	std::cout << "Read " << count << " BOVW descriptors from test files" << std::endl;
	cv::Mat test_matrix = cv::Mat(count, dimension, CV_32FC1);
	memcpy(test_matrix.data, test_data.data(), test_data.size()*sizeof(float));

	total = count;

	std::string output_name = argv[3];
	std::ofstream outputFile;
	outputFile.open(output_name);

	params.gamma = best_gamma;
	params.C = best_c;
    // Train the SVM with the best c and gamma values
    SVM_car.train(training_matrix, labels[0], cv::Mat(), cv::Mat(), params);
    SVM_cat.train(training_matrix, labels[1], cv::Mat(), cv::Mat(), params);
    SVM_bird.train(training_matrix, labels[2], cv::Mat(), cv::Mat(), params);
	
	std::vector<std::string> class_labels{"car", "cat", "bird"};
	int predicted_label;
	int correct_label;
	int correct = 0;
	std::vector<std::vector<int>> confusion_matrix(3);
	confusion_matrix[0] = *(new std::vector<int> (3, 0));
	confusion_matrix[1] = *(new std::vector<int> (3, 0));
	confusion_matrix[2] = *(new std::vector<int> (3, 0));

	float value;
	for (int i = 0; i < total; ++i){
		float max_value = 1000.0;
		cv::Mat example = test_matrix.row(i);
		
		value = SVM_car.predict(example, true);
		if(value < max_value){
			max_value = value;
			predicted_label = 0;
		}
		value = SVM_cat.predict(example, true);
		if(value < max_value){
			max_value = value;
			predicted_label = 1;
		}
		value = SVM_bird.predict(example, true);
		if(value < max_value){
			max_value = value;
			predicted_label = 2;
		}
		if(test_filenames[i].find("car") != -1){
			correct_label = 0;
		}
		else if (test_filenames[i].find("cat") != -1){
			correct_label = 1;
		}
		else{
			correct_label = 2;
		}
		if(correct_label == predicted_label){
			correct++;
		}
		confusion_matrix[correct_label][predicted_label]++;

		outputFile << test_filenames[i] << " -- " << class_labels[predicted_label] << std::endl;
 	}

	outputFile << "Accuracy: " << correct/(float)total << std::endl;
	std::cout << "Test Accuracy: " << correct/(float)total << std::endl;
	outputFile << "Confusion matrix: " << std::endl;
	outputFile << "\tcar\tcat\tbird" << std::endl;
	outputFile << "car\t" << confusion_matrix[0][0] << "\t" << confusion_matrix[0][1] << "\t" << confusion_matrix[0][2] << "\t" << std::endl;
	outputFile << "cat\t" << confusion_matrix[1][0] << "\t" << confusion_matrix[1][1] << "\t" << confusion_matrix[1][2] << "\t" << std::endl;
	outputFile << "bird\t" << confusion_matrix[2][0] << "\t" << confusion_matrix[2][1] << "\t" << confusion_matrix[2][2] << "\t" << std::endl;
	
	
	
	
	outputFile.close();
    


}
