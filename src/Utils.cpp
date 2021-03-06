#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/dir.h>
#include <stdexcept>
#include <fcntl.h>
#include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/nonfree/features2d.hpp>

class Utils{
public:
	static std::vector<std::string> split(std::string string, const char delim) {
		std::vector<std::string> elems;
		std::stringstream ss(string);
		std::string item;
		while (std::getline(ss, item, delim)) {
			if (!item.empty())
				elems.push_back(item);
		}
		return elems;
	}

	static bool existsFile(std::string filename) {
		struct stat st;
		if (stat(filename.c_str(), &st) == 0)
			return S_ISREG(st.st_mode) ? true : false;
		return false;
	}

	static std::vector<std::string> getAllElementsInDir(std::string path) {
		std::vector<std::string> list;
	#if _WIN32
		DIR *dp = opendir(path.c_str());
		if (dp == NULL)
			throw std::runtime_error("can't open " + path);
		struct dirent *dir_entry;
		while ((dir_entry = readdir(dp)) != NULL) {
			char *name = dir_entry->d_name;
			list.push_back(std::string(name));
		}
		closedir(dp);
	#elif __linux
		struct dirent **namelist = NULL;
		int len = scandir(path.c_str(), &namelist, NULL, alphasort);
		if (len < 0)
			throw std::runtime_error("can't open " + path);
		for (int i = 0; i < len; ++i) {
			char *name = namelist[i]->d_name;
			list.push_back(std::string(name));
			free(namelist[i]);
		}
		free(namelist);
	#endif
		return list;
	}
	
	static std::vector<std::string> getAllFilenames(std::string path) {
		std::vector<std::string> list = getAllElementsInDir(path);
		std::vector<std::string> file_list;
		for (std::string fname : list) {
			if (fname.compare(0, 1, ".") == 0) //archivos que comienzan por "."
				continue;
			std::string full_name = path + "/" + fname;
			if (existsFile(full_name))
				file_list.push_back(full_name);
		}
		return file_list;
	}

	static void printParams( cv::Algorithm* algorithm ) {
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

	static std::string type2str(int type) {
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

	static std::string vectorToString(std::vector<float> &vector){
		int size = vector.size();
		int i;

		std::stringstream ss;

		for(i = 0; i < size - 1; i++){
		//put arbitrary formatted data into the stream
			ss << vector[i] << ",";
		}
		ss << vector[size - 1];// << std::endl;
		//convert the stream buffer into a string
		std::string str = ss.str();
		return str;
	}

	static std::vector<std::string> getAllFilenames(std::vector<std::string> paths){
		std::vector<std::string> files;
		std::vector<std::string> aux;
		
		for(std::string path : paths){
			aux = getAllFilenames(path);
			files.insert(files.end(), aux.begin(), aux.end());
		}
		return files;
	}

	static cv::Mat joinMats(std::vector<cv::Mat> &mats){
		// printf("aaaaaaaaa\n");
		std::cout << "Intentando unir matrices" << std::endl;
		
		cv::Mat container;
		int nMats = mats.size();
		std::cout << "Uniendo " << nMats << " matrices" << std::endl;
		std::cout << "Mat[0] tiene " << mats[0].rows << " filas" << std::endl;
		
		for (int a = 0; a < nMats; ++a){
			// std::cout << "Matriz " << a << std::endl;
			container.push_back(mats[a]);
		}
		return container;
	}

	static std::string matToString(cv::Mat &matrix){
		int size = matrix.rows;
		int dimension = matrix.cols;
		int i;

		std::stringstream ss;

		for(i = 0; i < size - 1; i++){
		//put arbitrary formatted data into the stream
			for (int j = 0; j < dimension; ++j){
				ss << matrix.at<float>(i, j) << ",";
			}
			ss << matrix.at<float>(i, dimension - 1) << std::endl;
		}
		//convert the stream buffer into a string
		std::string str = ss.str();
		return str;
	}

	static std::vector<float> vectorFromString(std::string s){
		std::vector<std::string> vectorAsString;
		vectorAsString = split(s, ',');
		int size = vectorAsString.size();
		std::vector<float> v(size);
		int i;
		for(i = 0; i < size; i++){
			v[i] = std::stof(vectorAsString[i]);
		}
		return v;
	}

	static bool stringEndsWith (std::string const &fullString, std::string const &ending) {
	    if (fullString.length() >= ending.length()) {
	        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	    } else {
	        return false;
	    }
	}

};