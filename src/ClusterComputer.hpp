#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

class ClusterComputer{
public:
	ClusterComputer();
	~ClusterComputer();
	cv::Mat compute(std::vector<cv::Mat> descriptors);
};