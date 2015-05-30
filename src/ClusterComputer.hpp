#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

class ClusterComputer{
public:
	ClusterComputer(int k);
	~ClusterComputer();
	void compute(cv::Mat &descriptors, cv::Mat &labels, cv::Mat &centers);
private:
	int nClusters;

};