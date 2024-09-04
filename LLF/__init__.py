import cv2
import numpy as np

#image input version
@staticmethod
#static void buildLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& destPyramid, const int level)
def buildLaplacianPyramid(SRC, destPyramid, LEVEL):

	if destPyramid.size() != LEVEL + 1:
		destPyramid.resize(LEVEL + 1)

	cv2.buildPyramid(SRC, destPyramid, LEVEL);
	for i in range(LEVEL):
		temp
		cv2.pyrUp(destPyramid[i + 1], temp, destPyramid[i].size());
		cv2.subtract(destPyramid[i], temp, destPyramid[i]);

#pyramid input version
@staticmethod
#static void buildLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int level)
def buildLaplacianPyramid():
