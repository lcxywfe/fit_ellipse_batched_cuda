#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::RotatedRect> fit_ellipse_batched(
        std::vector<std::vector<cv::Point2f>> _batched_points);
