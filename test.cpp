#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include "fit_ellipse_batched.h"

using namespace cv;
using namespace std;

class Timer {
public:
    Timer() { reset(); }
    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        assert(!m_started);
        m_started = true;
        m_start = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        assert(m_started);
        assert(!m_stopped);
        m_stopped = true;
        m_stop = std::chrono::high_resolution_clock::now();
    }
    size_t get_time_in_ms() const {
        assert(m_stopped);
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_stop -
                                                                     m_start)
                .count();
    }

private:
    using time_point = std::chrono::high_resolution_clock::time_point;
    time_point m_start, m_stop;
    bool m_started, m_stopped;
};

int main() {

	Mat img = imread("image.png", CV_8UC1);
    threshold(img, img, 127,255,CV_THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    vector<RotatedRect> ellipses;


    {
        Timer timer;
        timer.reset();
        timer.start();
        Mat img(Size(800,500), CV_8UC3);
        for (unsigned i = 0; i<contours.size(); i++) {
            if (contours[i].size() >= 6) {
                printf("%zu\n", contours[i].size());
                while (contours[i].size() > 32)
                    contours[i].pop_back();
                RotatedRect temp = fitEllipse(Mat(contours[i]));
                drawContours(img, contours, i, Scalar(255,0,0), -1, 8);
                ellipse(img, temp, Scalar(0,255,255), 2, 8);
            }
        }
        timer.stop();
        std::cout << "time: " << timer.get_time_in_ms() << std::endl;

        imwrite("out.png", img);
    }

    {
        std::vector<std::vector<Point2f>> batched_points;

        for (int b  = 0; b < contours.size(); ++b) {
            batched_points.push_back(std::vector<Point2f>(0));
            for (int i = 0; i < min(int(contours[b].size()), 32); ++i) {
                batched_points.back().emplace_back(contours[b][i].x, contours[b][i].y);
            }
        }
        Timer timer;
        timer.reset();
        BatchedEllipseFitter fitter;
        timer.start();
        fitter.fit(batched_points);
        timer.stop();
        std::cout << "time: " << timer.get_time_in_ms() << std::endl;

    }
    return 0;
}
