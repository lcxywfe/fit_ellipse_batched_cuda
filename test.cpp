#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>

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

	Mat img = imread("a.png", CV_8UC1);
    threshold(img, img, 127,255,CV_THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    vector<RotatedRect> ellipses;

    Timer timer;
            timer.reset();
            timer.start();
    {
    Mat img(Size(800,500), CV_8UC3);
    for (unsigned i = 0; i<contours.size(); i++) {
        if (contours[i].size() >= 6) {
            printf("%zu\n", contours[i].size());
            RotatedRect temp = fitEllipse(Mat(contours[i]));

            // if (area(temp) <= 1.1 * contourArea(contours[i])) {
                // ellipses.push_back(temp);
                drawContours(img, contours, i, Scalar(255,0,0), -1, 8);
                ellipse(img, temp, Scalar(0,255,255), 2, 8);

                // imshow("Ellipses", img);
                // waitKey();
            // } else {
            //     //cout << "Reject ellipse " << i << endl;
            //     drawContours(img, contours, i, Scalar(0,255,0), -1, 8);
            //     ellipse(img, temp, Scalar(255,255,0), 2, 8);
            //     imshow("Ellipses", img);
            //     waitKey();
            // }
        }
    }
    timer.stop();
    std::cout << "time: " << timer.get_time_in_ms() << std::endl;

    imwrite("out.png", img);
    }
    return 0;
}
