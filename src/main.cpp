#include <opencv2/opencv.hpp>
#include <glob.h>
#include <string.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <opencv2/calib3d.hpp>

std::vector<std::string> glob(const std::string &pattern)
{
    using namespace std;

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0)
    {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    globfree(&glob_result);

    return filenames;
}
void image_resize(cv::InputArray image, cv::OutputArray output, int width = 0, int height = 0, int inter = cv::INTER_AREA)
{
    int h = image.size().height;
    int w = image.size().width;
    cv::Size dim;

    if (width == 0 && height == 0)
        return;

    if (width == 0)
    {
        float r = height / float(h);
        dim = cv::Size(int(w * r), height);
    }
    else
    {
        float r = width / float(w);
        dim = cv::Size(width, int(h * r));
    }
    cv::resize(image, output, dim, 0.0, 0.0, inter);
}
int main(int argc, char **argv)
{
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    auto criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    // prepare object points, like(0, 0, 0), (1, 0, 0), (2, 0, 0)...., (6, 5, 0)
    cv::Mat objp = cv::Mat::zeros(6 * 7, 3, CV_32F);
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            objp.at<cv::Vec3f>(i * 7 + j)[0] = float(j);
            objp.at<cv::Vec3f>(i * 7 + j)[1] = float(i);
        }
    }

    // Arrays to store object points and image points from all the images.
    std::vector<cv::Mat> objpoints;
    std::vector<cv::Mat> imgpoints;

    cv::Mat images;
    cv::Size shape;
    std::vector<std::string> filenames = glob("phone1_17p5mm/*.jpg");
    // std::vector<std::string> filenames = glob("example/*.jpg");
    for (int i = 0; i < filenames.size(); i++)
    {
        cv::Mat img;
        cv::Mat full_img = cv::imread(filenames[i]);
        image_resize(full_img, img, 400);
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        shape = gray.size();

        // Find the chess board corners
        cv::Mat corners;
        bool ret = cv::findChessboardCorners(gray, cv::Size(7, 6), corners);

        // If found, add object points, image points(after refining them)
        if (ret)
        {
            std::cout << shape << std::endl;
            objpoints.push_back(objp);
            // cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            imgpoints.push_back(corners);

            // Draw and display the corners
            cv::drawChessboardCorners(img, cv::Size(7, 6), corners, ret);
            cv::imshow("img", img);
            cv::waitKey(500);
        }
    }
    cv::destroyAllWindows();
    cv::Mat mtx, dist, rvecs, tvecs;
    double ret = cv::calibrateCamera(objpoints, imgpoints, shape, mtx, dist, rvecs, tvecs);
    cv::FileStorage fs("calibration_data.yml", cv::FileStorage::WRITE);
    fs << "mtx" << mtx;
    fs << "dist" << dist;
    fs.release();
    // std::cout << "ret: " << ret << std::endl;
    // std::cout << "mtx: " << mtx << std::endl;
    // std::cout << "dist: " << dist << std::endl;
    // std::cout << "rvecs: " << rvecs << std::endl;
    // std::cout << "tvecs: " << tvecs << std::endl;
    // cv::Mat distorted = cv::imread("calib_radial.jpg");
    cv::Mat full_img = cv::imread("distorted.jpg");
    cv::Mat distorted;
    image_resize(full_img, distorted, 400);
    // double scale = 1300 / 300;

    // Adjust the camera matrix for the new resolution
    // cv::Mat new_mtx = mtx.clone();
    // new_mtx.at<double>(0) *= scale;
    // new_mtx.at<double>(1) *= scale;
    // new_mtx.at<double>(0, 2) *= scale_x;
    // new_mtx.at<double>(1, 2) *= scale_y;
    int h = distorted.size().height;
    int w = distorted.size().width;
    cv::Rect roi;
    cv::Mat newcameramtx = cv::getOptimalNewCameraMatrix(mtx, dist, cv::Size(w, h), 1, cv::Size(w, h), &roi);
    cv::Mat dst;
    cv::undistort(distorted, dst, mtx, dist, newcameramtx);
    cv::imwrite("calibresult.jpg", dst(roi));
    return 0;
}