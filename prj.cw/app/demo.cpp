#include "color_correction/color_correction.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void help() {
    std::cout << "\nThis program allows you to change the brightness and contrast of images using two algorithms: autocontrast and greyworld.";
    std::cout << "\nUsage:\n demo <path-to-image> autocontrast -cut=0.5";
    std::cout << "\nor \n demo <path-to-image> greyworld";
}

void getHist(cv::Mat& gray, cv::Mat& hist) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
}

void getLumHist(cv::Mat& img, cv::Mat& hist) {
    cv::Mat gray;
    if (img.type() == CV_8UC1)
        gray = img;
    else if (img.type() == CV_8UC3)
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.type() == CV_8UC4)
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    getHist(gray, hist);
}

void getRGBHist(cv::Mat& img, cv::Mat& b_hist, cv::Mat& g_hist, cv::Mat& r_hist) {
    std::vector<cv::Mat> bgr_planes;
    split(img, bgr_planes);
    getHist(bgr_planes[0], b_hist);
    getHist(bgr_planes[1], g_hist);
    getHist(bgr_planes[2], r_hist);
}

void getLumHistImg(cv::Mat& hist, cv::Mat& histImg) {
    int histSize = 256;
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    histImg = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(hist, hist, 0, hist.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 0; i < histSize; i++) {
        line(histImg, cv::Point(bin_w * (i + 1 / 2), hist_h),
            cv::Point(bin_w * (i + 1 / 2), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(128, 128, 128), bin_w, 8, 0);
    }
}

void getRGBHistImg(cv::Mat& b_hist, cv::Mat& g_hist, cv::Mat& r_hist, cv::Mat& histImg) {
    int histSize = 256;
    int hist_w = 512, hist_h = 600;
    int bin_w = cvRound((double)hist_w / histSize);
    histImg = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImg.rows / 3, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(g_hist, g_hist, 0, histImg.rows / 3, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(r_hist, r_hist, 0, histImg.rows / 3, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 0; i < histSize; i++) {
        line(histImg, cv::Point(bin_w * (i + 1 / 2), hist_h),
            cv::Point(bin_w * (i + 1 / 2), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), bin_w, 8, 0);
        line(histImg, cv::Point(bin_w * (i + 1 / 2), hist_h * 2 / 3),
            cv::Point(bin_w * (i + 1 / 2), hist_h * 2 / 3 - cvRound(g_hist.at<float>(i))),
            cv::Scalar(0, 255, 0), bin_w, 8, 0);
        line(histImg, cv::Point(bin_w * (i + 1 / 2), hist_h / 3),
            cv::Point(bin_w * (i + 1 / 2), hist_h / 3 - cvRound(r_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), bin_w, 8, 0);
    }
}

void saveHist(cv::Mat& hist, std::string color, std::string name) {
    std::string path = std::filesystem::current_path().string() + "\\out\\" + name + ".tex";
    std::ofstream file(path);

    file << "\\documentclass{standalone}" << std::endl;
    file << "\\usepackage{pgfplots}" << std::endl;
    file << "\\begin{document}" << std::endl;
    file << "\\begin{tikzpicture}" << std::endl;
    file << "\\begin{axis}" << std::endl;

    file << "[height = 5cm, width = 7cm," << std::endl;
    file << "ybar, bar width = 0.5pt," << std::endl;
    file << "xmin = 0, xmax = 256," << std::endl;
    file << "enlarge y limits = { 0.1, upper }," << std::endl;
    file << "tick style = { draw = none }," << std::endl;
    file << "yticklabel = \\empty," << std::endl;
    file << "xtick distance = 256," << std::endl;
    file << "xticklabels = { 0, 0, 256 }]" << std::endl;

    file << "\\addplot[color=" << color << "] coordinates {";
    int histSize = 256;
    for (int i = 0; i < histSize; ++i) {
        file << "(" << i << "," << hist.at<float>(i) << ") ";
    }
    file << "};" << std::endl;

    file << "\\end{axis}" << std::endl;
    file << "\\end{tikzpicture}" << std::endl;
    file << "\\end{document}" << std::endl;

    file.close();

    std::cout << "Hisgoram saved to " << path << std::endl;
}

void saveImg(cv::Mat& img, std::string name) {
    std::string path = std::filesystem::current_path().string() + "\\out\\" + name + ".png";
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    bool result = false;
    result = cv::imwrite(path, img);
    if (result) {
        std::cout << "Resulting image saved to " << path << '\n';
    }
    else {
        std::cout << "Failed to save resulting image.\n";
    }
}

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    std::filesystem::create_directory("out");

    const cv::String keys = {
        "{help h usage ? |      | help message }"
        "{@input         |      | input image }"
        "{@type          |      | algorithm to use: autocontrast or greyworld }"
        "{cut            |0     | the percent to cut off from the image histogram on the low and high ends (optional for autocontrast) }"
    };

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || parser.has("h") || parser.has("usage")) {
        help();
        return 0;
    }
    cv::Mat srcImg;
    cv::Mat srcHist;
    cv::Mat dstImg;
    cv::Mat dstHist;
    cv::String imgPath = parser.get<cv::String>("@input");
    srcImg = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (srcImg.empty()){
        std::cout << "Can't open an image.\n";
        help();
        return 0;
    }

    std::string name = imgPath.substr(imgPath.find_last_of("/\\") + 1);
    std::string::size_type const p(name.find_last_of('.'));
    name = name.substr(0, p);

    if (parser.get<cv::String>("@type") == "autocontrast") {
        float cut = parser.get<float>("cut");
        cc::autoContrast(srcImg, dstImg, cut);

        cv::Mat hist;

        getLumHist(srcImg, hist);
        getLumHistImg(hist, srcHist);
        saveHist(hist, "black", name + "-lum-hist");

        getLumHist(dstImg, hist);
        getLumHistImg(hist, dstHist);
        saveHist(hist, "black", name + "-dst-lum-hist");
    }
    else if (parser.get<cv::String>("@type") == "greyworld") {
        cc::greyWorld(srcImg, dstImg);

        cv::Mat b_hist, g_hist, r_hist;

        getRGBHist(srcImg, b_hist, g_hist, r_hist);
        getRGBHistImg(b_hist, g_hist, r_hist, srcHist);
        saveHist(b_hist, "blue", name + "-blue-hist");
        saveHist(g_hist, "green", name + "-green-hist");
        saveHist(r_hist, "red", name + "-red-hist");

        getRGBHist(dstImg, b_hist, g_hist, r_hist);
        getRGBHistImg(b_hist, g_hist, r_hist, dstHist);
        saveHist(b_hist, "blue", name + "-dst-blue-hist");
        saveHist(g_hist, "green", name + "-dst-green-hist");
        saveHist(r_hist, "red", name + "-dst-red-hist");
    }
    else {
        std::cout << "Possibly, you have not specified which algorithm to apply or you have misspelled its name.\n";
        help();
        return 0;
    }

    saveImg(dstImg, name + "(dst)");

    cv::imshow("source image", srcImg);
    cv::imshow("source image histogram", srcHist);
    cv::imshow("destionation image", dstImg);
    cv::imshow("destionation image histogram", dstHist);
    std::cout << "Press any key to finish";
    cv::waitKey();
    
}