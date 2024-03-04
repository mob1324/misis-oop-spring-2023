#include "color_correction/color_correction.hpp"


void cc::autoContrast(const cv::Mat& src, cv::Mat& dst, float cut = 0) {

    CV_Assert(cut >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    cv::Mat gray;
    if (src.type() == CV_8UC1)
        gray = src;
    else if (src.type() == CV_8UC3)
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4)
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int histSize = 256;
    float alpha, beta;
    double low = 0;
    double high = 0;

    if (cut == 0) {
        cv::minMaxLoc(gray, &low, &high);
    }
    else {
        cv::Mat hist;

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);

        float pixels = accumulator.back();
        low = 0;
        while (accumulator[low] < pixels * cut / 100)
            low++;

        high = histSize - 1;
        while (accumulator[high] >= (pixels * (100 - cut) / 100))
            high--;
    }

    alpha = (histSize - 1) / (high - low);
    beta = -low * alpha;

    src.convertTo(dst, -1, alpha, beta);

    if (dst.type() == CV_8UC4) {
        int from_to[] = { 3, 3 };
        cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
    }
}


void cc::greyWorld(const cv::Mat& src, cv::Mat& dst) {

    CV_Assert((src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    cv::Scalar bgr = cv::mean(src);
    float bAvg = bgr[0];
    float gAvg = bgr[1];
    float rAvg = bgr[2];
    float grey = (bAvg + gAvg + rAvg) / 3;
    cv::multiply(src, cv::Scalar(grey / bAvg, grey / gAvg, grey / rAvg), dst);    
}