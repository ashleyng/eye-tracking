#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

const int kSmoothFaceFactor = 0.005;
const int kEyePercentWidth = 30;
const int kEyePercentHeight = 25;
const int kEyePercentTop = 25;
const int kEyePercentSide = 15;
const int kScaledEyeWidth = 50;
const double kGradientThreshold = 50.0;
const int kFastEyeWidth = 50;



void scale(const cv::Mat &src,cv::Mat &dst) {
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

Mat matrix_magnitude(Mat mat_x, Mat mat_y) {
    Mat mag(mat_x.rows, mat_x.cols, CV_64F);

    for (int y = 0; y < mat_x.rows; y++) {
        const double *x_row = mat_x.ptr<double>(y), *y_row = mat_y.ptr<double>(y);
        double *mag_row = mag.ptr<double>(y);
        for (int x = 0; x < mat_x.cols; x++) {
            double gx = x_row[x], gy = y_row[x];
            double magnitude = sqrt((gx * gx) + (gy * gy));
            mag_row[x] = magnitude;
        }
    }
    return mag;
}
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}

void possible_centers(int x, int y, const Mat &blurred, double gx, double gy, Mat &output) {

    for (int cy = 0; cy < output.rows; cy++) {
        double *output_row = output.ptr<double>(cy);
        const unsigned  char *blur_row = blurred.ptr<unsigned char>(cy);
        for (int cx = 0; cx < output.cols; cx++) {
            if (x == cx && y == cy) {
                continue;
            }
            double dx = x - cx;
            double dy = y - cy;
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = max(0.0, dotProduct);

            output_row[cx] += dotProduct * dotProduct;
        }
    }
}

Point unscalePoint(cv::Point p, cv::Rect origSize) {
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
}

Point find_centers(Mat face_image, Rect eye_region, string window_name) {
    Mat eye_unscaled = face_image(eye_region);
    Mat eye_scaled_gray;
    scale(eye_unscaled, eye_scaled_gray);
    cvtColor(eye_scaled_gray, eye_scaled_gray, COLOR_BGRA2GRAY);
    Mat gradient_x, gradient_y;
    Sobel(eye_scaled_gray, gradient_x, CV_64F, 1, 0, 5);
    Sobel(eye_scaled_gray, gradient_y, CV_64F, 0, 1, 5);
    Mat magnitude = matrix_magnitude(gradient_x, gradient_y);
    double gradientThresh = computeDynamicThreshold(magnitude, kGradientThreshold);

    for (int y = 0; y < eye_scaled_gray.rows; y++) {
        double *x_row = gradient_x.ptr<double>(y), *y_row = gradient_y.ptr<double>(y);
        const double *mag_row = magnitude.ptr<double>(y);
        for (int x = 0; x < eye_scaled_gray.cols; x++) {
            double gx = x_row[x], gy = y_row[x];
            double mag = mag_row[x];
            if (mag > gradientThresh) {
                x_row[x] = gx/mag;
                y_row[x] = gy/mag;
            }
            else {
                x_row[x] = 0.0;
                y_row[x] = 0.0;
            }
        }
    }

    Mat blurred;
    GaussianBlur(eye_scaled_gray, blurred, Size(5, 5), 0, 0);
    bitwise_not(blurred, blurred);

    Mat outSum = Mat::zeros(eye_scaled_gray.rows, eye_scaled_gray.cols, CV_64F);

    for (int y = 0; y < blurred.rows; y++) {
        const double *x_row = gradient_x.ptr<double>(y), *y_row = gradient_y.ptr<double>(y);
        for (int x = 0; x < blurred.cols; x++) {
            double gx = x_row[x], gy = y_row[x];
            if (gx == 0.0 && gy == 0.0) {
                continue;
            }
            possible_centers(x, y, blurred, gx, gy, outSum);
        }
    }

    double numGradients = (blurred.rows*blurred.cols);
    Mat out;
    outSum.convertTo(out, CV_32F, 1.0/numGradients);

    Point maxP;
    double maxVal;
    minMaxLoc(out, NULL, &maxVal, NULL, &maxP);
    Point pupil = unscalePoint(maxP, eye_region);
    return pupil;
}


void find_eyes(Mat color_image, Rect face) {
    Mat face_image = color_image(face);

    int eye_width = face.width * (kEyePercentWidth/100.0);
    int eye_height = face.height * (kEyePercentHeight/100.0);
    int eye_top = face.height * (kEyePercentTop/100.0);
    int eye_side = face.width * (kEyePercentSide/100.0);
    int right_eye_x = face.width - eye_width -  eye_side;

    // left eye
    Rect left_eye_region(eye_side, eye_top, eye_width, eye_height);
    Rect right_eye_region(right_eye_x, eye_top, eye_width, eye_height);

    rectangle(face_image, left_eye_region, Scalar(0, 0, 255));
    rectangle(face_image, right_eye_region, Scalar(0, 0, 255));

    Point left_pupil = find_centers(face_image, left_eye_region, "left eye");
    Point right_pupil = find_centers(face_image, right_eye_region, "right eye");

    right_pupil.x += right_eye_region.x;
    right_pupil.y += right_eye_region.y;
    left_pupil.x += left_eye_region.x;
    left_pupil.y += left_eye_region.y;

    circle(face_image, right_pupil, 3, Scalar(0, 255, 0));
    circle(face_image, left_pupil, 3, Scalar(0, 255, 0));

    imshow("window", color_image);

}


int main() {

    CascadeClassifier face_cascade;
    face_cascade.load("haar_data/haarcascade_frontalface_alt.xml");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    namedWindow("window");
    Mat frame;
    cap >> frame;
    while (1) {
        Mat gray_image;
        vector<Rect> faces;
        cvtColor(frame, gray_image, COLOR_BGRA2GRAY);
        face_cascade.detectMultiScale(gray_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT);

        for (int x = 0; x < faces.size(); x++) {
            rectangle(frame, faces[0], 1234);
        }

        if (faces.size() > 0) {
            find_eyes(frame, faces[0]);
        }

        if (waitKey(5) == 113) {
            break;
        }
        cap >> frame;
    }

    return 0;
}