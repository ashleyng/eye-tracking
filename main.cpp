#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "constants.h"

using namespace std;
using namespace cv;

bool calibration_done = false;
Rect screen;
Mat screen_image;


void scale(const Mat &src,Mat &dst) {
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

Point unscale_point(Point p, Rect origSize) {
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
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


/*
 * Find possible center in gradient location
 * doesn't use the postprocessing weight of color (section 2.1)
 */
void possible_centers(int x, int y, const Mat &blurred, double gx, double gy, Mat &output) {

    for (int cy = 0; cy < output.rows; cy++) {
        double *output_row = output.ptr<double>(cy);
        for (int cx = 0; cx < output.cols; cx++) {
            if (x == cx && y == cy) {
                continue;
            }
            // equation (2)
            double dx = x - cx;
            double dy = y - cy;
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;

            double dotProduct = (dx*gx + dy*gy);
            // ignores vectors pointing in opposite direction/negative dot products
            dotProduct = max(0.0, dotProduct);

            // summation
            output_row[cx] += dotProduct * dotProduct;
        }
    }
}

/*
 * Finds the pupils within the given eye region
 * returns points of where pupil is calculated to be
 *
 * face_image: image of face region from frame
 * eye_region: dimensions of eye region
 * window_name: display window name
 */
Point find_centers(Mat face_image, Rect eye_region, string window_name) {
    Mat eye_unscaled = face_image(eye_region);

    // scale and grey image
    Mat eye_scaled_gray;
    scale(eye_unscaled, eye_scaled_gray);
    cvtColor(eye_scaled_gray, eye_scaled_gray, COLOR_BGRA2GRAY);

    // get the gradient of eye regions
    Mat gradient_x, gradient_y;
    Sobel(eye_scaled_gray, gradient_x, CV_64F, 1, 0, 5);
    Sobel(eye_scaled_gray, gradient_y, CV_64F, 0, 1, 5);
    Mat magnitude = matrix_magnitude(gradient_x, gradient_y);

    // normalized displacement vectors
    normalize(gradient_x, gradient_x);
    normalize(gradient_y, gradient_y);

    // blur and invert the image
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

    Point max_point;
    double max_value;
    minMaxLoc(out, NULL, &max_value, NULL, &max_point);
    Point pupil = unscale_point(max_point, eye_region);
    return pupil;
}

/*
 * returns an array of points of the pupils
 * [left pupil, right pupil]
 *
 * color_image: image of the whole frame
 * face: dimensions of face in color_image
 */
void find_eyes(Mat color_image, Rect face, Point &left_pupil_dst, Point &right_pupil_dst) {
    // image of face
    Mat face_image = color_image(face);

    int eye_width = face.width * (kEyePercentWidth/100.0);
    int eye_height = face.height * (kEyePercentHeight/100.0);
    int eye_top = face.height * (kEyePercentTop/100.0);
    int eye_side = face.width * (kEyePercentSide/100.0);
    int right_eye_x = face.width - eye_width -  eye_side;

    // eye regions
    Rect left_eye_region(eye_side, eye_top, eye_width, eye_height);
    Rect right_eye_region(right_eye_x, eye_top, eye_width, eye_height);

    // draw eye regions
    rectangle(face_image, left_eye_region, Scalar(0, 0, 255));
    rectangle(face_image, right_eye_region, Scalar(0, 0, 255));

    // get points of pupils within eye region
    Point left_pupil = find_centers(face_image, left_eye_region, "left eye");
    Point right_pupil = find_centers(face_image, right_eye_region, "right eye");

    // convert points to fit on frame image
    right_pupil.x += right_eye_region.x;
    right_pupil.y += right_eye_region.y;
    left_pupil.x += left_eye_region.x;
    left_pupil.y += left_eye_region.y;

    Point center;
    center.x = (right_pupil.x - left_pupil.x)/2 + left_pupil.x;
    center.y = (right_pupil.y + left_pupil.y)/2;

    // draw pupils
    circle(face_image, right_pupil, 3, Scalar(0, 255, 0));
    circle(face_image, left_pupil, 3, Scalar(0, 255, 0));
    circle(face_image, center, 3, Scalar(255, 0, 0));

    left_pupil_dst = left_pupil;
    right_pupil_dst = right_pupil;

    imshow("window", color_image);
}


int main() {

    CascadeClassifier face_cascade;
    face_cascade.load("haar_data/haarcascade_frontalface_alt_tree.xml");
    screen_image = imread("screen_test.png");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    namedWindow("window");
    Mat frame;
    cap >> frame;
    int count = 0;
    while (1) {
        Mat gray_image;
        vector<Rect> faces;
        cvtColor(frame, gray_image, COLOR_BGRA2GRAY);
        face_cascade.detectMultiScale(gray_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT);

        Point left_pupil, right_pupil;
        if (faces.size() > 0) {
            find_eyes(frame, faces[0], left_pupil, right_pupil);
            cout << "Center:" << "(" << faces[0].width/2 << "," << faces[0].height/2 << ")" << "    " << "Rectangle:" << faces[0] << "    " << "Left pupil:" << left_pupil << "   " << "Right pupil:" << right_pupil;
            cout << "\n";
        }

        // if 'q' is tapped, exit
        int wait_key = waitKey(8);
        if (wait_key == 113) {
            break;
        }

        // if space is tap, take calibration
        else if(wait_key == 32) {
            // left screen
            if (count == 0) {
                screen.x = (right_pupil.x - left_pupil.x)/2 + left_pupil.x;
            }
            // top screen
            else if (count == 1) {
                screen.y = (right_pupil.y + left_pupil.y)/2;
            }
            // right screen
            else if (count == 2) {
                assert (((right_pupil.x - left_pupil.x)/2 + left_pupil.x) > screen.x);
                screen.width = ((right_pupil.x - left_pupil.x)/2 + left_pupil.x) - screen.x;
            }
            // bottom screen
            else if (count == 3) {
                assert (((right_pupil.y + left_pupil.y)/2) > screen.y);
                screen.height = ((right_pupil.y + left_pupil.y)/2) - screen.y;
                calibration_done = true;
            }
            count++;
        }

        cap >> frame;
    }

    return 0;
}