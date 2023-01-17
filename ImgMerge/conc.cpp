
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << std::endl;
        return 1;
    }
    // Читать изображение
    Mat img_1big = imread(argv[1], IMREAD_COLOR), img_1;
    Mat img_2big = imread(argv[2], IMREAD_COLOR), img_2;

    // let's downscale the image using new  width and height
    int down_width = 640;
    int down_height = 360;
    //resize down
    resize(img_1big, img_1, Size(down_width, down_height), INTER_LINEAR);
    resize(img_2big, img_2, Size(down_width, down_height), INTER_LINEAR);

    // Инициализация
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Шаг 1: Определение ориентированного углового положения FAST
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Шаг 2: Рассчитать дескриптор BRIEF в соответствии с положением особой точки
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1 = {};
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB", outimg1);

    // Шаг 3: Сопоставьте дескрипторы BRIEF на двух изображениях, используя расстояние Хэмминга.
    vector<DMatch> matches = {};

    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, matches);
    cout << descriptors_1.size() << "==" << descriptors_2.size() << endl;
    // Шаг 4: Проверка пары точек совпадения
    double min_dist = 10000, max_dist = 0;

    // Найдите минимальное и максимальное расстояния между всеми совпадениями, 
    // то есть расстояние между наиболее похожими и наименее похожими двумя наборами точек
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // Когда расстояние между дескрипторами более чем в два раза превышает минимальное расстояние, 
    // считается, что сопоставление неправильное, но иногда минимальное расстояние будет очень маленьким, 
    // и в качестве нижнего предела устанавливается эмпирическое значение 30.
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //--Шаг 5: Нарисуйте совпадающие результаты
    //Mat img_match;
    Mat img_goodmatch = {};
    //drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    //imshow("full", img_match);
    imshow("good", img_goodmatch);

    //
    //create for each point a new Keypoint with size 1:

    vector<Point2f> kps1 = {}, kps2 = {};
    cout << keypoints_1.size() << " == " << keypoints_2.size() << "===" << good_matches.size() << '\n';
    for (size_t i = 0; i < good_matches.size(); i++) {
        kps1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        kps2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    //for (size_t i = 0; i < keypoints_2.size(); i++) {
    //    kps2.push_back(keypoints_2[i].pt);
    //}

    if (kps1.size() < 4 || kps2.size() < 4) {
        return 22;
    }

    Mat H = findHomography(kps2, kps1, RANSAC, 3), comboImage = {};
    cout << "===" << H << endl << H.at<double>(0, 0) << endl;
    
    //warpPerspective(img_2, comboImage, H, Size(img_1.cols + img_2.cols, img_1.rows));
    //warpPerspective(img_1, comboImage, H, Size(img_1.cols + img_2.cols, img_1.rows));
    
    warpPerspective(img_2, comboImage, H, Size(img_1.cols + img_2.cols, img_2.rows));
    // 
    //HOW?
    Vec3b bl = { 0, 0, 0 };
    for (int i = 0; i < img_1.rows; i++) {
        for (int j = 0; j < img_1.cols; j++) {
            //comboImage[i][j] = img_1[i][j];
            //if (comboImage.at<Vec3b>(Point(j, i)) == bl) {
                comboImage.at<Vec3b>(Point(j, i)) = img_1.at<Vec3b>(Point(j, i));
            //} else {
            //    comboImage.at<Vec3b>(Point(j + img_1.cols, i)) = comboImage.at<Vec3b>(Point(j, i));
            //    comboImage.at<Vec3b>(Point(j, i)) = img_1.at<Vec3b>(Point(j, i));
            //}
        }
    }

    imshow("result", comboImage);
    waitKey(0);

    return 0;
}

