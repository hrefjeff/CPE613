#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()  {

    Mat image;
    image = imread("thethreeamigos.jpeg", IMREAD_UNCHANGED);   // Read the file
    namedWindow("cvmat", WINDOW_AUTOSIZE );// Create a window for display.
    imshow("cvmat", image );                   // Show our image inside it.

    // flatten the mat.
    uint totalElements = image.total()*image.channels(); // Note: image.total() == rows*cols.
    Mat flat = image.reshape(1, totalElements); // 1xN mat of 1 channel, O(1) operation
    if(!image.isContinuous()) {
        flat = flat.clone(); // O(N),
    }
    // flat.data is your array pointer
    auto * ptr = flat.data; // usually, its uchar*
    // You have your array, its length is flat.total() [rows=1, cols=totalElements]
    // Converting to vector
    std::vector<uchar> vec(flat.data, flat.data + flat.total());
    // Testing by reconstruction of cvMat
    Mat restored = Mat(image.rows, image.cols, image.type(), ptr); // OR vec.data() instead of ptr
    namedWindow("reconstructed", WINDOW_AUTOSIZE);
    imshow("reconstructed", restored);

    waitKey(0);

}