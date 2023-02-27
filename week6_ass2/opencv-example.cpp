#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // Read the input image as a gray-scale image
    Mat inputImage = imread("3graygoats.jpg", IMREAD_GRAYSCALE);

    // Convert the input image to a vector
    vector<unsigned char> inputVector(inputImage.data, inputImage.data + inputImage.total());

    // Create a new Mat object from the vector data
    Mat outputImage;
    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);
    copy(inputVector.begin(), inputVector.end(), outputImage.data);

    // Display the original and reconstructed images
    imshow("Original", inputImage);
    imshow("Reconstructed", outputImage);
    waitKey(0);

    return 0;
}