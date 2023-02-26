#include <opencv2/opencv.hpp>
#include <vector>

int main()
{
    // Read the input image as a color image
    cv::Mat inputImage = cv::imread("thethreeamigos.jpeg", cv::IMREAD_COLOR);

    // Extract the red, green, and blue channels
    cv::Mat bgr_channels[3];
    cv::split(inputImage, bgr_channels);

    // Convert each channel to a vector
    std::vector<unsigned char> blueVector(bgr_channels[0].data, bgr_channels[0].data + bgr_channels[0].total());
    std::vector<unsigned char> greenVector(bgr_channels[1].data, bgr_channels[1].data + bgr_channels[1].total());
    std::vector<unsigned char> redVector(bgr_channels[2].data, bgr_channels[2].data + bgr_channels[2].total());

    // Create new Mat objects for each channel using the size of the original image
    cv::Mat blueMat, greenMat, redMat;
    blueMat.create(inputImage.rows, inputImage.cols, CV_8UC1);
    greenMat.create(inputImage.rows, inputImage.cols, CV_8UC1);
    redMat.create(inputImage.rows, inputImage.cols, CV_8UC1);

    // Copy the data from the vectors to the Mat objects
    std::copy(blueVector.begin(), blueVector.end(), blueMat.data);
    std::copy(greenVector.begin(), greenVector.end(), greenMat.data);
    std::copy(redVector.begin(), redVector.end(), redMat.data);

    // Merge the channels to reconstruct the original image
    cv::Mat outputImage;
    cv::merge(std::vector<cv::Mat>{blueMat, greenMat, redMat}, outputImage);

    // Display the original and reconstructed images
    cv::imshow("Original", inputImage);
    cv::imshow("Reconstructed", outputImage);
    cv::waitKey(0);

    return 0;
}
