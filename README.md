# Image Filtering Library

This cross-platform C++ library provides image filtering based on the recursive Gauss-Deriche filter and the accurate Gauss algorithm. The library also includes a console application for visualizing the algorithm's operation.

## User Description

In the command line, the user inputs the following parameters separated by spaces:

1. Path to the black-and-white image.
2. Algorithm type: "gaussian" for the accurate Gauss filter or "deriche" for the recursive Gauss-Deriche filter.
3. Sigma parameter (numeric value).
4. Visualization parameter ("yes" or "no").
5. Path to the folder to save the resulting image (e.g., C://test/output_data1.jpg).

## How to Use

1. Clone the repository.
2. Build the library and console application.
3. Run the console application from the command line, providing the required parameters.


## Dependencies

- OpenCV 
