cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_opperations)

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

# Define our library target
add_library(custom_opperations SHARED op.cpp)
# Enable C++14
target_compile_features(custom_opperations PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(custom_opperations "${TORCH_LIBRARIES}")
# Link against OpenCV
# target_link_libraries(custom_opperations opencv_core opencv_imgproc)