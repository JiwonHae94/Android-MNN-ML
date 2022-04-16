cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_library(
        MNN
        SHARED
        mnnnet-native.cpp
)

set_target_properties(
        MNN
        PROPERTIES IMPORTED_LOCATION
        ${lib_DIR}/${ANDROID_ABI}/libMNN.so
)