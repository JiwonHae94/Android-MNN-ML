project(android-mnn-ml)
cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
find_library( android-lib  android )
find_library( log-lib log )

find_library( jnigraphics-lib jnigraphics )

add_library(MNN SHARED IMPORTED )
set_target_properties(
        MNN
        PROPERTIES IMPORTED_LOCATION
        include/MNN/libs/${ANDROID_ABI}/libMNN.so
)

add_library(
        mnn-core
        SHARED
        mnnnet-native.cpp
)

target_link_libraries(
        mnn-core
        MNN
        ${jnigraphics-lib}
        ${log-lib}
        ${android-lib}
)