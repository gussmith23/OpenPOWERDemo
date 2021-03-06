project (OpenPOWERDemo)

cmake_minimum_required (VERSION 2.6)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_definitions ("-Wall")

include_directories(${OpenCV_INCLUDE_DIRS})

if(WIN32)
add_executable (openPOWERDemo openPOWERDemo.cpp cmt/common.cpp cmt/gui.cpp 
    cmt/CMT.cpp cmt/Consensus.cpp cmt/Fusion.cpp cmt/Matcher.cpp cmt/Tracker.cpp
    cmt/fastcluster/fastcluster.cpp cmt/getopt/getopt.cpp
    )
else()
add_executable (openPOWERDemo openPOWERDemo.cpp cmt/common.cpp cmt/gui.cpp 
    cmt/CMT.cpp cmt/Consensus.cpp cmt/Fusion.cpp cmt/Matcher.cpp cmt/Tracker.cpp
    cmt/fastcluster/fastcluster.cpp)
endif()

target_link_libraries(openPOWERDemo ${OpenCV_LIBS})

# copy needed libs to output dir on build
add_custom_command(TARGET openPOWERDemo PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/thirdparty $<TARGET_FILE_DIR:openPOWERDemo>)

