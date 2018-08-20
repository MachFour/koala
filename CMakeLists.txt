cmake_minimum_required(VERSION 3.11)
project(reference)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_INSTALL_PREFIX /home/max/uni/thesis/experiments/compiled)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -ffast-math")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -pg")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -Og")

find_package(OpenCV REQUIRED)
#find_package(Tesseract REQUIRED)
add_executable(reference main.cpp meanshift.cpp meanshift.h reference.h Interval.h Interval.cpp utils.cpp wordBB.h randomColour.h randomColour.cpp)

#https://stackoverflow.com/questions/20382549/cmake-for-tesseract-and-opencv#20392329
target_link_libraries(reference lept tesseract ${OpenCV_LIBS})

# add the install targets
install (TARGETS reference DESTINATION .)
