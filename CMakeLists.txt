cmake_minimum_required(VERSION 3.11)
project(koala)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_INSTALL_PREFIX /home/max/uni/thesis/experiments/compiled)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -ffast-math")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -pg")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -Og")

find_package(OpenCV REQUIRED core highgui imgproc imgcodecs plot ml)
#find_package(Tesseract REQUIRED)
add_executable(koala
        main.cpp meanshift.cpp Interval.cpp drawingutils.cpp randomColour.cpp plotutils.cpp tableExtract.cpp ocrutils.cpp InputParser.cpp table.cpp
        meanshift.h tableExtract.h Interval.h wordBB.h randomColour.h plotutils.h table.h drawingutils.h ocrutils.h levenshtein.h InputParser.h
        tableComparison.cpp tableComparison.h matutils.cpp matutils.h wordBB.cpp IntervalGroup.tpp preprocessing.cpp preprocessing.h fileutils.cpp fileutils.h)


#https://stackoverflow.com/questions/20382549/cmake-for-tesseract-and-opencv#20392329
target_link_libraries(koala lept tesseract ${OpenCV_LIBS})

# add the install targets
install (TARGETS koala DESTINATION .)
