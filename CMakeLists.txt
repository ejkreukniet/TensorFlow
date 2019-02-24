cmake_minimum_required(VERSION 3.13)
project(TensorFlow)

set(CMAKE_CXX_STANDARD 14)

add_executable(TensorFlow
        assignment.py
        house-votes-84.data.txt
        kera.py
        mammographic_masses.data.txt
        mammographic_masses.names.txt
        pca.py)
