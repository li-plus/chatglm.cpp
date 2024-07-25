cmake_minimum_required(VERSION 3.12)
project(ChatGLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-expansion-to-defined")   # suppress ggml warnings

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

option(CHATGLM_ENABLE_EXAMPLES "chatglm: enable c++ examples" ON)
option(CHATGLM_ENABLE_PYBIND "chatglm: enable python binding" OFF)
option(CHATGLM_ENABLE_TESTING "chatglm: enable testing" OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
if (CHATGLM_ENABLE_PYBIND)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

# third-party libraries
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
    enable_language(CUDA)
    # ref: https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions
    set(CUDA_ARCH_LIST "52;61;70;75")
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
        set(CUDA_ARCH_LIST "${CUDA_ARCH_LIST};80")
    endif ()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
        set(CUDA_ARCH_LIST "${CUDA_ARCH_LIST};86")
    endif ()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        set(CUDA_ARCH_LIST "${CUDA_ARCH_LIST};89;90")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} CACHE STRING "")
endif ()

include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "chatglm: disable sentencepiece shared libraries by default")
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "chatglm: disable tcmalloc by default")
include_directories(third_party/sentencepiece/src)
add_subdirectory(third_party/sentencepiece)

include_directories(third_party/sentencepiece/third_party/protobuf-lite)

set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/abseil-cpp)

add_subdirectory(third_party/re2)

include_directories(third_party/stb)

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
    configure_file(third_party/ggml/src/ggml-metal.metal ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)
endif ()

if (GGML_PERF)
    add_compile_definitions(GGML_PERF)
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/tests/*.cpp)

add_library(chatglm STATIC chatglm.cpp)
target_link_libraries(chatglm PUBLIC ggml sentencepiece-static re2)

# c++ examples
if (CHATGLM_ENABLE_EXAMPLES)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE chatglm)

    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        set(CHATGLM_OPENMP_TARGET OpenMP::OpenMP_CXX)
    endif ()
    add_executable(perplexity tests/perplexity.cpp)
    target_link_libraries(perplexity PRIVATE chatglm ${CHATGLM_OPENMP_TARGET})
endif ()

# GoogleTest
if (CHATGLM_ENABLE_TESTING)
    enable_testing()

    # ref: https://github.com/google/googletest/blob/main/googletest/README.md
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Specify the commit you depend on and update it regularly.
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    # Now simply link against gtest or gtest_main as needed. Eg
    add_executable(chatglm_test chatglm_test.cpp)
    target_link_libraries(chatglm_test PRIVATE chatglm gtest_main)
    gtest_discover_tests(chatglm_test)
endif ()

if (CHATGLM_ENABLE_PYBIND)
    add_subdirectory(third_party/pybind11)
    pybind11_add_module(_C chatglm_pybind.cpp)
    target_link_libraries(_C PRIVATE chatglm)
endif ()

# lint
file(GLOB PY_SOURCES
    ${PROJECT_SOURCE_DIR}/chatglm_cpp/*.py
    ${PROJECT_SOURCE_DIR}/examples/*.py
    ${PROJECT_SOURCE_DIR}/tests/*.py
    ${PROJECT_SOURCE_DIR}/convert.py
    ${PROJECT_SOURCE_DIR}/setup.py)
add_custom_target(lint
    COMMAND clang-format -i ${CPP_SOURCES}
    COMMAND isort ${PY_SOURCES}
    COMMAND black ${PY_SOURCES} --verbose)

# check all
add_custom_target(check-all
    COMMAND cmake --build build -j
    COMMAND ./build/bin/chatglm_test
    COMMAND python3 setup.py develop
    COMMAND python3 -m pytest --forked tests/test_chatglm_cpp.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# mypy
add_custom_target(mypy
    mypy chatglm_cpp examples --exclude __init__.pyi
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# stub
add_custom_target(stub
    pybind11-stubgen chatglm_cpp -o .
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
endif ()
