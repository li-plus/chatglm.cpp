cmake_minimum_required(VERSION 3.12)
project(ChatGLM.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
add_compile_definitions(GGML_CUDA_MMV_Y=2)  # for large vocab
include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "chatglm: disable sentencepiece shared libraries by default")
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "chatglm: disable tcmalloc by default")
include_directories(third_party/sentencepiece/src)
add_subdirectory(third_party/sentencepiece)

option(GGML_HIPBLAS "chatglm: use hipBLAS" OFF)
option(GGML_CUDA_FORCE_DMMV "chatglm: use dmmv instead of mmvq CUDA kernels" OFF)
option(GGML_CUDA_FORCE_MMQ "chatglm: use mmq kernels instead of cuBLAS" OFF)

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
    set(CUDA_ARCHITECTURES "52;61;70;75;80;86" CACHE STRING "chatglm: cuda architectures to compile")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endif ()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
    configure_file(third_party/ggml/src/ggml-metal.metal ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)
endif ()

if (GGML_PERF)
    add_compile_definitions(GGML_PERF)
endif ()

file(GLOB CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")

add_library(chatglm STATIC chatglm.cpp)
target_link_libraries(chatglm PUBLIC ggml sentencepiece-static)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE chatglm)

if (GGML_HIPBLAS)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUBLAS)
        if (BUILD_SHARED_LIBS)
            set_target_properties(chatglm PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        if (GGML_CUDA_FORCE_DMMV)
            target_compile_definitions(chatglm PRIVATE GGML_CUDA_FORCE_DMMV)
        endif()
        if (GGML_CUDA_FORCE_MMQ)
            target_compile_definitions(chatglm PRIVATE GGML_CUDA_FORCE_MMQ)
        endif()
        target_compile_definitions(chatglm PRIVATE GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        target_compile_definitions(chatglm PRIVATE GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        target_compile_definitions(chatglm PRIVATE K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        #set_source_files_properties(ggml-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(chatglm PRIVATE hip::device PUBLIC hip::host roc::rocblas roc::hipblas)

        if (GGML_STATIC)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        endif()
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
endif ()

# GoogleTest
option(CHATGLM_ENABLE_TESTING "chatglm: enable testing" OFF)
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

option(CHATGLM_ENABLE_PYBIND "chatglm: enable python binding" OFF)
if (CHATGLM_ENABLE_PYBIND)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(chatglm ggml sentencepiece-static PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
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
    COMMAND black ${PY_SOURCES} --line-length 120)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
endif ()
