cmake_minimum_required(VERSION 4.1)
project(EcoTorch)

if(APPLE)
    set(CMAKE_CXX_STANDARD 17)
    set(PACKAGE_NAME "apple")

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
            ${PACKAGE_NAME}
            NB_STATIC
            src/ecotorch/watcher/cpp/bindings.cpp
    )

    find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)

    execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Active SDK Path: ${MACOS_SDK_PATH}")

    target_link_libraries(${PACKAGE_NAME} PRIVATE
            ${IOKIT_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
            IOReport
    )

    target_compile_options(${PACKAGE_NAME} PRIVATE -fblocks)

    install(TARGETS ${PACKAGE_NAME} DESTINATION ecotorch/watcher)
endif()