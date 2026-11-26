cmake_minimum_required(VERSION 3.20)
project(AimbotEnterprise LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
    message(FATAL_ERROR "This project targets Windows (DXGI + Win32 API)")
endif()

# ── Compiler settings ──
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

if (MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc)
endif()

# ── OpenCV ──
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ── System libs ──
set(SYSTEM_LIBS
    shcore.lib
    winmm.lib
    d3d11.lib
    dxgi.lib
    user32.lib
    gdi32.lib
)

# ── Target ──
add_executable(aimbot src/main.cpp)

target_include_directories(aimbot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(aimbot PRIVATE
    ${OpenCV_LIBS}
    ${SYSTEM_LIBS}
)

# ── Copy runtime files next to the exe ──
add_custom_command(TARGET aimbot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:aimbot>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config/config.ini"
        "$<TARGET_FILE_DIR:aimbot>/config.ini"
    COMMENT "Copying config.ini to output directory"
)

# If user already has best.onnx in models/, copy it too
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/best.onnx")
    add_custom_command(TARGET aimbot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/models/best.onnx"
            "$<TARGET_FILE_DIR:aimbot>/best.onnx"
        COMMENT "Copying best.onnx to output directory"
    )
endif()

# ── Optional test target ──
option(AIMBOT_RUN_TESTS_IN_RELEASE "Run self-tests even in Release builds" OFF)
if (AIMBOT_RUN_TESTS_IN_RELEASE)
    target_compile_definitions(aimbot PRIVATE AIMBOT_RUN_TESTS)
endif()
