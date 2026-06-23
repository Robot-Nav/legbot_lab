cmake_minimum_required(VERSION 3.12)
project(vbot_sim2real_tests)

# ─── Google Test ────────────────────────────────────────────────────────────
# On Ubuntu: sudo apt install libgtest-dev
# If gtest is not found, try building from source or using FetchContent

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found via find_package, trying FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# ─── Test target: motor protocol ────────────────────────────────────────────

add_executable(test_motor_protocol
    test_motor_protocol.cpp
)
target_include_directories(test_motor_protocol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_motor_protocol PRIVATE GTest::gtest GTest::gtest_main)
target_compile_features(test_motor_protocol PRIVATE cxx_std_17)

# ─── Test target: IMU protocol ──────────────────────────────────────────────

add_executable(test_imu_protocol
    test_imu_protocol.cpp
)
target_include_directories(test_imu_protocol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_imu_protocol PRIVATE GTest::gtest GTest::gtest_main)
target_compile_features(test_imu_protocol PRIVATE cxx_std_17)

# ─── Register with CTest ────────────────────────────────────────────────────

enable_testing()
add_test(NAME motor_protocol  COMMAND test_motor_protocol)
add_test(NAME imu_protocol    COMMAND test_imu_protocol)
