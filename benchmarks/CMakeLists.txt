# example-robot-data is otherwise only looked up when BUILD_TESTING is ON, which
# happens after this directory is added: find it here so that the benchmarks that
# need it (ur5 uses EXAMPLE_ROBOT_DATA_MODEL_DIR) can be built on their own.
find_package(example-robot-data 4.0.9 QUIET)

set(${PROJECT_NAME}_BENCHMARK soft_derivatives)
if(TARGET example-robot-data::example-robot-data)
    list(APPEND ${PROJECT_NAME}_BENCHMARK ur5)
else()
    message(
        STATUS
        "example-robot-data not found: skipping the ur5 benchmark"
    )
endif()

set(${PROJECT_NAME}_BENCHMARK_TOOLS timings.hpp timings.cpp)

add_library(
    ${PROJECT_NAME}_BENCHMARK_TOOLS
    SHARED
    ${${PROJECT_NAME}_BENCHMARK_TOOLS}
)

foreach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_NAME}.cpp)
    target_link_libraries(
        ${BENCHMARK_NAME}
        ${PROJECT_NAME}
        ${PROJECT_NAME}_BENCHMARK_TOOLS
    )
    # only some benchmarks need example-robot-data (ur5); link it when available
    if(TARGET example-robot-data::example-robot-data)
        target_link_libraries(
            ${BENCHMARK_NAME}
            example-robot-data::example-robot-data
        )
    endif()
    add_custom_target(
        "benchmarks-cpp-${BENCHMARK_NAME}"
        ${BENCHMARK_NAME} \${INPUT}
    )
endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
