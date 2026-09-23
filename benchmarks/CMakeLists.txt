set(${PROJECT_NAME}_BENCHMARK ur5 soft_derivatives)

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
