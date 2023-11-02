set(${PROJECT_NAME}_FACTORY_TEST
  factory/pinocchio_model.hpp
  factory/random_generator.hpp
  factory/state.hpp
  factory/state-lpf.hpp
  # factory/action-lpf.hpp
  # factory/statesoft.hpp
  # factory/lpf.hpp
  # factory/diff-action-soft-abstract.hpp
  # factory/diff-action-soft3d.hpp
  # factory/diff-action-soft3d-friction.hpp
  # factory/diff-action-soft1d.hpp
  # factory/action-soft.hpp
  factory/pinocchio_model.cpp
  factory/state.cpp
  factory/state-lpf.cpp
  # factory/action-lpf.cpp
  # factory/statesoft.cpp
  # factory/diff-action-soft-abstract.cpp
  # factory/diff-action-soft3d.cpp
  # factory/diff-action-soft3d-friction.cpp
  # factory/diff-action-soft1d.cpp
  # factory/action-soft.cpp
  )

add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
target_link_libraries(
  ${PROJECT_NAME}_unittest PUBLIC ${PROJECT_NAME}
                                  example-robot-data::example-robot-data)

                                  add_unit_test(test_states test_states.cpp)
target_link_libraries(test_states PUBLIC ${PROJECT_NAME}_unittest)

add_unit_test(test_states_lpf test_states_lpf.cpp)
target_link_libraries(test_states_lpf PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_statessoft test_statessoft.cpp)
# target_link_libraries(test_statessoft PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_lpf test_lpf.cpp)
# target_link_libraries(test_lpf PUBLIC ${PROJECT_NAME}_unittest)

# add_unit_test(test_diff_actions test_diff_actions.cpp)
# target_link_libraries(test_diff_actions PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_diff_actions_soft3d test_diff_actions_soft3d.cpp)
# target_link_libraries(test_diff_actions_soft3d PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_diff_actions_soft3d_friction test_diff_actions_soft3d_friction.cpp)
# target_link_libraries(test_diff_actions_soft3d_friction PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_diff_actions_soft1d test_diff_actions_soft1d.cpp)
# target_link_libraries(test_diff_actions_soft1d PUBLIC ${PROJECT_NAME}_unittest)


if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
