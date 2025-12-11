set(${PROJECT_NAME}_FACTORY_TEST
  factory/crocoddyl/pinocchio_model.hpp
  factory/crocoddyl/state.hpp
  factory/crocoddyl/actuation.hpp
  factory/crocoddyl/contact.hpp
  factory/crocoddyl/activation.hpp
  factory/crocoddyl/cost.hpp
  factory/diff-action.hpp
  factory/state-lpf.hpp
  factory/action-lpf.hpp
  factory/state-soft.hpp
  factory/diff-action-soft-abstract.hpp
  factory/diff-action-soft3d.hpp
  factory/diff-action-soft1d.hpp
  # factory/diff-action-soft3d-friction.hpp
  factory/action-soft.hpp
  factory/crocoddyl/pinocchio_model.cpp
  factory/crocoddyl/state.cpp
  factory/crocoddyl/actuation.cpp
  factory/crocoddyl/contact.cpp
  factory/crocoddyl/activation.cpp
  factory/crocoddyl/cost.cpp
  factory/diff-action.cpp
  factory/state-lpf.cpp
  factory/action-lpf.cpp
  factory/state-soft.cpp
  factory/diff-action-soft-abstract.cpp
  factory/diff-action-soft3d.cpp
  factory/diff-action-soft1d.cpp
  # factory/diff-action-soft3d-friction.cpp
  factory/action-soft.cpp
  )

add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
target_link_libraries(${PROJECT_NAME}_unittest PUBLIC ${PROJECT_NAME}
  example-robot-data::example-robot-data)

add_unit_test(test_states test_states.cpp)
target_link_libraries(test_states PUBLIC ${PROJECT_NAME}_unittest)

add_unit_test(test_actuations test_actuations.cpp)
target_link_libraries(test_actuations PUBLIC ${PROJECT_NAME}_unittest)

# add_unit_test(test_diff_actions test_diff_actions.cpp)
# target_link_libraries(test_diff_actions PUBLIC ${PROJECT_NAME}_unittest)

add_unit_test(test_states_lpf test_states_lpf.cpp)
target_link_libraries(test_states_lpf PUBLIC ${PROJECT_NAME}_unittest)

add_unit_test(test_actions_lpf test_actions_lpf.cpp)
target_link_libraries(test_actions_lpf PUBLIC ${PROJECT_NAME}_unittest)

ADD_UNIT_TEST(test_states_soft test_states_soft.cpp)
target_link_libraries(test_states_soft PUBLIC ${PROJECT_NAME}_unittest)

ADD_UNIT_TEST(test_diff_actions_soft3d test_diff_actions_soft3d.cpp)
target_link_libraries(test_diff_actions_soft3d PUBLIC ${PROJECT_NAME}_unittest)

ADD_UNIT_TEST(test_diff_actions_soft1d test_diff_actions_soft1d.cpp)
target_link_libraries(test_diff_actions_soft1d PUBLIC ${PROJECT_NAME}_unittest)

ADD_UNIT_TEST(test_actions_soft test_actions_soft.cpp)
target_link_libraries(test_actions_soft PUBLIC ${PROJECT_NAME}_unittest)

# ADD_UNIT_TEST(test_diff_actions_soft3d_friction test_diff_actions_soft3d_friction.cpp)
# target_link_libraries(test_diff_actions_soft3d_friction PUBLIC ${PROJECT_NAME}_unittest)
