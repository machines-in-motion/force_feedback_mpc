# CMake generated Testfile for 
# Source directory: /home/skleff/force_feedback_ws/force_feedback_mpc/tests
# Build directory: /home/skleff/force_feedback_ws/force_feedback_mpc/build_debug/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_diff_actions "/test_diff_actions")
set_tests_properties(test_diff_actions PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;45;add_unit_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_states_lpf "/test_states_lpf")
set_tests_properties(test_states_lpf PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;48;add_unit_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_actions_lpf "/test_actions_lpf")
set_tests_properties(test_actions_lpf PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;51;add_unit_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_states_soft "/test_states_soft")
set_tests_properties(test_states_soft PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;54;ADD_UNIT_TEST;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_diff_actions_soft3d "/test_diff_actions_soft3d")
set_tests_properties(test_diff_actions_soft3d PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;57;ADD_UNIT_TEST;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_diff_actions_soft1d "/test_diff_actions_soft1d")
set_tests_properties(test_diff_actions_soft1d PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;60;ADD_UNIT_TEST;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
add_test(test_actions_soft "/test_actions_soft")
set_tests_properties(test_actions_soft PROPERTIES  _BACKTRACE_TRIPLES "/home/skleff/force_feedback_ws/force_feedback_mpc/cmake/test.cmake;101;add_test;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;63;ADD_UNIT_TEST;/home/skleff/force_feedback_ws/force_feedback_mpc/tests/CMakeLists.txt;0;")
subdirs("python")
