#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "force_feedback_mpc::force_feedback_mpc" for configuration "Release"
set_property(TARGET force_feedback_mpc::force_feedback_mpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(force_feedback_mpc::force_feedback_mpc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libforce_feedback_mpc.so.1.0"
  IMPORTED_SONAME_RELEASE "libforce_feedback_mpc.so.1.0"
  )

list(APPEND _cmake_import_check_targets force_feedback_mpc::force_feedback_mpc )
list(APPEND _cmake_import_check_files_for_force_feedback_mpc::force_feedback_mpc "${_IMPORT_PREFIX}/lib/libforce_feedback_mpc.so.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
