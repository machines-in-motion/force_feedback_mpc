# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Define the Python sub-packages
set(PYTHON_PACKAGES
    lpf_mpc
    soft_mpc
    core_mpc_utils
)

# Install the __init__.py files and modules for each sub-package
foreach(PACKAGE ${PYTHON_PACKAGES})
    install(FILES ${PACKAGE}/__init__.py
            DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/${PACKAGE})
    
    file(GLOB MODULES "${PACKAGE}/*.py" "${PACKAGE}/*.yml")
    
    foreach(MODULE ${MODULES})
        get_filename_component(MODULE_NAME ${MODULE} NAME)
        install(FILES ${MODULE}
                DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/${PACKAGE})
        
        # add_custom_target(
        #     "${MODULE_NAME}"
        #     ${CMAKE_COMMAND}
        #     -E
        #     env
        #     PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
        #     ${Python3_EXECUTABLE}
        #     "${PROJECT_SOURCE_DIR}/python/${PACKAGE}/${MODULE_NAME}"
        #     \${INPUT})
    endforeach()
endforeach()

