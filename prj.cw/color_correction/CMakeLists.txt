project(color_correction)

add_library(${PROJECT_NAME} src/color_correction.cpp include/color_correction/color_correction.hpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

find_package(OpenCV)
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core opencv_imgproc)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT})
    add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
        DESTINATION ${CMAKE_INSTALL_PREFIX})
else ()
    message(STATUS "Couldn't find Doxygen")
endif ()