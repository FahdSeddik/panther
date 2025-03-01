cmake_minimum_required(VERSION 3.15...3.27)
project(my_project) # Replace 'my_project' with the name of your project

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT Python_FOUND)
  message(FATAL_ERROR "Python 3.12 not found")
endif()

# Use the found Python library and include directories
set(Python_LIBRARY ${Python_LIBRARIES})
set(Python_INCLUDE_DIR ${Python_INCLUDE_DIRS})

# Print the paths for verification
message(STATUS "Python Library: ${Python_LIBRARY}")
message(STATUS "Python Include Directory: ${Python_INCLUDE_DIR}")

# Ensure the paths are correctly set
foreach(lib ${Python_LIBRARY})
  if (lib MATCHES "optimized|debug")
    continue()
  endif()
  if (NOT EXISTS ${lib})
    message(FATAL_ERROR "Python library not found at ${lib}")
  endif()
endforeach()

if (NOT EXISTS ${Python_INCLUDE_DIR})
  message(FATAL_ERROR "Python include directory not found at ${Python_INCLUDE_DIR}")
endif()

# Add the Python include directory to the include path
include_directories(${Python_INCLUDE_DIR})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

# Add the nanobind module
nanobind_add_module(my_ext my_ext.cpp)

# Link the Python library
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_libraries(my_ext PRIVATE ${Python_LIBRARY_DEBUG})
else()
  target_link_libraries(my_ext PRIVATE ${Python_LIBRARY_RELEASE})
endif()