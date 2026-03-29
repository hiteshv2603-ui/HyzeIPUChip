cmake_minimum_required(VERSION 3.20)
project(HyzeIpuDriver CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPCI REQUIRED libpci)

# Existing PCIe driver
add_executable(hyze_ipu_driver hyze_ipu_pcie_driver.cpp)
target_link_libraries(hyze_ipu_driver PRIVATE ${LIBPCI_LIBRARIES} pthread)
target_include_directories(hyze_ipu_driver PRIVATE ${LIBPCI_INCLUDE_DIRS})
target_compile_options(hyze_ipu_driver PRIVATE ${LIBPCI_CFLAGS_OTHER})

# NEW: GGUF inference engine
add_executable(hyze_gguf_ipu hyze_gguf_ipu.cpp)
target_compile_features(hyze_gguf_ipu PRIVATE cxx_std_20)
target_compile_options(hyze_gguf_ipu PRIVATE -O2 -Wall -Wextra)
target_link_libraries(hyze_gguf_ipu PRIVATE pthread)
if(LIBPCI_FOUND)
    target_link_libraries(hyze_gguf_ipu PRIVATE ${LIBPCI_LIBRARIES})
    target_include_directories(hyze_gguf_ipu PRIVATE ${LIBPCI_INCLUDE_DIRS})
endif()

install(TARGETS hyze_ipu_driver hyze_gguf_ipu RUNTIME DESTINATION bin)
