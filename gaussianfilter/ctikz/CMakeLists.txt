add_library(ctikz ctikz.cpp include/ctikz/ctikz.hpp)
set_property(TARGET ctikz PROPERTY CXX_STANDARD 20)
target_include_directories(ctikz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS ctikz)