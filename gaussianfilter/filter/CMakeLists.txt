add_library(filter
        src/filter.cpp
        include/filter/filter.hpp
        )

set_property(TARGET filter PROPERTY CXX_STANDARD 20)

target_include_directories(filter
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

target_link_libraries(filter PRIVATE ${OpenCV_LIBS} ctikz)

install(TARGETS filter)




