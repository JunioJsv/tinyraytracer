set(RLIMGUI_DIR ${CMAKE_SOURCE_DIR}/external/rlImGui)
set(IMGUI_DIR   ${CMAKE_SOURCE_DIR}/external/imgui)

add_library(rlImGui STATIC
        ${RLIMGUI_DIR}/rlImGui.cpp

        ${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
)

target_include_directories(rlImGui PUBLIC
        ${RLIMGUI_DIR}
        ${RLIMGUI_DIR}/extras
        ${IMGUI_DIR}
)

target_compile_features(rlImGui PUBLIC cxx_std_17)

target_compile_definitions(rlImGui PUBLIC
        IMGUI_DISABLE_OBSOLETE_FUNCTIONS
        IMGUI_DISABLE_OBSOLETE_KEYIO
)

target_link_libraries(rlImGui PUBLIC raylib)