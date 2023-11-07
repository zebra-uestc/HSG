add_executable(main main.cpp)

set_target_properties(
    main
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_PATH}
)