-- Bermuda Chess Engine Build Configuration
-- XMake build system with xrepo package management

set_project("Bermuda")
set_version("0.1.0")
set_description("A UCI chess engine with competitive Stockfish aspirations")

add_rules("mode.debug", "mode.release")

-- C++ Standard
set_languages("cxx17")

-- Optimization flags based on mode
if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
else
    set_optimize("aggressive")
    set_symbols("hidden")
end

-- Platform-specific flags
if is_plat("windows") then
    add_cxflags("/W4", "/permissive-", {tools = {"msvc"}})
    add_defines("_CRT_SECURE_NO_WARNINGS")
elseif is_plat("macosx", "linux") then
    add_cxflags("-Wall", "-Wextra", "-Wpedantic")
    add_cxxflags("-march=native")
end

-- ============================================================================
-- Engine Library
-- ============================================================================

target("bermuda_engine")
    set_kind("static")
    add_files("engine/**.cpp")
    add_headerfiles("engine/**.h")
    add_includedirs("engine")
    set_group("Core")

-- ===========================================================================
-- Engine Binary
-- ===========================================================================
target("bermuda")
    set_kind("binary")
    add_files("binary/**.cpp")
    add_includedirs("engine")
    add_deps("bermuda_engine")
    set_group("Core")

-- ============================================================================
-- Tests
-- ============================================================================

target("bermuda_tests")
    set_kind("binary")
    add_files("tests/test_*.cpp", "tests/main.cpp", "tests/catch_amalgamated.cpp")
    add_headerfiles("tests/*.hpp")
    add_includedirs("engine")
    add_deps("bermuda_engine", "bermuda_tuner_lib")
    set_group("Tests")

    -- Add test after build
    after_build(function (target)
        local output = target:targetfile() or target:targetdir() or "<unknown>"
        print("✓ Tests built successfully: " .. tostring(output))
    end)

-- ============================================================================
-- Magic Number Generator (Utility)
-- ============================================================================

target("magic_gen")
    set_kind("binary")
    add_files("tools/magic_gen.cpp")
    set_group("Tools")

target("bermuda_tuner_lib")
    set_kind("static")
    add_files("tools/texel_tuner/src/**.cpp")
    add_headerfiles("tools/texel_tuner/include/**.h")
    add_includedirs("tools/texel_tuner/include", {public = true})
    add_includedirs("engine", {public = true})
    add_deps("bermuda_engine")
    set_group("Tools")

target("bermuda_tuner_tests")
    set_kind("binary")
    add_files("tests/test_tuner.cpp", "tests/main.cpp", "tests/catch_amalgamated.cpp")
    add_includedirs("tests")
    add_deps("bermuda_tuner_lib")
    set_group("Tests")

target("bermuda_tuner")
    set_kind("binary")
    add_files("tools/texel_tuner.cpp")
    add_deps("bermuda_tuner_lib")
    set_group("Tools")

-- ============================================================================
-- Custom Tasks
-- ============================================================================

-- Run all tests
task("test")
    set_menu {usage = "xmake test", description = "Run all unit tests"}
    on_run(function ()
        print("Running tests...")
        local buildir = get_config("buildir") or "build"
        local plat = get_config("plat") or os.host()
        local arch = get_config("arch") or os.arch()
        local mode_name = is_mode("debug") and "debug" or "release"
        local suffix = is_host("windows") and ".exe" or ""
        local tests = {
            {
                target = "bermuda_tests",
                path = path.join(buildir, plat, arch, mode_name, "bermuda_tests" .. suffix)
            },
            {
                target = "bermuda_tuner_tests",
                path = path.join(buildir, plat, arch, mode_name, "bermuda_tuner_tests" .. suffix)
            },
        }
        for _, test in ipairs(tests) do
            if not os.isfile(test.path) then
                print("Missing test binary: " .. test.path)
                print("Building required test targets...")
                local result = os.execv("xmake", {"build", test.target})
                if result ~= 0 or not os.isfile(test.path) then
                    raise("Failed to build test target: " .. test.target)
                end
            end
        end
        for _, test in ipairs(tests) do
            local result = os.execv(test.path)
            if result ~= 0 then
                raise("Test executable failed: " .. test.path)
            end
        end
    end)

-- Build documentation
task("docs")
    set_menu {usage = "xmake docs", description = "Build documentation"}
    on_run(function ()
        print("Building documentation...")
        print("(Add doxygen integration here later)")
    end)

-- ============================================================================
-- Install Rules
-- ============================================================================

-- Installation paths (optional)
-- Uncomment to enable installation
-- target("bermuda_engine")
--     add_installfiles("engine/(*.h)", {prefixdir = "include"})
-- target("bermuda_tests")
--     add_installfiles("build/$(plat)/$(arch)/$(mode)/bermuda_tests", {prefixdir = "bin"})
