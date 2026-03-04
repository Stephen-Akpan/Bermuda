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
    add_deps("bermuda_engine")
    set_group("Tests")
    
    -- Add test after build
    after_build(function (target)
        print("✓ Tests built successfully: " .. target:targetfile())
    end)

-- ============================================================================
-- Magic Number Generator (Utility)
-- ============================================================================

target("magic_gen")
    set_kind("binary")
    add_files("tools/magic_gen.cpp")
    set_group("Tools")

-- ============================================================================
-- Custom Tasks
-- ============================================================================

-- Run all tests
task("test")
    set_menu {usage = "xmake test", description = "Run all unit tests"}
    on_run(function ()
        print("Running tests...")
        local exe = path.join(get_config("buildir"), get_config("plat"), get_config("arch"), 
                             is_mode("debug") and "debug" or "release", "bermuda_tests")
        if is_host("windows") then
            exe = exe .. ".exe"
        end
        os.execv(exe)
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
