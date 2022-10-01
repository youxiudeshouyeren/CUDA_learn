# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sjx/code/CUDA_learn/MyCode/gemm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sjx/code/CUDA_learn/MyCode/gemm/build

# Include any dependencies generated for this target.
include CMakeFiles/first_attempt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/first_attempt.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/first_attempt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/first_attempt.dir/flags.make

CMakeFiles/first_attempt.dir/first_attempt.cu.o: CMakeFiles/first_attempt.dir/flags.make
CMakeFiles/first_attempt.dir/first_attempt.cu.o: ../first_attempt.cu
CMakeFiles/first_attempt.dir/first_attempt.cu.o: CMakeFiles/first_attempt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sjx/code/CUDA_learn/MyCode/gemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/first_attempt.dir/first_attempt.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sjx/code/CUDA_learn/MyCode/gemm/first_attempt.cu -o CMakeFiles/first_attempt.dir/first_attempt.cu.o
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sjx/code/CUDA_learn/MyCode/gemm/first_attempt.cu -MT CMakeFiles/first_attempt.dir/first_attempt.cu.o -o CMakeFiles/first_attempt.dir/first_attempt.cu.o.d

CMakeFiles/first_attempt.dir/first_attempt.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/first_attempt.dir/first_attempt.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/first_attempt.dir/first_attempt.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/first_attempt.dir/first_attempt.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target first_attempt
first_attempt_OBJECTS = \
"CMakeFiles/first_attempt.dir/first_attempt.cu.o"

# External object files for target first_attempt
first_attempt_EXTERNAL_OBJECTS =

first_attempt: CMakeFiles/first_attempt.dir/first_attempt.cu.o
first_attempt: CMakeFiles/first_attempt.dir/build.make
first_attempt: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
first_attempt: /usr/lib/x86_64-linux-gnu/libpthread.so
first_attempt: CMakeFiles/first_attempt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sjx/code/CUDA_learn/MyCode/gemm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable first_attempt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/first_attempt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/first_attempt.dir/build: first_attempt
.PHONY : CMakeFiles/first_attempt.dir/build

CMakeFiles/first_attempt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/first_attempt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/first_attempt.dir/clean

CMakeFiles/first_attempt.dir/depend:
	cd /home/sjx/code/CUDA_learn/MyCode/gemm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sjx/code/CUDA_learn/MyCode/gemm /home/sjx/code/CUDA_learn/MyCode/gemm /home/sjx/code/CUDA_learn/MyCode/gemm/build /home/sjx/code/CUDA_learn/MyCode/gemm/build /home/sjx/code/CUDA_learn/MyCode/gemm/build/CMakeFiles/first_attempt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/first_attempt.dir/depend
