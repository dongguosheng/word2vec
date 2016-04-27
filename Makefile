# Some Macros
# ---------------
# Compiler Name
CC		=	g++
# Compile Flags
CXXFLAGS=	-g -Wall -std=c++11 -fopenmp
# Linker Flags
LDFLAGS	=
# Include 
INCLUDES=
# Libraries
LIBS	=
# Object Files
OBJS	=	main.o 
# Name of Executable
TARGET	=	$(patsubst %.cpp, run_%, $(wildcard *.cpp))
# ---------------

all:	$(TARGET)

run_%: %.cpp Makefile
	$(CC) $(CXXFLAGS) $(INCLUDES) $< -o $@ -L. $(LDFLAGS)

clean:
	-rm -f *.o core.* run_*
