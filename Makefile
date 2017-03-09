# Some Macros
# ---------------
# Compiler Name
CC		=	g++
# Compile Flags
CXXFLAGS=	-O2 -Wall -std=c++11 -fopenmp
# Linker Flags
LDFLAGS	=
# Include 
INCLUDES=
# Libraries
LIBS	=
# Object Files
OBJS	=	main.o
# Name of Executable
TARGET	=	run_main
# ---------------

all:	$(TARGET)

run_main: main.cpp Makefile query2vec.h layers.h mat.h 
	$(CC) MurmurHash3.cpp $(CXXFLAGS) $(INCLUDES) $< -o $@ -L. $(LDFLAGS)

clean:
	-rm -f *.o core.* run_*
