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
TARGET	=	w2v_train
# ---------------

all:	$(TARGET)

w2v_train: main.cpp Makefile
	$(CC) $(CXXFLAGS) $(INCLUDES) $< -o $@ -L. $(LDFLAGS)

clean:
	-rm -f *.o core.* w2v_train
