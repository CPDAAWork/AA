################################################################################
# Makefile for SEARCH
################################################################################
SHELL = /bin/sh
CC = gcc



CFLAGS = -Wall -Wextra -std=c99 -O2 -march=native

ifeq ($(OMP),yes)
	CFLAGS += -DOMP -fopenmp
endif

ifeq ($(VEC),yes)
	CFLAGS += -DVEC -ftree-vectorize -fopt-info-vec
endif

MORE = src/more/matrix.c src/more/papi_aux.c

DEPS_DOT_MATRIX = src/dot_matrix.c
EXEC_DOT_MATRIX = bin/dot_matrix.out

BUILD_DIR = bin

PAPIH = /share/apps/papi/5.5.0/include/

PAPILIB = /share/apps/papi/5.5.0/lib/


.DEFAULT_GOAL = all

compdotmatrix: $(DEPS_DOT_MATRIX) $(MORE)
	$(CC) $(CFLAGS) -iquote $(PAPIH) -L $(PAPILIB) $(DEPS_DOT_MATRIX) $(MORE) -o $(EXEC_DOT_MATRIX) -l papi

checkdirs:
	@mkdir -p $(BUILD_DIR)

all: checkdirs compdotmatrix

exportPAPI:
	@export LD_LIBRARY_PATH=$(PAPILIB): $LD_LIBRARY_PATH
