UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    # Homebrew GCC supports OpenMP; Apple clang does not
    CC     = gcc-14
    OMP    = -fopenmp
else
    CC     = gcc
    OMP    = -fopenmp
endif
CFLAGS  = -O3 -march=native -fno-math-errno -fno-trapping-math -fPIC -std=c11 $(OMP)
LDFLAGS = -lm $(OMP)
SRC    = src/hod.c

ifeq ($(UNAME), Darwin)
    EXT    = dylib
    SHFLAG = -dynamiclib
else
    EXT    = so
    SHFLAG = -shared
endif

TARGET = src/libhod.$(EXT)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SHFLAG) $(LDFLAGS) -o $@ $<

clean:
	rm -f src/libhod.so src/libhod.dylib
