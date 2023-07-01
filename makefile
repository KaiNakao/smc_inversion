COMPILER_CPP = icpx
COMPILER_F = ifx
CFLAGS   = -Wall -Wextra -Wno-sign-compare -Wno-tautological-constant-compare -O3 -fiopenmp -lifcore -qmkl -lmpi -fstack-usage
FFLAGS   = -O3 -fiopenmp
# CFLAGS   = -Wall -Wextra -Wno-sign-compare -Wno-tautological-constant-compare -g -fiopenmp -lifcore -qmkl
# FFLAGS   = -g -fiopenmp
TARGET   = ./main
OBJDIR   = ./obj
SOURCES_CPP = $(wildcard src/*.cpp)
OBJECTS_CPP = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES_CPP:.cpp=.o)))
OBJECT_DC3D = $(addprefix $(OBJDIR)/, DC3Dfortran.o) 

$(TARGET): $(OBJECTS_CPP) $(OBJECT_DC3D)
	$(COMPILER_CPP) $(CFLAGS) -o $@ $^

obj/init.o: src/init.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^

obj/gfunc.o: src/gfunc.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^

obj/smc_slip.o: src/smc_slip.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^

obj/smc_fault.o: src/smc_fault.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^
	
obj/linalg.o: src/linalg.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^

obj/main.o: src/main.cpp
	$(COMPILER_CPP) $(CFLAGS) $(INCLUDE) -o $@ -c $^

obj/DC3Dfortran.o: src/DC3Dfortran.f
	$(COMPILER_F) $(FFLAGS) -o $@ -c $^

all: clean $(TARGET)

clean:
	rm -f $(OBJECTS_CPP) $(OBJECT_DC3D) $(OBJECT_NNLS) $(TARGET)
