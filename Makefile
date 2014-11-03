
include vars.mk

ifndef L_QGLVIEWER
L_QGLVIEWER=-lQGLViewer
endif

ifndef NARCH
NARCH=30
endif

####################
### LIB EXTERNES ###
####################

OS=$(shell uname -s)

#############
## REMINDER #########################################################
# - Penser à utiliser -isystem  au lieu de -I pour les lib externes #
# - Nvcc support experimental du c++11 ...                          #
#####################################################################

# Linux ########################################################
ifeq ($(OS), Linux)

VIEWER_LIBPATH = -L/usr/lib/x86_64-linux-gnu -L/usr/X11R6/lib64 
VIEWER_INCLUDEPATH = -isystem /usr/include -isystem /usr/share/qt4/mkspecs/linux-g++-64 -isystem /usr/X11R6/include -isystem /usr/include/qt4/ $(foreach dir, $(shell ls /usr/include/qt4 | xargs), -isystem /usr/include/qt4/$(dir))
VIEWER_LIBS = -lglut -lGLU -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread -lGLEW -lX11 -lXt -lXi -lXmu -lXext
VIEWER_DEFINES = -D_REENTRANT -DQT_NO_DEBUG -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED

CUDA_INCLUDEPATH = -isystem /usr/local/cuda/include
CUDA_LIBPATH = -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
CUDA_LIBS = -lcuda -lcudart -lcudadevrt

NVCC=nvcc

OPENAL_INCLUDEPATH =
OPENAL_LIBPATH =
OPENAL_LIBS = -lopenal -lalut

DISTRIB=$(filter-out Distributor ID:, $(shell lsb_release -i))
ifeq ($(DISTRIB), Ubuntu)
else #Other
endif

endif
###############################################################

#Compilateurs
LINK= nvcc
LDFLAGS= $(L_QGLVIEWER) $(VIEWER_LIBS) $(CUDA_LIBS) $(OPENAL_LIBS) -llog4cpp -lgomp 
INCLUDE = -I$(SRCDIR) $(foreach dir, $(call subdirs, $(SRCDIR)), -I$(dir)) $(VIEWER_INCLUDEPATH) $(CUDA_INCLUDEPATH) $(OPENAl_INCLUDEPATH)
LIBS = -Llocal/lib/ $(VIEWER_LIBPATH) $(CUDA_LIBPATH) $(OPENAL_LIBPATH)
DEFINES= $(VIEWER_DEFINES) $(OPT)

CXX=g++

CLANG_WERR= -Wall -Wextra -Wmissing-format-attribute -Wmissing-noreturn -Wredundant-decls -Wsequence-point -Wswitch-default -Wdeprecated -Wunreachable-code  -Wsign-conversion -Wold-style-cast -Wcovered-switch-default -Wmissing-variable-declarations -Wfloat-equal -Wunknown-warning-option
CLANG_WNOERR= -Wno-weak-vtables -Wno-c++98-compat-pedantic -Wno-unused-parameter -Wno-deprecated-register -Wno-conversion -Wno-shadow -Wno-padded -Wno-global-constructors -Wno-exit-time-destructors -Wno-source-uses-openmp -Wno-effc++

GCC_WERR= -Wall -Wextra -Wmissing-format-attribute -Wmissing-noreturn -Wredundant-decls -Wsequence-point -Wdeprecated -Wunreachable-code -Wold-style-cast -Wfloat-equal -Wsuggest-attribute=const -Wsuggest-attribute=pure
GCC_WNOERR= -Wno-unused-parameter -Wno-conversion -Wno-shadow -Wno-padded -Wno-effc++ -Wno-double-promotion -Wno-sign-conversion

CLANG_FLAGS= -std=c++11 -m64 -fopenmp $(CLANG_WERR) $(CLANG_WNOERR)
GCC_FLAGS= -std=c++11 -m64 -fopenmp $(GCC_WERR) $(GCC_WNOERR)

ifeq ($(CXX), clang)
	CXXFLAGS=$(CLANG_FLAGS)
else
	CXXFLAGS=$(GCC_FLAGS)
endif

NVCCFLAGS= -arch sm_$(NARCH) --relocatable-device-code true -std=c++11 -Xcompiler="-Wall -Wextra -Wno-unused-parameter -m64 -fopenmp"

ifeq ($(LINK), nvcc)
LINKFLAGS=$(NVCCFLAGS)
else
LINKFLAGS=$(CXXFLAGS)
endif

#preprocesseur QT
MOC=moc
MOCFLAGS=

# Autres flags
DEBUGFLAGS= -g -O0
CUDADEBUGFLAGS= -G -g -Xptxas="-v"
PROFILINGFLAGS= -pg
RELEASEFLAGS= -O3

# Source et destination des fichiers
TARGET = main

SRCDIR = $(realpath .)/src
OBJDIR = $(realpath .)/obj
EXCL=#excluded dirs in src
EXCLUDED_SUBDIRS = $(foreach DIR, $(EXCL), $(call subdirs, $(SRCDIR)/$(DIR)))
SUBDIRS =  $(filter-out $(EXCLUDED_SUBDIRS), $(call subdirs, $(SRCDIR)))

SRC_EXTENSIONS = c C cc cpp cu
SRC_WEXT = $(addprefix *., $(SRC_EXTENSIONS))

HEADERS_EXTENSIONS = h hh hpp
HEADERS_WEXT = $(addprefix *., $(HEADERS_EXTENSIONS))

MOCSRC = $(shell grep -rlw $(SRCDIR)/ -e 'Q_OBJECT' --include=*.h | xargs) #need QT preprocessor
MOCOUTPUT = $(addsuffix .moc, $(basename $(MOCSRC)))
SRC = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(SRC_WEXT), $(wildcard $(DIR)/$(EXT))))
HEADERS = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(HEADERS_WEXT), $(wildcard $(DIR)/$(EXT))))
OBJ = $(subst $(SRCDIR), $(OBJDIR), $(addsuffix .o, $(SRC)))

include rules.mk
