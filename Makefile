
include vars.mk

ifndef L_QGLVIEWER
L_QGLVIEWER=-lQGLViewer
endif

ifndef NARCH
NARCH=11
endif

####################
### LIB EXTERNES ###
####################

OS=$(shell uname -s)

# Linux ########################################################
ifeq ($(OS), Linux)

VIEWER_LIBPATH = -L/usr/lib/x86_64-linux-gnu -L/usr/X11R6/lib64 
VIEWER_INCLUDEPATH = -I/usr/include/Qt -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/share/qt4/mkspecs/linux-g++-64 -I/usr/include/QtOpenGL -I/usr/include/QtXml -I/usr/X11R6/include -I/usr/include/qt4/ $(foreach dir, $(shell ls /usr/include/qt4 | xargs), -I/usr/include/qt4/$(dir))
VIEWER_LIBS = -lglut -lGLU -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread -lGLEW -lX11 -lXt -lXi -lXmu -lXext
VIEWER_DEFINES = -D_REENTRANT -DQT_NO_DEBUG -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED

CUDA_INCLUDEPATH = -I/usr/local/cuda/include
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
LDFLAGS= $(L_QGLVIEWER) $(VIEWER_LIBS) -llog4cpp $(CUDA_LIBS) $(OPENAL_LIBS)
INCLUDE = -Ilocal/include/ -I$(SRCDIR) $(foreach dir, $(call subdirs, $(SRCDIR)), -I$(dir)) $(VIEWER_INCLUDEPATH) $(CUDA_INCLUDEPATH) $(OPENAl_INCLUDEPATH)
LIBS = -Llocal/lib/ $(VIEWER_LIBPATH) $(CUDA_LIBPATH) $(OPENAL_LIBPATH)
DEFINES= $(VIEWER_DEFINES) $(OPT)

CXX=g++
CXXFLAGS= -W -Wall -Wextra -Wno-unused-parameter -pedantic -std=c++11 -m64
#-Wshadow -Wstrict-aliasing -Weffc++ -Werror
NVCCFLAGS= -Xcompiler -Wall -m64 -arch sm_$(NARCH) --relocatable-device-code true

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
