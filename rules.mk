
# Macros
containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
not_containing = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
subdirs = $(shell find $1 -type d)

# Règles

default: all

debug: CXXFLAGS += $(DEBUGFLAGS)
debug: NVCCFLAGS += $(CUDADEBUGFLAGS)
ifeq ($(LINK), nvcc)
debug: LINKFLAGS = $(CUDADEBUGFLAGS) 
else
debug: LINKFLAGS = $(DEBUGFLAGS) 
endif
debug: DEFINES+= -D_DEBUG -D_CONSOLE_LOG_LEVEL="DEBUG" -D_DEBUG_LEVEL=1
debug: all

profile: LINKFLAGS += $(PROFILINGFLAGS)
profile: CFLAGS += $(PROFILINGFLAGS)
profile: CXXFLAGS += $(PROFILINGFLAGS)
profile: all

release: LINKFLAGS += $(RELEASEFLAGS)
release: CFLAGS += $(RELEASEFLAGS)
release: CXXFLAGS += $(RELEASEFLAGS)
release: NVCCFLAGS += $(RELEASEFLAGS)
release: all

all: create_dirs $(TARGET)

$(TARGET): $(MOCOUTPUT) $(OBJ) $(SRC)
	@echo
	@echo
	$(LINK) $(LIBS) $(OBJ) -o $@ $(LDFLAGS) $(LINKFLAGS) $(DEFINES)
	@echo

#QT macro preprocessor
$(SRCDIR)%.moc : $(SRCDIR)%.hpp
	@echo
	$(MOC) $(INCLUDE) $(DEFINES) -o $@ -i $^ $(MOCFLAGS)
################


$(OBJDIR)%.cpp.o : $(SRCDIR)%.cpp
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo

$(OBJDIR)%.cu.o: $(SRCDIR)%.cu
	@echo
	$(NVCC) $(INCLUDE) -o $@ -c $^ $(NVCCFLAGS) $(DEFINES)


# "-" pour enlever les messages d'erreurs
# "@" pour silent
.PHONY: clean cleanall create_dirs all distrib tags

clean:
	-@rm -f $(OBJ) 

cleanall:
	-@rm -rf $(TARGET) $(TARGET).out $(OBJDIR) $(MOCOUTPUT)

create_dirs:
	@mkdir -p $(subst $(SRCDIR), $(OBJDIR), $(SUBDIRS))

distrib:
	echo $(DISTRIB)
	
tags:
	$(CXX) $(INCLUDE) -M $(SRC) 2> /dev/null | grep -o '[^ ]*\.h\{1,2\}p\{0,2\}[^p]' | sort | uniq | ctags -L - --c++-kinds=+p --fields=+iaS --extra=+q 
