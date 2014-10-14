
# Macros
containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
not_containing = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
subdirs = $(shell find $1 -type d)

# Règles

debug: CXXFLAGS += $(DEBUGFLAGS)
debug: NVCCFLAGS += $(CUDADEBUGFLAGS)
ifeq ($(LINK), NVCC)
debug: LINKFLAGS = $(CUDADEBUGFLAGS) 
else
debug: LINKFLAGS = $(DEBUGFLAGS) 
endif
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

$(TARGET): $(MOCOUTPUT) $(OBJ)
	@echo
	@echo
	$(LINK) $(LIBS) $(OBJ) -o $@ $(LDFLAGS) $(LINKFLAGS) $(DEFINES)
	@echo

#QT macro preprocessor
$(SRCDIR)%.moc : $(SRCDIR)%.hpp
	@echo
	$(MOC) $(INCLUDE) $(DEFINES) -o $@ -i $^ $(MOCFLAGS)
################


$(OBJDIR)%.o : $(SRCDIR)%.C 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo
$(OBJDIR)%.o : $(SRCDIR)%.cc 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo
$(OBJDIR)%.o : $(SRCDIR)%.cpp 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo


$(OBJDIR)%.o: $(SRCDIR)%.cu 
	@echo
	$(NVCC) $(INCLUDE) -o $@ -c $^ $(NVCCFLAGS) $(DEFINES)


# "-" pour enlever les messages d'erreurs
# "@" pour silent
.PHONY: clean cleanall create_dirs all distrib

clean:
	-@rm -f $(OBJ) 

cleanall:
	-@rm -rf $(TARGET) $(TARGET).out $(OBJDIR) $(MOCOUTPUT)

create_dirs:
	@mkdir -p $(subst $(SRCDIR), $(OBJDIR), $(SUBDIRS))

distrib:
	echo $(DISTRIB)
