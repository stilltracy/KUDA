
# include paths to KUDA, PIN, and CUDA
include ../paths.config


DEBUG=1

##
## PIN tools
##

##############################################################
#
# Here are some things you might want to configure
#
##############################################################

TARGET_COMPILER?=gnu
ifdef OS
    ifeq (${OS},Windows_NT)
        TARGET_COMPILER=ms
    endif
endif

##############################################################
#
# include *.config files
#
##############################################################

ifeq ($(TARGET_COMPILER),gnu)
    include ../makefile.gnu.config
    CXXFLAGS ?= -Wall -Wno-unknown-pragmas $(DBG) $(OPT)
endif

ifeq ($(TARGET_COMPILER),ms)
    include ../makefile.ms.config
    DBG?=
endif

##############################################################
#
# Tools sets
#
##############################################################


TOOL_ROOTS = lockTool goldilocks vectorClock trace djitplus fasttrack eraser 
#transactionalmem
STATIC_TOOL_ROOTS =
APP_NAMES = 

SPECIAL_TOOL_ROOTS = 

TOOLS = $(TOOL_ROOTS:%=$(OBJDIR)%$(PINTOOL_SUFFIX))
STATIC_TOOLS = $(STATIC_TOOL_ROOTS:%=$(OBJDIR)%$(SATOOL_SUFFIX))
SPECIAL_TOOLS = $(SPECIAL_TOOL_ROOTS:%=$(OBJDIR)%$(PINTOOL_SUFFIX))
APPS = $(APP_NAMES:%=$(OBJDIR)%$(EXEEXT))

##############################################################
#
# build rules
#
##############################################################

all: tools apps
tools: $(TOOLS) $(STATIC_TOOLS) $(SPECIAL_TOOLS)
apps: $(APPS)
test: $(TOOL_ROOTS:%=%.test) $(STATIC_TOOL_ROOTS:%=%.test) $(SPECIAL_TOOL_ROOTS:%=%.test)


##############################################################
#
# applications
#
##############################################################




##############################################################
#
# pin tools
#
##############################################################

lockTool.test : $(OBJDIR)lockTool$(PINTOOL_SUFFIX)

goldilocks.test : $(OBJDIR)goldilocks$(PINTOOL_SUFFIX)

vectorClock.test  : $(OBJDIR)vectorClock$(PINTOOL_SUFFIX)

trace.test : $(OBJDIR)trace$(PINTOOL_SUFFIX)

djitplus.test : $(OBJDIR)djitplus$(PINTOOL_SUFFIX)

fasttrack.test : $(OBJDIR)fasttrack$(PINTOOL_SUFFIX)

eraser.test : $(OBJDIR)eraser$(PINTOOL_SUFFIX)

##############################################################
#
# build rules
#
##############################################################

$(APPS): $(OBJDIR)make-directory

$(OBJDIR)make-directory:
	mkdir -p $(OBJDIR)
	touch $(OBJDIR)make-directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)%.o : %.cpp $(OBJDIR)make-directory
	$(CXX) -c $(CXXFLAGS) $(PIN_CXXFLAGS) ${OUTOPT}$@ $<

$(TOOLS): $(PIN_LIBNAMES)

$(TOOLS): %$(PINTOOL_SUFFIX) : %.o
	${PIN_LD} $(PIN_LDFLAGS) $(LINK_DEBUG) ${LINK_OUT}$@ $< ${PIN_LPATHS} $(PIN_LIBS) $(DBG)

$(STATIC_TOOLS): $(PIN_LIBNAMES)

$(STATIC_TOOLS): %$(SATOOL_SUFFIX) : %.o
	${PIN_LD} $(PIN_SALDFLAGS) $(LINK_DEBUG) ${LINK_OUT}$@ $< ${PIN_LPATHS} $(SAPIN_LIBS) $(DBG)

## cleaning
clean:
	-rm -rf $(OBJDIR) *.out *.tested *.failed *.makefile.copy *.out.*.*
