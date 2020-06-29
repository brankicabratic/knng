CC := g++ # This is the main compiler
SRCDIR := src
BUILDDIR := build
TARGET := bin/knng
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -O3 -std=c++17 # -Wall

# CONFIGURATION OF BOOST
LIB := -L C:/boost_1_69_0/build/lib -lboost_filesystem-mgw81-mt-x32-1_69
INC := -I C:/boost_1_69_0

RM := rm

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	$(RM) -rf $(BUILDDIR) $(TARGET)

.PHONY: clean