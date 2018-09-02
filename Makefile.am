ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/src $(DEPS_CFLAGS)

dist_doc_DATA = COPYING AUTHORS README

include_HEADERS = include/rnnoise-nu.h

lib_LTLIBRARIES = librnnoise-nu.la
noinst_HEADERS = src/arch.h  \
		 src/celt_lpc.h  \
		 src/common.h  \
		 src/_kiss_fft_guts.h  \
		 src/kiss_fft.h  \
		 src/opus_types.h  \
		 src/pitch.h  \
		 src/rnn_data.h  \
		 src/rnn.h  \
		 src/tansig_table.h

librnnoise_nu_la_SOURCES = \
	src/denoise.c \
	src/rnn.c \
	src/rnn_data.c \
	src/models/cb.c \
	src/models/mp.c \
	src/models/bd.c \
	src/models/lq.c \
	src/models/sh.c \
	src/rnn_reader.c \
	src/models.c \
	src/pitch.c \
	src/kiss_fft.c \
	src/celt_lpc.c

librnnoise_nu_la_LIBADD = $(DEPS_LIBS) $(lrintf_lib) $(LIBM)
librnnoise_nu_la_LDFLAGS = -no-undefined \
 -version-info @OP_LT_CURRENT@:@OP_LT_REVISION@:@OP_LT_AGE@

if OP_ENABLE_EXAMPLES
noinst_PROGRAMS = examples/rnnoise_nu_demo
endif

examples_rnnoise_nu_demo_SOURCES = examples/rnnoise_demo.c
examples_rnnoise_nu_demo_LDADD = librnnoise-nu.la $(LIBM)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = rnnoise-nu.pc

debug:
	$(MAKE) CFLAGS="${CFLAGS} -O0 -ggdb -DOP_ENABLE_ASSERTIONS" all

EXTRA_DIST = \
 rnnoise-nu.pc.in \
 rnnoise-nu-uninstalled.pc.in \
 doc/Doxyfile.in \
 doc/Makefile

# Targets to build and install just the library without the docs
librnnoise-nu install-librnnoise-nu: NO_DOXYGEN = 1

rnnoise-nu: all
install-rnnoise-nu: install

# Or just the docs
docs: doc/doxygen-build.stamp

install-docs:
	@if [ -z "$(NO_DOXYGEN)" ]; then \
	  ( cd doc && \
	    echo "Installing documentation in $(DESTDIR)$(docdir)"; \
	    $(INSTALL) -d $(DESTDIR)$(docdir)/html/search; \
	    for f in `find html -type f \! -name "installdox"` ; do \
	      $(INSTALL_DATA) $$f $(DESTDIR)$(docdir)/$$f;     \
	    done ) \
	fi

doc/doxygen-build.stamp: doc/Doxyfile \
                         $(top_srcdir)/include/*.h
	@[ -n "$(NO_DOXYGEN)" ] || ( cd doc && doxygen && touch $(@F) )


if HAVE_DOXYGEN

# Or everything (by default)
all-local: docs

install-data-local: install-docs

clean-local:
	$(RM) -r doc/html
	$(RM) -r doc/latex
	$(RM) doc/doxygen-build.stamp

uninstall-local:
	$(RM) -r $(DESTDIR)$(docdir)/html

endif

.PHONY: rnnoise-nu install-rnnoise-nu docs install-docs
