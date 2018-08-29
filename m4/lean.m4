# autotools lean macros
# hg 2012-09-01 05a8d3fa4611

# Copyright (c) 2012 Gregor Richards
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
# USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
# TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.

# These macros make auto* tests faster by removing some of autoconf's most
# absurd defaults. The basic principle is to not check for things that have no
# alternatives. That is, don't perform a test if it's either going to pass and
# affect nothing, or fail and just prevent you from building. These tests
# provide very little real value since modern systems that they fail on are few
# and far between.


# automake's sanity checks provide nothing useful, since all they can do is
# fail, sometimes spuriously, and prevent builds which may otherwise have
# succeeded.
AC_DEFUN([AM_SANITY_CHECK], [ ])    


# Checking for C89 compliance nowadays is just plain silly.
AC_DEFUN([_AC_PROG_CC_C89], [ ])


# For the same reason, checking for C standard headers is usually stupid.
# However, we simply avoid checking for them in the most ridiculous cases.
m4_define([ACX_PRELEAN_AC_CHECK_HEADER], m4_defn([AC_CHECK_HEADER]))
AC_DEFUN([AC_CHECK_HEADER], [
    m4_case([$4],
            [], [ACX_PRELEAN_AC_CHECK_HEADER([$1], [$2], [$3], [ ])],
                [m4_indir([ACX_PRELEAN_AC_CHECK_HEADER], $@)])
])
m4_define([_AC_HEADERS_EXPANSION], [
    m4_divert_text([DEFAULTS], [ac_header_list=])
    AC_CHECK_HEADERS([$ac_header_list], [], [], [ ])
    m4_define([_AC_HEADERS_EXPANSION], [])
])
m4_define([ACX_PRELEAN_AC_CHECK_SIZEOF], m4_defn([AC_CHECK_SIZEOF]))
AC_DEFUN([AC_CHECK_SIZEOF], [
    m4_case([$3],
            [], [ACX_PRELEAN_AC_CHECK_SIZEOF([$1], [], [ ])],
                [m4_indir([ACX_PRELEAN_AC_CHECK_SIZEOF])], $@)])


# And add warnings for known-nasty builtin checks
m4_define([ACX_UNLEAN_AC_FUNC_MMAP], m4_defn([AC_FUNC_MMAP]))
AC_DEFUN([AC_FUNC_MMAP], [
    AC_DIAGNOSE([syntax], [$0: AC_FUNC_MMAP does not work in cross environments and incurs high costs. Check for mmap directly if you aren't concerned about enormously-broken implementations. Use ACX_LEAN_AC_FUNC_MMAP to silence this warning.])
    ACX_LEAN_AC_FUNC_MMAP
])


# POSIX says that make sets $(MAKE). That's good enough for me.
AC_DEFUN([AC_PROG_MAKE_SET], [
    ac_cv_prog_make_make_set=yes
    SET_MAKE=
    AC_SUBST([SET_MAKE])
])


# configure will simply fail, often spuriously, if you don't tell it that
# you're cross compiling, so there's very little reason to explicitly check.
AC_DEFUN([_AC_COMPILER_EXEEXT_CROSS], [
    if test "$cross_compiling" = "maybe"; then
        cross_compiling=yes
    fi
])


# Allow the default GCC-and-compatible CFLAGS to be changed
GCC_DEFAULT_CFLAGS="-g -O2"


# The builtin -g test is simplified by avoiding rechecks for GCC (of course GCC
# supports -g)
m4_define([ACX_PRELEAN__AC_PROG_CC_G], m4_defn([_AC_PROG_CC_G]))
m4_define([_AC_PROG_CC_G], [
    if test "$GCC" = "yes"; then
        acx_lean_CFLAGS_set=${CFLAGS+set}
        ac_cv_prog_cc_g=yes
        if test "$acx_lean_CFLAGS_set" != "set"; then
            CFLAGS="$GCC_DEFAULT_CFLAGS"
        fi
    else
        ACX_PRELEAN__AC_PROG_CC_G
    fi
])


# Option to force caching
AC_DEFUN([ACX_LEAN_FORCE_CACHE], [
    m4_define([acx_lean_forced_cache], [yes])
    if test "$cache_file" = "/dev/null"; then
        cache_file=config.cache
        touch config.cache
    fi
])


# Force the use of a cache file if we use subdirectories, as otherwise we
# retest things in the subdirs.
m4_define([ACX_PRELEAN_AC_CONFIG_SUBDIRS], m4_defn([AC_CONFIG_SUBDIRS]))
AC_DEFUN([AC_CONFIG_SUBDIRS], [
    m4_ifdef([acx_lean_forced_cache], [], [
        AC_DIAGNOSE([syntax], [$0: Use ACX_LEAN_FORCE_CACHE after initialization to avoid extra costs with configure subdirs.])
    ])
    ACX_PRELEAN_AC_CONFIG_SUBDIRS($1)
])
