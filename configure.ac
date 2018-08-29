# autoconf source script for generating configure

AC_INIT([rnnoise-nu],[1.0],[gregor.richards@uwaterloo.ca])
AC_CONFIG_SRCDIR([src/denoise.c])
AC_CONFIG_AUX_DIR([scripts])
AC_CONFIG_MACRO_DIR([m4])

AC_USE_SYSTEM_EXTENSIONS
AC_SYS_LARGEFILE

m4_include([m4/lean.m4])

AM_INIT_AUTOMAKE([1.11 foreign no-define dist-zip subdir-objects])
AM_MAINTAINER_MODE([enable])

AC_C_INLINE

LT_INIT

m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_DEFINE([RNNOISE_BUILD], [], [This is a build of the library])

dnl Library versioning for libtool.
dnl For some inane reason, in .so.major.minor.revision:
dnl - major = CURRENT-AGE
dnl - minor = AGE
dnl - revision = REVISION

OP_LT_CURRENT=1
OP_LT_REVISION=0
OP_LT_AGE=0

AC_SUBST(OP_LT_CURRENT)
AC_SUBST(OP_LT_REVISION)
AC_SUBST(OP_LT_AGE)

AC_ARG_ENABLE([assertions],
  AS_HELP_STRING([--enable-assertions], [Enable assertions in code]),,
  enable_assertions=no)

AS_IF([test "$enable_assertions" = "yes"], [
  AC_DEFINE([OP_ENABLE_ASSERTIONS], [1], [Enable assertions in code])
])

AC_ARG_ENABLE([examples],
  AS_HELP_STRING([--disable-examples], [Do not build example applications]),,
  enable_examples=yes)
AM_CONDITIONAL([OP_ENABLE_EXAMPLES], [test "$enable_examples" = "yes"])

AS_CASE(["$ac_cv_search_lrintf"],
  ["no"],[],
  ["none required"],[],
  [lrintf_lib="$ac_cv_search_lrintf"])

LT_LIB_M

AC_SUBST([lrintf_lib])

CC_ATTRIBUTE_VISIBILITY([default], [
  CC_FLAG_VISIBILITY([CFLAGS="${CFLAGS} -fvisibility=hidden"])
])

dnl Check for doxygen
AC_ARG_ENABLE([doc],
  AS_HELP_STRING([--disable-doc], [Do not build API documentation]),,
  [enable_doc=yes]
)

AS_IF([test "$enable_doc" = "yes"], [
  AC_CHECK_PROG([HAVE_DOXYGEN], [doxygen], [yes], [no])
  AC_CHECK_PROG([HAVE_DOT], [dot], [yes], [no])
],[
  HAVE_DOXYGEN=no
])

AM_CONDITIONAL([HAVE_DOXYGEN], [test "$HAVE_DOXYGEN" = "yes"])

AC_CONFIG_FILES([
  Makefile
  rnnoise-nu.pc
  rnnoise-nu-uninstalled.pc
  doc/Doxyfile
])
AC_CONFIG_HEADERS([config.h])
AC_OUTPUT

AC_MSG_NOTICE([
------------------------------------------------------------------------
  $PACKAGE_NAME $PACKAGE_VERSION: Automatic configuration OK.

    Assertions ................... ${enable_assertions}

    Hidden visibility ............ ${cc_cv_flag_visibility}

    API code examples ............ ${enable_examples}
    API documentation ............ ${enable_doc}
------------------------------------------------------------------------
])
