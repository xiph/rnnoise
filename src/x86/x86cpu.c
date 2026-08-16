/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cpu_support.h"
#include "pitch.h"
#include "x86cpu.h"

#ifdef RNN_ENABLE_X86_RTCD

#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef _MSC_VER

#include <intrin.h>
#include <immintrin.h>
static _inline void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
    __cpuid((int*)CPUInfo, InfoType);
}
static _inline unsigned long long xgetbv0(void)
{
    return _xgetbv(0);
}

#else

#if defined(CPU_INFO_BY_C)
#include <cpuid.h>
#endif

static void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
#if defined(CPU_INFO_BY_ASM)
#if defined(__i386__) && defined(__PIC__)
/* %ebx is PIC register in 32-bit, so mustn't clobber it. */
    __asm__ __volatile__ (
        "xchg %%ebx, %1\n"
        "cpuid\n"
        "xchg %%ebx, %1\n":
        "=a" (CPUInfo[0]),
        "=r" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        /* We clear ECX to avoid a valgrind false-positive prior to v3.17.0. */
        "0" (InfoType), "2" (0)
    );
#else
    __asm__ __volatile__ (
        "cpuid":
        "=a" (CPUInfo[0]),
        "=b" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        /* We clear ECX to avoid a valgrind false-positive prior to v3.17.0. */
        "0" (InfoType), "2" (0)
    );
#endif
#elif defined(CPU_INFO_BY_C)
    /* We use __get_cpuid_count to clear ECX to avoid a valgrind false-positive
        prior to v3.17.0.*/
    if (!__get_cpuid_count(InfoType, 0, &(CPUInfo[0]), &(CPUInfo[1]), &(CPUInfo[2]), &(CPUInfo[3]))) {
        /* Our function cannot fail, but __get_cpuid{_count} can.
           Returning all zeroes will effectively disable all SIMD, which is
            what we want on CPUs that don't support CPUID. */
        CPUInfo[3] = CPUInfo[2] = CPUInfo[1] = CPUInfo[0] = 0;
    }
#else
# error "Configured to use x86 RTCD, but no CPU detection method available. " \
 "Reconfigure with --disable-rtcd (or send patches)."
#endif
}

static unsigned long long xgetbv0(void)
{
#if defined(CPU_INFO_BY_ASM)
    unsigned int eax;
    unsigned int edx;

    __asm__ __volatile__(
        "xgetbv":
        "=a" (eax),
        "=d" (edx) :
        "c" (0)
        );

    return ((unsigned long long)edx << 32) |
        (unsigned long long)eax;

#elif defined(CPU_INFO_BY_C)
    return _xgetbv(0);

#else
# error "Configured to use x86 RTCD, but no XGETBV detection method available. " \
        "Reconfigure with --disable-rtcd (or send patches)."
#endif
}
#endif

static int rnn_cpu_feature_check(void)
{
    const unsigned int v2_ecx_mask =
        (1U << 0) | (1U << 9) | (1U << 13) |
        (1U << 19) | (1U << 20) | (1U << 23);

    const unsigned int v3_ecx_mask =
        (1U << 12) | (1U << 22) | (1U << 27) | (1U << 28);

    const unsigned int v3_ebx_mask =
        (1U << 3) | (1U << 5) | (1U << 8);

    const unsigned int v4_ebx_mask =
        (1U << 16) | (1U << 17) | (1U << 28) |
        (1U << 30) | (1U << 31);

    const unsigned long long avx_xcr0_mask = 0x06ULL;
    const unsigned long long avx512_xcr0_mask = 0xE6ULL;

#ifdef _WIN32
    const DWORD64 avx_xstate_mask = XSTATE_AVX;
    const DWORD64 avx512_xstate_mask = XSTATE_AVX |
        XSTATE_AVX512_KMASK |
        XSTATE_AVX512_ZMM_H |
        XSTATE_AVX512_ZMM;
#endif

    unsigned int info[4];
    unsigned int nIds;
    unsigned long long xcr0;
#ifdef _WIN32
    DWORD64 xstate;
#endif

    cpuid(info, 0);
    nIds = info[0];

    /*
     * x86-64-v2
     */
    if (nIds < 1)
        return 1;

    cpuid(info, 1);

    if ((info[3] & (1U << 26)) == 0 ||
        (info[2] & v2_ecx_mask) != v2_ecx_mask)
        return 1;

    /*
     * x86-64-v3
     */
    if (nIds < 7)
        return 2;

    if ((info[2] & v3_ecx_mask) != v3_ecx_mask)
        return 2;

    cpuid(info, 7);

    if ((info[1] & v3_ebx_mask) != v3_ebx_mask)
        return 2;

    /*
     * XMM + YMM state is required for v3.
     */
    xcr0 = xgetbv0();

    if ((xcr0 & avx_xcr0_mask) != avx_xcr0_mask)
        return 2;

#ifdef _WIN32
    xstate = GetEnabledXStateFeatures();

    if ((xstate & avx_xstate_mask) != avx_xstate_mask)
        return 2;
#endif

#ifdef RNNOISE_X86_64_V4
    /*
     * x86-64-v4
     */
    if ((info[1] & v4_ebx_mask) != v4_ebx_mask)
        return 3;

    /*
     * Opmask + ZMM state is additionally required for v4.
     */
    if ((xcr0 & avx512_xcr0_mask) != avx512_xcr0_mask)
        return 3;

#ifdef _WIN32
    if ((xstate & avx512_xstate_mask) != avx512_xstate_mask)
        return 3;
#endif

    return 4;
#else
    return 3;
#endif
}

static int rnn_select_arch_impl(void)
{
    return rnn_cpu_feature_check() - 1;
}

int rnn_select_arch(void) {
    int arch = rnn_select_arch_impl();
#ifdef FUZZING
    /* Randomly downgrade the architecture. */
    arch = rand()%(arch+1);
#endif
    return arch;
}

#endif
