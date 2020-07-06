/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file pitch.c
   @brief Pitch analysis
 */

/*
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

#include "pitch.h"
#include "common.h"
//#include "modes.h"
//#include "stack_alloc.h"
//#include "mathops.h"
#include "celt_lpc.h"
#include "math.h"

extern void celt_fir5(const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         opus_val16 *mem);

void pitch_downsample(
                      celt_sig *x[],
                      opus_val16 *x_lp,
                      int len,
                      int C,
                      opus_val16 *xx) /* scratch buffer of size len/2 */
{
   int i;
   opus_val32 ac[5];
   opus_val16 tmp=Q15ONE;
   opus_val16 lpc[4], mem[5]={0,0,0,0,0};
   opus_val16 lpc2[5];
   opus_val16 c1 = QCONST16(.8f,15);
   for (i=1;i<len>>1;i++)
      x_lp[i] = SHR32(HALF32(HALF32(x[0][(2*i-1)]+x[0][(2*i+1)])+x[0][2*i]), shift);
   x_lp[0] = SHR32(HALF32(HALF32(x[0][1])+x[0][0]), shift);
   if (C==2)
   {
      for (i=1;i<len>>1;i++)
         x_lp[i] += SHR32(HALF32(HALF32(x[1][(2*i-1)]+x[1][(2*i+1)])+x[1][2*i]), shift);
      x_lp[0] += SHR32(HALF32(HALF32(x[1][1])+x[1][0]), shift);
   }

   _celt_autocorr(x_lp, ac, NULL, 0,
                  4, len>>1, xx);

   /* Noise floor -40 dB */
   ac[0] *= 1.0001f;
   /* Lag windowing */
   for (i=1;i<=4;i++)
   {
      /*ac[i] *= exp(-.5*(2*M_PI*.002*i)*(2*M_PI*.002*i));*/
      ac[i] -= ac[i]*(.008f*i)*(.008f*i);
   }

   _celt_lpc(lpc, ac, 4);
   for (i=0;i<4;i++)
   {
      tmp = MULT16_16_Q15(QCONST16(.9f,15), tmp);
      lpc[i] = MULT16_16_Q15(lpc[i], tmp);
   }
   /* Add a zero */
   lpc2[0] = lpc[0] + QCONST16(.8f,SIG_SHIFT);
   lpc2[1] = lpc[1] + MULT16_16_Q15(c1,lpc[0]);
   lpc2[2] = lpc[2] + MULT16_16_Q15(c1,lpc[1]);
   lpc2[3] = lpc[3] + MULT16_16_Q15(c1,lpc[2]);
   lpc2[4] = MULT16_16_Q15(c1,lpc[3]);
   celt_fir5(x_lp, lpc2, x_lp, len>>1, mem);
}

extern void pitch_search(const opus_val16 *x_lp, opus_val16 *y,
                  int len, int max_pitch, int *pitch);
static opus_val16 compute_pitch_gain(opus_val32 xy, opus_val32 xx, opus_val32 yy)
{
   return xy/sqrt(1+xx*yy);
}

static const int second_check[16] = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};
opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
      int N, int *T0_, int prev_period, opus_val16 prev_gain)
{
   int k, i, T, T0;
   opus_val16 g, g0;
   opus_val16 pg;
   opus_val32 xy,xx,yy,xy2;
   opus_val32 xcorr[3];
   opus_val32 best_xy, best_yy;
   int offset;
   int minperiod0;

   minperiod0 = minperiod;
   maxperiod /= 2;
   minperiod /= 2;
   *T0_ /= 2;
   prev_period /= 2;
   N /= 2;
   x += maxperiod;
   if (*T0_>=maxperiod)
      *T0_=maxperiod-1;

   T = T0 = *T0_;
   opus_val32 yy_lookup[maxperiod+1];
   dual_inner_prod(x, x, x-T0, N, &xx, &xy);
   yy_lookup[0] = xx;
   yy=xx;
   for (i=1;i<=maxperiod;i++)
   {
      yy = yy+MULT16_16(x[-i],x[-i])-MULT16_16(x[N-i],x[N-i]);
      yy_lookup[i] = MAX32(0, yy);
   }
   yy = yy_lookup[T0];
   best_xy = xy;
   best_yy = yy;
   g = g0 = compute_pitch_gain(xy, xx, yy);
   /* Look for any pitch at T/k */
   for (k=2;k<=15;k++)
   {
      int T1, T1b;
      opus_val16 g1;
      opus_val16 cont=0;
      opus_val16 thresh;
      T1 = (2*T0+k)/(2*k);
      if (T1 < minperiod)
         break;
      /* Look for another strong correlation at T1b */
      if (k==2)
      {
         if (T1+T0>maxperiod)
            T1b = T0;
         else
            T1b = T0+T1;
      } else
      {
         T1b = (2*second_check[k]*T0+k)/(2*k);
      }
      dual_inner_prod(x, &x[-T1], &x[-T1b], N, &xy, &xy2);
      xy = HALF32(xy + xy2);
      yy = HALF32(yy_lookup[T1] + yy_lookup[T1b]);
      g1 = compute_pitch_gain(xy, xx, yy);
      if (abs(T1-prev_period)<=1)
         cont = prev_gain;
      else if (abs(T1-prev_period)<=2 && 5*k*k < T0)
         cont = HALF16(prev_gain);
      else
         cont = 0;
      thresh = MAX16(QCONST16(.3f,15), MULT16_16_Q15(QCONST16(.7f,15),g0)-cont);
      /* Bias against very high pitch (very short period) to avoid false-positives
         due to short-term correlation */
      if (T1<3*minperiod)
         thresh = MAX16(QCONST16(.4f,15), MULT16_16_Q15(QCONST16(.85f,15),g0)-cont);
      else if (T1<2*minperiod)
         thresh = MAX16(QCONST16(.5f,15), MULT16_16_Q15(QCONST16(.9f,15),g0)-cont);
      if (g1 > thresh)
      {
         best_xy = xy;
         best_yy = yy;
         T = T1;
         g = g1;
      }
   }
   best_xy = MAX32(0, best_xy);
   if (best_yy <= best_xy)
      pg = Q15ONE;
   else
      pg = best_xy/(best_yy+1);

   for (k=0;k<3;k++)
      xcorr[k] = celt_inner_prod(x, x-(T+k-1), N);
   if ((xcorr[2]-xcorr[0]) > MULT16_32_Q15(QCONST16(.7f,15),xcorr[1]-xcorr[0]))
      offset = 1;
   else if ((xcorr[0]-xcorr[2]) > MULT16_32_Q15(QCONST16(.7f,15),xcorr[1]-xcorr[2]))
      offset = -1;
   else
      offset = 0;
   if (pg > g)
      pg = g;
   *T0_ = 2*T+offset;

   if (*T0_<minperiod0)
      *T0_=minperiod0;
   return pg;
}
