#ifndef XCORR_IMPL_H
#define XCORR_IMPL_H

extern "C"
{

void xcorr_native_impl(const float * x, const float * y, float sum[4], int len);

}
#endif