use libc::c_int;

#[no_mangle]
pub extern "C" fn _celt_lpc(lpc: *mut f32, ac: *const f32, p: c_int) {
    unsafe {
        let lpc_slice = std::slice::from_raw_parts_mut(lpc, p as usize);
        let ac_slice = std::slice::from_raw_parts(ac, p as usize + 1);
        celt_lpc(lpc_slice, ac_slice);
    }
}

fn celt_lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    let mut error = ac[0];

    for b in lpc.iter_mut() {
        *b = 0.0;
    }

    if ac[0] == 0.0 {
        return;
    }

    for i in 0..p {
        // Sum up this iteration's reflection coefficient
        let mut rr = 0.0;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        // Update LPC coefficients and total error
        lpc[i] = r;
        for j in 0..((i + 1) / 2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;
        // Bail out once we get 30 dB gain
        if error < 0.001 * ac[0] {
            return;
        }
    }
}

#[no_mangle]
pub extern "C" fn celt_pitch_xcorr(
    x: *const f32,
    y: *const f32,
    xcorr: *mut f32,
    len: c_int,
    max_pitch: c_int,
) {
    unsafe {
        let x_slice = std::slice::from_raw_parts(x, len as usize);
        let y_slice = std::slice::from_raw_parts(y, len as usize + max_pitch as usize - 1);
        let xcorr_slice = std::slice::from_raw_parts_mut(xcorr, max_pitch as usize);
        pitch_xcorr(x_slice, y_slice, xcorr_slice, max_pitch as usize);
    }
}

fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], max_pitch: usize) {
    for i in 0..max_pitch {
        let mut sum = 0.0;
        for j in 0..x.len() {
            sum += x[j] * y[j + i];
        }
        xcorr[i] = sum;
    }
}
