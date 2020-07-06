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
        pitch_xcorr(x_slice, y_slice, xcorr_slice);
    }
}

fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32]) {
    for i in 0..xcorr.len() {
        let mut sum = 0.0;
        for j in 0..x.len() {
            sum += x[j] * y[j + i];
        }
        xcorr[i] = sum;
    }
}

/// Returns the indices with the largest and second-largest normalized auto-correlation.
///
/// `xcorr` is the autocorrelation of `ys`, taken with windows of length `len`.
///
/// To be a little more precise, the function that we're maximizing is xcorr[i] * xcorr[i],
/// divided by the squared norm of ys[i..(i+len)] (but with a bit of fudging to avoid dividing
/// by small things).
fn find_best_pitch(xcorr: &[f32], ys: &[f32], len: usize) -> (usize, usize) {
    let mut best_num = -1.0;
    let mut second_best_num = -1.0;
    let mut best_den = 0.0;
    let mut second_best_den = 0.0;
    let mut best_pitch = 0;
    let mut second_best_pitch = 1;
    let mut y_sq_norm = 1.0;
    for y in &ys[0..len] {
        y_sq_norm += y * y;
    }
    for (i, &corr) in xcorr.iter().enumerate() {
        if corr > 0.0 {
            let num = corr * corr;
            if num * second_best_den > second_best_num * y_sq_norm {
                if num * best_den > best_num * y_sq_norm {
                    second_best_num = best_num;
                    second_best_den = best_den;
                    second_best_pitch = best_pitch;
                    best_num = num;
                    best_den = y_sq_norm;
                    best_pitch = i;
                } else {
                    second_best_num = num;
                    second_best_den = y_sq_norm;
                    second_best_pitch = i;
                }
            }
        }
        y_sq_norm += ys[i + len] * ys[i + len] - ys[i] * ys[i];
        y_sq_norm = y_sq_norm.max(1.0);
    }
    (best_pitch, second_best_pitch)
}

#[no_mangle]
pub extern "C" fn pitch_search(
    x_lp: *const f32,
    y: *const f32,
    len: c_int,
    max_pitch: c_int,
    pitch: *mut c_int,
) {
    unsafe {
        let x_slice = std::slice::from_raw_parts(x_lp, len as usize);
        let y_slice = std::slice::from_raw_parts(y, len as usize + max_pitch as usize);
        *pitch = rs_pitch_search(x_slice, y_slice, len as usize, max_pitch as usize) as c_int;
    }
}

// TODO: document this. There are some puzzles, commented below.
fn rs_pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;

    // FIXME: allocation
    let mut x_lp4 = vec![0.0; len / 4];
    let mut y_lp4 = vec![0.0; lag / 4];
    // It seems like only the first half of this is really used? The second half seems to always
    // stay zero.
    let mut xcorr = vec![0.0; max_pitch / 2];

    // It says "again", but this was only downsampled once? Also, it's downsampling only the first
    // half by 2.
    /* Downsample by 2 again */
    for j in 0..x_lp4.len() {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..y_lp4.len() {
        y_lp4[j] = y[2 * j];
    }
    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr[0..(max_pitch / 4)]);

    let (best_pitch, second_best_pitch) =
        find_best_pitch(&xcorr[0..(max_pitch / 4)], &y_lp4, len / 4);

    /* Finer search with 2x decimation */
    for i in 0..(max_pitch as isize / 2) {
        xcorr[i as usize] = 0.0;
        if (i - 2 * best_pitch as isize).abs() > 2 && (i - 2 * second_best_pitch as isize).abs() > 2
        {
            continue;
        }
        let mut sum = 0.0;
        // TODO: factor out an inner_prod function
        for j in 0..(len / 2) {
            sum += x_lp[j] * y[j + i as usize];
        }
        xcorr[i as usize] = sum.max(-1.0);
    }

    let (best_pitch, _) = find_best_pitch(&xcorr, &y, len / 2);

    /* Refine by pseudo-interpolation */
    let offset: isize = if best_pitch > 0 && best_pitch < (max_pitch / 2) - 1 {
        let a = xcorr[best_pitch - 1];
        let b = xcorr[best_pitch];
        let c = xcorr[best_pitch + 1];
        if c - a > 0.7 * (b - a) {
            1
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best_pitch as isize - offset) as usize
}
