use libc::c_int;
use once_cell::sync::OnceCell;

mod rnn;

fn inner_prod(xs: &[f32], ys: &[f32], n: usize) -> f32 {
    xs[..n]
        .iter()
        .zip(ys[..n].iter())
        .map(|(&x, &y)| x * y)
        .sum()
}

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

// Computes various terms of the correlation (what's the right word?) between x and y. Note that
// the C version has been heavily optimized, unlike this one.
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

fn fir5(x: &[f32], num: &[f32], y: &mut [f32], mem: &mut [f32]) {
    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];

    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    let mut mem2 = mem[2];
    let mut mem3 = mem[3];
    let mut mem4 = mem[4];

    for i in 0..x.len() {
        let sum = x[i] + num0 * mem0 + num1 * mem1 + num2 * mem2 + num3 * mem3 + num4 * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = x[i];
        y[i] = sum;
    }

    mem[0] = mem0;
    mem[1] = mem1;
    mem[2] = mem2;
    mem[3] = mem3;
    mem[4] = mem4;
}

#[no_mangle]
pub extern "C" fn _celt_autocorr(
    x: *const f32,
    ac: *mut f32,
    window: *const f32,
    overlap: c_int,
    lag: c_int,
    n: c_int,
    _xx: *const f32,
) -> c_int {
    assert_eq!(overlap, 0);
    assert!(window.is_null());
    unsafe {
        let x_slice = std::slice::from_raw_parts(x, n as usize);
        let ac_slice = std::slice::from_raw_parts_mut(ac, lag as usize + 1);
        celt_autocorr(x_slice, ac_slice);
        return 0;
    }
}

/// Computes the autocorrelation of the sequence `x` (the number of terms to compute is determined
/// by the length of `ac`).
fn celt_autocorr(x: &[f32], ac: &mut [f32]) {
    let n = x.len();
    let lag = ac.len() - 1;
    // FIXME: check if ac.len() is lag or lag - 1.
    let fast_n = n - lag;
    pitch_xcorr(&x[0..fast_n], x, ac);

    for k in 0..ac.len() {
        let mut d = 0.0;
        for i in (k + fast_n)..n {
            d += x[i] * x[i - k];
        }
        ac[k] += d;
    }
}

#[no_mangle]
pub extern "C" fn pitch_downsample(
    x: *const *const f32,
    x_lp: *mut f32,
    len: c_int,
    c: c_int,
    _xx: *const f32,
) {
    assert_eq!(c, 1);
    unsafe {
        let x_slice = std::slice::from_raw_parts(*x, len as usize);
        let x_lp_slice = std::slice::from_raw_parts_mut(x_lp, len as usize / 2);
        rs_pitch_downsample(x_slice, x_lp_slice);
    }
}

fn rs_pitch_downsample(x: &[f32], x_lp: &mut [f32]) {
    let mut ac = [0.0; 5];
    let mut lpc = [0.0; 4];
    let mut mem = [0.0; 5];
    let mut lpc2 = [0.0; 5];

    for i in 1..(x.len() / 2) {
        x_lp[i] = ((x[2 * i - 1] + x[2 * i + 1]) / 2.0 + x[2 * i]) / 2.0;
    }
    x_lp[0] = (x[1] / 2.0 + x[0]) / 2.0;

    celt_autocorr(x_lp, &mut ac);

    // Noise floor -40 dB
    ac[0] *= 1.0001;
    // Lag windowing
    for i in 1..5 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }

    celt_lpc(&mut lpc, &ac);
    let mut tmp = 1.0;
    for i in 0..4 {
        tmp *= 0.9;
        lpc[i] *= tmp;
    }
    // Add a zero
    lpc2[0] = lpc[0] + 0.8;
    lpc2[1] = lpc[1] + 0.8 * lpc[0];
    lpc2[2] = lpc[2] + 0.8 * lpc[1];
    lpc2[3] = lpc[3] + 0.8 * lpc[2];
    lpc2[4] = 0.8 * lpc[3];

    // FIXME: allocation
    let x_lp_copy = x_lp.to_owned();
    fir5(&x_lp_copy, &lpc2, x_lp, &mut mem);
}

fn pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

#[no_mangle]
pub extern "C" fn remove_doubling(
    x: *const f32,
    max_period: c_int,
    min_period: c_int,
    n: c_int,
    t0_: *mut c_int,
    prev_period: c_int,
    prev_gain: f32,
) -> f32 {
    unsafe {
        let x_slice = std::slice::from_raw_parts(x, max_period as usize + n as usize);
        let (t0, gain) = rs_remove_doubling(
            x_slice,
            max_period as usize,
            min_period as usize,
            n as usize,
            *t0_ as usize,
            prev_period as usize,
            prev_gain,
        );
        *t0_ = t0 as c_int;
        gain
    }
}

// TODO: document this.
fn rs_remove_doubling(
    x: &[f32],
    mut max_period: usize,
    mut min_period: usize,
    mut n: usize,
    mut t0: usize,
    mut prev_period: usize,
    prev_gain: f32,
) -> (usize, f32) {
    let init_min_period = min_period;
    min_period /= 2;
    max_period /= 2;
    t0 /= 2;
    prev_period /= 2;
    n /= 2;
    t0 = t0.min(max_period - 1);

    let mut t = t0;

    // Note that because we can't index with negative numbers, the x in the C code is our
    // x[max_period..].
    // FIXME: allocation
    let mut yy_lookup = vec![0.0f32; max_period + 1];
    let xx = inner_prod(&x[max_period..], &x[max_period..], n);
    let mut xy = inner_prod(&x[max_period..], &x[(max_period - t0)..], n);
    yy_lookup[0] = xx;

    let mut yy = xx;
    for i in 1..=max_period {
        yy += x[max_period - i] * x[max_period - i] - x[max_period + n - i] * x[max_period + n - i];
        yy_lookup[i] = yy.max(0.0);
    }

    yy = yy_lookup[t0];
    let mut best_xy = xy;
    let mut best_yy = yy;

    let g0 = pitch_gain(xy, xx, yy);
    let mut g = g0;

    // Look for any pitch at T/k */
    for k in 2..=15 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < min_period {
            break;
        }
        // Look for another strong correlation at t1b
        let t1b = if k == 2 {
            if t1 + t0 > max_period {
                t0
            } else {
                t0 + t1
            }
        } else {
            (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
        };
        xy = inner_prod(&x[max_period..], &x[(max_period - t1)..], n);
        let xy2 = inner_prod(&x[max_period..], &x[(max_period - t1b)..], n);
        xy = (xy + xy2) / 2.0;
        yy = (yy_lookup[t1] + yy_lookup[t1b]) / 2.0;

        let g1 = pitch_gain(xy, xx, yy);
        let cont = if (t1 as isize - prev_period as isize).abs() <= 1 {
            prev_gain
        } else if (t1 as isize - prev_period as isize).abs() <= 2 && 5 * k * k < t0 {
            prev_gain / 2.0
        } else {
            0.0
        };

        // Bias against very high pitch (very short period) to avoid false-positives due to
        // short-term correlation.
        let thresh = if t1 < 3 * min_period {
            (0.85 * g0 - cont).max(0.4)
        } else if t1 < 2 * min_period {
            (0.9 * g0 - cont).max(0.5)
        } else {
            (0.7 * g0 - cont).max(0.3)
        };
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }

    let best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr = [0.0; 3];
    for k in 0..3 {
        xcorr[k] = inner_prod(&x[max_period..], &x[(max_period - (t + k - 1))..], n);
    }
    let offset: isize = if xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]) {
        1
    } else if xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]) {
        -1
    } else {
        0
    };

    let pg = pg.min(g);
    let t0 = (2 * t).wrapping_add(offset as usize).max(init_min_period);

    (t0, pg)
}

const FRAME_SIZE_SHIFT: usize = 2;
const FRAME_SIZE: usize = 120 << FRAME_SIZE_SHIFT;
const WINDOW_SIZE: usize = 2 * FRAME_SIZE;
const FREQ_SIZE: usize = FRAME_SIZE + 1;
pub const NB_BANDS: usize = 22;
const NB_DELTA_CEPS: usize = 6;
pub const NB_FEATURES: usize = NB_BANDS + 3 * NB_DELTA_CEPS + 2;
const EBAND_5MS: [usize; 22] = [
    // 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];
type Complex = num_complex::Complex<f32>;

#[no_mangle]
pub extern "C" fn compute_band_energy(band_e: *mut f32, x: *const Complex) {
    unsafe {
        let band_e_slice = std::slice::from_raw_parts_mut(band_e, NB_BANDS);
        let x_slice = std::slice::from_raw_parts(x, WINDOW_SIZE);
        rs_compute_band_corr(band_e_slice, x_slice, x_slice);
    }
}

#[no_mangle]
pub extern "C" fn compute_band_corr(band_e: *mut f32, x: *const Complex, p: *const Complex) {
    unsafe {
        let band_e_slice = std::slice::from_raw_parts_mut(band_e, NB_BANDS);
        let x_slice = std::slice::from_raw_parts(x, WINDOW_SIZE);
        let p_slice = std::slice::from_raw_parts(p, WINDOW_SIZE);
        rs_compute_band_corr(band_e_slice, x_slice, p_slice);
    }
}

fn rs_compute_band_corr(out: &mut [f32], x: &[Complex], p: &[Complex]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }

    for i in 0..(NB_BANDS - 1) {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) << FRAME_SIZE_SHIFT;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = (EBAND_5MS[i] << FRAME_SIZE_SHIFT) + j;
            let corr = x[idx].re * p[idx].re + x[idx].im * p[idx].im;
            out[i] += (1.0 - frac) * corr;
            out[i + 1] += frac * corr;
        }
    }
    out[0] *= 2.0;
    out[NB_BANDS - 1] *= 2.0;
}

#[no_mangle]
pub extern "C" fn interp_band_gain(g: *mut f32, band_e: *const f32) {
    unsafe {
        let g_slice = std::slice::from_raw_parts_mut(g, FREQ_SIZE);
        let band_e_slice = std::slice::from_raw_parts(band_e, NB_BANDS);
        rs_interp_band_gain(g_slice, band_e_slice);
    }
}

fn rs_interp_band_gain(out: &mut [f32], band_e: &[f32]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }

    for i in 0..(NB_BANDS - 1) {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) << FRAME_SIZE_SHIFT;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = (EBAND_5MS[i] << FRAME_SIZE_SHIFT) + j;
            out[idx] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

struct CommonState {
    half_window: [f32; FRAME_SIZE],
    dct_table: [f32; NB_BANDS * NB_BANDS],
}

static COMMON: OnceCell<CommonState> = OnceCell::new();

fn common() -> &'static CommonState {
    if COMMON.get().is_none() {
        let pi = std::f64::consts::PI;
        let mut half_window = [0.0; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            let sin = (0.5 * pi * (i as f64 + 0.5) / FRAME_SIZE as f64).sin();
            half_window[i] = (0.5 * pi * sin * sin).sin() as f32;
        }

        let mut dct_table = [0.0; NB_BANDS * NB_BANDS];
        for i in 0..NB_BANDS {
            for j in 0..NB_BANDS {
                dct_table[i * NB_BANDS + j] =
                    ((i as f64 + 0.5) * j as f64 * pi / NB_BANDS as f64).cos() as f32;
                if j == 0 {
                    dct_table[i * NB_BANDS + j] *= 0.5f32.sqrt();
                }
            }
        }
        let _ = COMMON.set(CommonState {
            half_window,
            dct_table,
        });
    }
    COMMON.get().unwrap()
}

#[no_mangle]
pub extern "C" fn dct(out: *mut f32, input: *const f32) {
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out, NB_BANDS);
        let in_slice = std::slice::from_raw_parts(input, NB_BANDS);
        rs_dct(out_slice, in_slice);
    }
}

/// A brute-force DCT (discrete cosine transform) of size NB_BANDS.
fn rs_dct(out: &mut [f32], x: &[f32]) {
    let c = common();
    for i in 0..NB_BANDS {
        let mut sum = 0.0;
        for j in 0..NB_BANDS {
            sum += x[j] * c.dct_table[j * NB_BANDS + i];
        }
        out[i] = (sum as f64 * (2.0 / NB_BANDS as f64).sqrt()) as f32;
    }
}

#[no_mangle]
pub extern "C" fn apply_window(x: *mut f32) {
    unsafe {
        let x_slice = std::slice::from_raw_parts_mut(x, WINDOW_SIZE);
        rs_apply_window(x_slice);
    }
}

fn rs_apply_window(x: &mut [f32]) {
    let c = common();
    for i in 0..FRAME_SIZE {
        x[i] *= c.half_window[i];
        x[WINDOW_SIZE - 1 - i] *= c.half_window[i];
    }
}

//
