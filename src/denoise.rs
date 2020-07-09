use libc::c_int;

use crate::{
    Complex, CEPS_MEM, FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_DELTA_CEPS, NB_FEATURES, PITCH_BUF_SIZE,
    PITCH_FRAME_SIZE, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, WINDOW_SIZE,
};

#[repr(C)]
pub struct DenoiseState {
    analysis_mem: [f32; FRAME_SIZE],
    /// This is some sort of ring buffer, storing the last bunch of cepstra.
    cepstral_mem: [[f32; crate::NB_BANDS]; crate::CEPS_MEM],
    /// The index pointing to the most recent cepstrum in `cepstral_mem`. The previous cepstra are
    /// at indices mem_id - 1, mem_id - 1, etc (wrapped appropriately).
    mem_id: c_int,
    synthesis_mem: [f32; FRAME_SIZE],
    pitch_buf: [f32; crate::PITCH_BUF_SIZE],
    pitch_enh_buf: [f32; crate::PITCH_BUF_SIZE],
    last_gain: f32,
    last_period: c_int,
    mem_hp_x: [f32; 2],
    lastg: [f32; crate::NB_BANDS],
    rnn: crate::rnn::RnnState,
}

#[no_mangle]
pub extern "C" fn frame_analysis(
    st: *mut DenoiseState,
    x: *mut Complex,
    ex: *mut f32,
    input: *const f32,
) {
    unsafe {
        let x_slice = std::slice::from_raw_parts_mut(x, FREQ_SIZE);
        let ex_slice = std::slice::from_raw_parts_mut(ex, crate::NB_BANDS);
        let input_slice = std::slice::from_raw_parts(input, FRAME_SIZE);
        rs_frame_analysis(&mut *st, x_slice, ex_slice, input_slice);
    }
}

fn rs_frame_analysis(state: &mut DenoiseState, x: &mut [Complex], ex: &mut [f32], input: &[f32]) {
    let mut buf = [0.0; WINDOW_SIZE];
    for i in 0..FRAME_SIZE {
        buf[i] = state.analysis_mem[i];
    }
    for i in 0..crate::FRAME_SIZE {
        buf[i + crate::FRAME_SIZE] = input[i];
        state.analysis_mem[i] = input[i];
    }
    crate::rs_apply_window(&mut buf[..]);
    crate::rs_forward_transform(x, &buf[..]);
    crate::rs_compute_band_corr(ex, x, x);
}

#[no_mangle]
pub extern "C" fn compute_frame_features(
    st: *mut DenoiseState,
    x: *mut Complex,
    p: *mut Complex,
    ex: *mut f32,
    ep: *mut f32,
    exp: *mut f32,
    features: *mut f32,
    input: *const f32,
) {
    unsafe {
        let x = std::slice::from_raw_parts_mut(x, FREQ_SIZE);
        // Why WINDOW_SIZE and not FREQ_SIZE?
        let p = std::slice::from_raw_parts_mut(p, WINDOW_SIZE);
        let ex = std::slice::from_raw_parts_mut(ex, NB_BANDS);
        let ep = std::slice::from_raw_parts_mut(ep, NB_BANDS);
        let exp = std::slice::from_raw_parts_mut(exp, NB_BANDS);
        let features = std::slice::from_raw_parts_mut(features, NB_FEATURES);
        let input = std::slice::from_raw_parts(input, FRAME_SIZE);
        rs_compute_frame_features(&mut *st, x, p, ex, ep, exp, features, input);
    }
}

fn rs_compute_frame_features(
    state: &mut DenoiseState,
    x: &mut [Complex],
    p: &mut [Complex],
    ex: &mut [f32],
    ep: &mut [f32],
    exp: &mut [f32],
    features: &mut [f32],
    input: &[f32],
) -> usize {
    let mut ly = [0.0; NB_BANDS];
    let mut p_buf = [0.0; WINDOW_SIZE];
    // Apparently, PITCH_BUF_SIZE wasn't the best name...
    let mut pitch_buf = [0.0; PITCH_BUF_SIZE / 2];
    let mut tmp = [0.0; NB_BANDS];

    rs_frame_analysis(state, x, ex, input);
    for i in 0..(PITCH_BUF_SIZE - FRAME_SIZE) {
        state.pitch_buf[i] = state.pitch_buf[i + FRAME_SIZE];
    }
    for i in 0..FRAME_SIZE {
        state.pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE + i] = input[i];
    }

    crate::rs_pitch_downsample(&state.pitch_buf[..], &mut pitch_buf);
    let pitch_idx = crate::rs_pitch_search(
        &pitch_buf[(PITCH_MAX_PERIOD / 2)..],
        &pitch_buf,
        PITCH_FRAME_SIZE,
        PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD,
    );
    let pitch_idx = PITCH_MAX_PERIOD - pitch_idx;

    let (pitch_idx, gain) = crate::rs_remove_doubling(
        &pitch_buf[..],
        PITCH_MAX_PERIOD,
        PITCH_MIN_PERIOD,
        PITCH_FRAME_SIZE,
        pitch_idx,
        state.last_period as usize,
        state.last_gain,
    );
    state.last_period = pitch_idx as i32;
    state.last_gain = gain;

    for i in 0..WINDOW_SIZE {
        p_buf[i] = state.pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_idx + i];
    }
    crate::rs_apply_window(&mut p_buf[..]);
    crate::rs_forward_transform(p, &p_buf[..]);
    crate::rs_compute_band_corr(ep, p, p);
    crate::rs_compute_band_corr(exp, x, p);
    for i in 0..NB_BANDS {
        exp[i] /= (0.001 + ex[i] * ep[i]).sqrt();
    }
    crate::rs_dct(&mut tmp[..], exp);
    for i in 0..NB_DELTA_CEPS {
        features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
    }

    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);
    let mut log_max = -2.0;
    let mut follow = -2.0;
    let mut e = 0.0;
    for i in 0..NB_BANDS {
        ly[i] = (1e-2 + ex[i]).log10().max(log_max - 7.0).max(follow - 1.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 1.5).max(ly[i]);
        e += ex[i];
    }

    if e < 0.04 {
        /* If there's no audio, avoid messing up the state. */
        for i in 0..NB_FEATURES {
            features[i] = 0.0;
        }
        return 1;
    }
    crate::rs_dct(features, &ly[..]);
    features[0] -= 12.0;
    features[1] -= 4.0;
    let ceps_0_idx = state.mem_id as usize;
    let ceps_1_idx = if state.mem_id < 1 {
        CEPS_MEM + state.mem_id as usize - 1
    } else {
        state.mem_id as usize - 1
    };
    let ceps_2_idx = if state.mem_id < 2 {
        CEPS_MEM + state.mem_id as usize - 2
    } else {
        state.mem_id as usize - 2
    };

    for i in 0..NB_BANDS {
        state.cepstral_mem[ceps_0_idx][i] = features[i];
    }
    state.mem_id += 1;

    let ceps_0 = &state.cepstral_mem[ceps_0_idx];
    let ceps_1 = &state.cepstral_mem[ceps_1_idx];
    let ceps_2 = &state.cepstral_mem[ceps_2_idx];
    for i in 0..NB_DELTA_CEPS {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0 * ceps_1[i] + ceps_2[i];
    }

    /* Spectral variability features. */
    let mut spec_variability = 0.0;
    if state.mem_id == CEPS_MEM as i32 {
        state.mem_id = 0;
    }
    for i in 0..CEPS_MEM as usize {
        let mut min_dist = 1e15f32;
        for j in 0..CEPS_MEM as usize {
            let mut dist = 0.0;
            for k in 0..NB_BANDS {
                let tmp = state.cepstral_mem[i][k] - state.cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if j != i {
                min_dist = min_dist.min(dist);
            }
        }
        spec_variability += min_dist;
    }

    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;

    return 0;
}

#[no_mangle]
pub extern "C" fn frame_synthesis(state: *mut DenoiseState, out: *mut f32, y: *const Complex) {
    unsafe {
        let out = std::slice::from_raw_parts_mut(out, WINDOW_SIZE);
        let y = std::slice::from_raw_parts(y, FREQ_SIZE);
        rs_frame_synthesis(&mut *state, out, y);
    }
}

fn rs_frame_synthesis(state: &mut DenoiseState, out: &mut [f32], y: &[Complex]) {
    let mut x = [0.0; WINDOW_SIZE];
    crate::rs_inverse_transform(&mut x[..], y);
    crate::rs_apply_window(&mut x[..]);
    for i in 0..FRAME_SIZE {
        out[i] = x[i] + state.synthesis_mem[i];
        state.synthesis_mem[i] = x[FRAME_SIZE + i];
    }
}

fn rs_biquad(y: &mut [f32], mem: &mut [f32], x: &[f32], b: &[f32], a: &[f32]) {
    for i in 0..x.len() {
        let xi = x[i] as f64;
        let yi = (x[i] + mem[0]) as f64;
        mem[0] = (mem[1] as f64 + (b[0] as f64 * xi - a[0] as f64 * yi)) as f32;
        mem[1] = (b[1] as f64 * xi - a[1] as f64 * yi) as f32;
        y[i] = yi as f32;
    }
}

#[no_mangle]
pub extern "C" fn biquad(
    y: *mut f32,
    mem: *mut f32,
    x: *const f32,
    b: *const f32,
    a: *const f32,
    n: c_int,
) {
    unsafe {
        let y = std::slice::from_raw_parts_mut(y, n as usize);
        let x = std::slice::from_raw_parts(x, n as usize);
        let mem = std::slice::from_raw_parts_mut(mem, 2);
        let b = std::slice::from_raw_parts(b, 2);
        let a = std::slice::from_raw_parts(a, 2);
        rs_biquad(y, mem, x, b, a);
    }
}

fn rs_pitch_filter(
    x: &mut [Complex],
    p: &mut [Complex],
    ex: &[f32],
    ep: &[f32],
    exp: &[f32],
    g: &[f32],
) {
    let mut r = [0.0; NB_BANDS];
    let mut rf = [0.0; FREQ_SIZE];
    for i in 0..NB_BANDS {
        r[i] = if exp[i] > g[i] {
            1.0
        } else {
            let exp_sq = exp[i] * exp[i];
            let g_sq = g[i] * g[i];
            exp_sq * (1.0 - g_sq) / (0.001 + g_sq * (1.0 - exp_sq))
        };
        r[i] = 1.0_f32.min(0.0_f32.max(r[i])).sqrt();
        r[i] *= (ex[i] / (1e-8 + ep[i])).sqrt();
    }
    crate::rs_interp_band_gain(&mut rf[..], &r[..]);
    for i in 0..FREQ_SIZE {
        x[i] += rf[i] * p[i];
    }

    let mut new_e = [0.0; NB_BANDS];
    crate::rs_compute_band_corr(&mut new_e[..], x, x);
    let mut norm = [0.0; NB_BANDS];
    let mut normf = [0.0; FREQ_SIZE];
    for i in 0..NB_BANDS {
        norm[i] = (ex[i] / (1e-8 + new_e[i])).sqrt();
    }
    crate::rs_interp_band_gain(&mut normf[..], &norm[..]);
    for i in 0..FREQ_SIZE {
        x[i] *= normf[i];
    }
}

#[no_mangle]
pub extern "C" fn pitch_filter(
    x: *mut Complex,
    p: *mut Complex,
    ex: *const f32,
    ep: *const f32,
    exp: *const f32,
    g: *const f32,
) {
    unsafe {
        let x = std::slice::from_raw_parts_mut(x, FREQ_SIZE);
        let p = std::slice::from_raw_parts_mut(p, FREQ_SIZE);
        let ex = std::slice::from_raw_parts(ex, NB_BANDS);
        let ep = std::slice::from_raw_parts(ep, NB_BANDS);
        let exp = std::slice::from_raw_parts(exp, NB_BANDS);
        let g = std::slice::from_raw_parts(g, NB_BANDS);
        rs_pitch_filter(x, p, ex, ep, exp, g);
    }
}

#[no_mangle]
pub extern "C" fn rnnoise_process_frame(
    state: *mut DenoiseState,
    output: *mut f32,
    input: *const f32,
) -> f32 {
    unsafe {
        let output = std::slice::from_raw_parts_mut(output, FRAME_SIZE);
        let input = std::slice::from_raw_parts(input, FRAME_SIZE);
        rs_process_frame(&mut *state, output, input)
    }
}

fn rs_process_frame(state: &mut DenoiseState, output: &mut [f32], input: &[f32]) -> f32 {
    let mut x_freq = [Complex::from(0.0); FREQ_SIZE];
    let mut p = [Complex::from(0.0); WINDOW_SIZE];
    let mut x_time = [0.0; FRAME_SIZE];
    let mut ex = [0.0; NB_BANDS];
    let mut ep = [0.0; NB_BANDS];
    let mut exp = [0.0; NB_BANDS];
    let mut features = [0.0; NB_FEATURES];
    let mut g = [0.0; NB_BANDS];
    let mut gf = [1.0; FREQ_SIZE];
    let a_hp = [-1.99599, 0.99600];
    let b_hp = [-2.0, 1.0];
    let mut vad_prob = [0.0];

    rs_biquad(
        &mut x_time[..],
        &mut state.mem_hp_x[..],
        input,
        &b_hp[..],
        &a_hp[..],
    );
    let silence = rs_compute_frame_features(
        state,
        &mut x_freq[..],
        &mut p[..],
        &mut ex[..],
        &mut ep[..],
        &mut exp[..],
        &mut features[..],
        &x_time[..],
    );
    if silence == 0 {
        crate::rnn::rs_compute_rnn(&mut state.rnn, &mut g[..], &mut vad_prob[..], &features[..]);
        rs_pitch_filter(
            &mut x_freq[..],
            &mut p[..],
            &mut ex[..],
            &mut ep[..],
            &mut exp[..],
            &mut g[..],
        );
        for i in 0..NB_BANDS {
            g[i] = g[i].max(0.6 * state.lastg[i]);
            state.lastg[i] = g[i];
        }
        crate::rs_interp_band_gain(&mut gf[..], &g[..]);
        for i in 0..FREQ_SIZE {
            x_freq[i] *= gf[i];
        }
    }

    rs_frame_synthesis(state, output, &x_freq[..]);
    vad_prob[0]
}
