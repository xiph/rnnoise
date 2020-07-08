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