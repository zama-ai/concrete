pub fn combine_errors(p_error1: f64, p_error2: f64) -> f64 {
    // (1 - p_error) = (1 - p_error1) * (1 - p_error2)
    p_error1 + p_error2 - p_error1 * p_error2
}

pub fn repeat_p_error(p_error: f64, count: u64) -> f64 {
    let mut global_p_error = 0.0;

    for _ in 0..count {
        global_p_error = combine_errors(global_p_error, p_error);
    }

    global_p_error
}
