pub fn combine_errors(p_error1: f64, p_error2: f64) -> f64 {
    // (1 - p_error) = (1 - p_error1) * (1 - p_error2)
    p_error1 + p_error2 - p_error1 * p_error2
}

pub fn repeat_p_error(p_error: f64, count: u64) -> f64 {
    if p_error * count as f64 > 1. {
        iterative_repeat_p_error(p_error, count)
    } else {
        binomial_decomposition_repeat_p_error(p_error, count)
    }
}

// (1 - global_p_error) = (1 - p_error)^count
// global_p_error = 1 - (1-p)^N = 1 - (1 - N p + N(N-1)/2 p^2 - N(N-1)(N-2)/(2*3) p^3...)
// global_p_error = N p - N(N-1)/2 p^2 + N(N-1)(N-2)/(2*3) p^3...
fn binomial_decomposition_repeat_p_error(p_error: f64, count: u64) -> f64 {
    // This guarantees abs(factor) is decreasing
    // Without that, factors grow and lose precision
    assert!(p_error * (count as f64) <= 1.0);

    let mut global_p_error = 0.0;

    let mut factor = -1.0;

    for i in 1..=count {
        factor *= -p_error * (count - i + 1) as f64 / i as f64;

        let new_global_p_error = global_p_error + factor;

        #[allow(clippy::float_cmp)]
        //if factor is too small to make a difference
        if new_global_p_error == global_p_error {
            // abs(factor) is decreasing and factor sign alternates
            // so the remaining series is bounded (in absolute value) by abs(factor) which makes no difference
            break;
        }

        global_p_error = new_global_p_error;
    }

    global_p_error
}

fn iterative_repeat_p_error(p_error: f64, count: u64) -> f64 {
    let mut global_p_error = 0.0;

    for _ in 0..count {
        global_p_error = combine_errors(global_p_error, p_error);
    }

    global_p_error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::float_cmp)]
    fn assert_eq_both_repeat_p_error(p_error: f64, count: u64) {
        let iterative = iterative_repeat_p_error(p_error, count);
        let binomial = binomial_decomposition_repeat_p_error(p_error, count);

        assert!(((iterative - binomial) / (iterative + binomial)).abs() < 0.000_000_1);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_repeat_p_error() {
        assert_eq!(repeat_p_error(0.5, 1), 0.5);
        assert_eq!(repeat_p_error(0.5, 2), 0.75);

        assert_eq_both_repeat_p_error(0.00001, 10000);

        assert_eq_both_repeat_p_error(0.001, 100);

        assert_eq_both_repeat_p_error(0.000_000_000_01, 100);
    }
}
