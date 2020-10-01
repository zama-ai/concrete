use std::error::Error;
use std::fmt;

pub enum ProAPIError {
    PolynomialSizeError {
        size_1: usize,
        size_2: usize,
        description: String,
    },
    NoNoiseInCiphertext {
        var: f64,
        description: String,
    },
    DimensionError {
        dim_1: usize,
        dim_2: usize,
        description: String,
    },
    InvalidEncoderError {
        nb_bit_precision: usize,
        delta: f64,
        description: String,
    },
    MessageOutsideIntervalError {
        message: f64,
        o: f64,
        delta: f64,
        description: String,
    },
    MessageTooBigError {
        message: f64,
        delta: f64,
        description: String,
    },
    DeltaError {
        delta_1: f64,
        delta_2: f64,
        description: String,
    },
    PaddingError {
        p_1: usize,
        p_2: usize,
        description: String,
    },
    NotEnoughPaddingError {
        p: usize,
        min_p: usize,
        description: String,
    },
    IndexError {
        nb_ct: usize,
        n: usize,
        description: String,
    },
    ConstantMaximumError {
        cst: f64,
        max: f64,
        description: String,
    },
    ZeroInIntervalError {
        o: f64,
        delta: f64,
        description: String,
    },
    NbCTError {
        nb_ct1: usize,
        nb_ct2: usize,
        description: String,
    },
    PrecisionError {
        description: String,
    },
    MinMaxError {
        min: f64,
        max: f64,
        description: String,
    },
    RadiusError {
        radius: f64,
        description: String,
    },
    MonomialError {
        polynomial_size: usize,
        monomial: usize,
        description: String,
    },
    NotPowerOfTwoError {
        polynomial_size: usize,
        description: String,
    },
    ZeroCiphertextsInStructureError {
        nb_ciphertexts: usize,
        description: String,
    },
    WrongSizeError {
        size: usize,
        description: String,
    },
    NotEnoughValidEncoderError {
        nb_valid_encoders: usize,
        nb_actions: usize,
        description: String,
    },
    LweToRlweError {
        dimension: usize,
        polynomial_size: usize,
        description: String,
    },
}
impl fmt::Display for ProAPIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProAPIError::PolynomialSizeError {
                size_1: _,
                size_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NoNoiseInCiphertext {
                var: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::DimensionError {
                dim_1: _,
                dim_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotEnoughPaddingError {
                p: _,
                min_p: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::InvalidEncoderError {
                nb_bit_precision: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MessageOutsideIntervalError {
                message: _,
                o: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MessageTooBigError {
                message: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::DeltaError {
                delta_1: _,
                delta_2: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::PaddingError {
                p_1: _,
                p_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::IndexError {
                nb_ct: _,
                n: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::ConstantMaximumError {
                cst: _,
                max: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::ZeroInIntervalError {
                o: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NbCTError {
                nb_ct1: _,
                nb_ct2: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::PrecisionError { description } => writeln!(f, "\n{}", description),

            ProAPIError::MinMaxError {
                min: _,
                max: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::RadiusError {
                radius: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MonomialError {
                polynomial_size: _,
                monomial: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotPowerOfTwoError {
                polynomial_size: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::ZeroCiphertextsInStructureError {
                nb_ciphertexts: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::WrongSizeError {
                size: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotEnoughValidEncoderError {
                nb_valid_encoders: _,
                nb_actions: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::LweToRlweError {
                dimension: _,
                polynomial_size: _,
                description,
            } => writeln!(f, "\n{}", description),
        }
    }
}

impl fmt::Debug for ProAPIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProAPIError::PolynomialSizeError {
                size_1: _,
                size_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NoNoiseInCiphertext {
                var: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::DimensionError {
                dim_1: _,
                dim_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotEnoughPaddingError {
                p: _,
                min_p: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::InvalidEncoderError {
                nb_bit_precision: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MessageOutsideIntervalError {
                message: _,
                o: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MessageTooBigError {
                message: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::DeltaError {
                delta_1: _,
                delta_2: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::PaddingError {
                p_1: _,
                p_2: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::IndexError {
                nb_ct: _,
                n: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::ConstantMaximumError {
                cst: _,
                max: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::ZeroInIntervalError {
                o: _,
                delta: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NbCTError {
                nb_ct1: _,
                nb_ct2: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::PrecisionError { description } => writeln!(f, "\n{}", description),

            ProAPIError::MinMaxError {
                min: _,
                max: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::RadiusError {
                radius: _,
                description,
            } => writeln!(f, "\n{}", description),

            ProAPIError::MonomialError {
                polynomial_size: _,
                monomial: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotPowerOfTwoError {
                polynomial_size: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::ZeroCiphertextsInStructureError {
                nb_ciphertexts: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::WrongSizeError {
                size: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::NotEnoughValidEncoderError {
                nb_valid_encoders: _,
                nb_actions: _,
                description,
            } => writeln!(f, "\n{}", description),
            ProAPIError::LweToRlweError {
                dimension: _,
                polynomial_size: _,
                description,
            } => writeln!(f, "\n{}", description),
        }
    }
}

impl Error for ProAPIError {
    fn description(&self) -> &str {
        match self {
            ProAPIError::PolynomialSizeError {
                size_1: _,
                size_2: _,
                description,
            } => description,

            ProAPIError::NoNoiseInCiphertext {
                var: _,
                description,
            } => description,

            ProAPIError::DimensionError {
                dim_1: _,
                dim_2: _,
                description,
            } => description,

            ProAPIError::NotEnoughPaddingError {
                p: _,
                min_p: _,
                description,
            } => description,

            ProAPIError::InvalidEncoderError {
                nb_bit_precision: _,
                delta: _,
                description,
            } => description,

            ProAPIError::MessageOutsideIntervalError {
                message: _,
                o: _,
                delta: _,
                description,
            } => description,

            ProAPIError::MessageTooBigError {
                message: _,
                delta: _,
                description,
            } => description,

            ProAPIError::DeltaError {
                delta_1: _,
                delta_2: _,
                description,
            } => description,

            ProAPIError::PaddingError {
                p_1: _,
                p_2: _,
                description,
            } => description,

            ProAPIError::IndexError {
                nb_ct: _,
                n: _,
                description,
            } => description,

            ProAPIError::ConstantMaximumError {
                cst: _,
                max: _,
                description,
            } => description,

            ProAPIError::ZeroInIntervalError {
                o: _,
                delta: _,
                description,
            } => description,

            ProAPIError::NbCTError {
                nb_ct1: _,
                nb_ct2: _,
                description,
            } => description,

            ProAPIError::PrecisionError { description } => description,

            ProAPIError::MinMaxError {
                min: _,
                max: _,
                description,
            } => description,

            ProAPIError::RadiusError {
                radius: _,
                description,
            } => description,

            ProAPIError::MonomialError {
                polynomial_size: _,
                monomial: _,
                description,
            } => description,

            ProAPIError::NotPowerOfTwoError {
                polynomial_size: _,
                description,
            } => description,
            ProAPIError::ZeroCiphertextsInStructureError {
                nb_ciphertexts: _,
                description,
            } => description,
            ProAPIError::WrongSizeError {
                size: _,
                description,
            } => description,
            ProAPIError::NotEnoughValidEncoderError {
                nb_valid_encoders: _,
                nb_actions: _,
                description,
            } => description,
            ProAPIError::LweToRlweError {
                dimension: _,
                polynomial_size: _,
                description,
            } => description,
        }
    }
}

macro_rules! PolynomialSizeError {
    ($size_1: expr, $size_2: expr) => {
        ProAPIError::PolynomialSizeError {
            size_1: $size_1,
            size_2: $size_2,
            description: format!(
                "{}: {} != {} \n{:#?}\n ",
                "Different polynomial sizes: ".red().bold(),
                $size_1,
                $size_2,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! NoNoiseInCiphertext {
    ($var: expr) => {
        ProAPIError::NoNoiseInCiphertext {
            var: $var,
            description: format!(
                "{} {} {} \n{:#?}\n ",
                "The integer representation has not enough precision to represent error samples from the normal law of variance".red().bold(),
                $var,
                "so the ciphertext does not contain any noise!\n{:#?}\n",
                Backtrace::new()
            ),
        };
    };
}

macro_rules! DimensionError {
    ($dim_1: expr, $dim_2:expr) => {
        ProAPIError::DimensionError {
            dim_1: $dim_1,
            dim_2: $dim_2,
            description: format!(
                "{}: {} != {}\n{:#?}\n",
                "Different dimensions".red().bold(),
                $dim_1,
                $dim_2,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! InvalidEncoderError {
    ($nb_bit_precision: expr, $delta: expr) => {
        ProAPIError::InvalidEncoderError {
            nb_bit_precision: $nb_bit_precision,
            delta: $delta,
            description: format!(
                "{}: nb_bit_precision = {}, delta = {}\n{:#?}\n",
                "Invalid Encoder".red().bold(),
                $nb_bit_precision,
                $delta,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! MessageOutsideIntervalError {
    ($message: expr, $o: expr, $delta: expr) => {
        ProAPIError::MessageOutsideIntervalError {
            message: $message,
            o: $o,
            delta: $delta,
            description: format!(
                "The message {} is {} [{}, {}] defined by o = {} and delta = {}\n{:#?}\n",
                "outside the interval".red().bold(),
                $message,
                $o,
                $o + $delta,
                $o,
                $delta,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! MessageTooBigError {
    ($message: expr, $delta: expr) => {
        ProAPIError::MessageTooBigError {
            message: $message,
            delta: $delta,
            description: format!(
                "The absolute value of the message {} is {} = {}\n{:#?}\n",
                "bigger than delta".red().bold(),
                $message.abs(),
                $delta,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! DeltaError {
    ($delta_1: expr, $delta_2: expr) => {
        ProAPIError::DeltaError {
            delta_1: $delta_1,
            delta_2: $delta_2,
            description: format!(
                "{} : {} != {}\n{:#?}\n",
                "Deltas should be the same".red().bold(),
                $delta_1,
                $delta_2,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! PaddingError {
    ($p_1: expr, $p_2: expr) => {
        ProAPIError::PaddingError {
            p_1: $p_1,
            p_2: $p_2,
            description: format!(
                "{}: {} != {}\n{:#?}\n",
                "Number of bits of padding should be the same".red().bold(),
                $p_1,
                $p_2,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! NotEnoughPaddingError {
    ($p: expr, $min_p: expr) => {
        ProAPIError::NotEnoughPaddingError {
            p: $p,
            min_p: $min_p,
            description: format!(
                "{} we need at least {} bits of padding, and we only have {}\n{:#?}\n",
                "Not enough padding:".red().bold(),
                $min_p,
                $p,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! IndexError {
    ($nb_ct: expr, $n: expr) => {
        ProAPIError::IndexError {
            nb_ct: $nb_ct,
            n: $n,
            description: format!(
                "{}: number of ciphertexts = {} <= index = {}\n{:#?}\n",
                "Can't access the ciphertext".red().bold(),
                $nb_ct,
                $n,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! ConstantMaximumError {
    ($cst: expr, $max: expr) => {
        ProAPIError::ConstantMaximumError {
            cst: $cst,
            max: $max,
            description: format!(
                "Absolute value of the constant (= {}) is {} (= {})\n{:#?}\n",
                $cst,
                "bigger than the maximum".red().bold(),
                $max,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! ZeroInIntervalError {
    ($o: expr, $delta: expr) => {
        ProAPIError::ZeroInIntervalError {
            o: $o,
            delta: $delta,
            description: format!(
                "{} = [{},{}] with o = {}, delta = {}\n{:#?}\n",
                "Zero should be in the input interval".red().bold(),
                $o,
                $o + $delta,
                $o,
                $delta,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! NbCTError {
    ($nb_ct1: expr, $nb_ct2: expr) => {
        ProAPIError::NbCTError {
            nb_ct1: $nb_ct1,
            nb_ct2: $nb_ct2,
            description: format!(
                "{} : {} != {}\n{:#?}\n",
                "The number of constants and the number of ciphertexts must be the same"
                    .red()
                    .bold(),
                $nb_ct1,
                $nb_ct2,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! PrecisionError {
    () => {
        ProAPIError::PrecisionError {
            description: format!(
                "{}\n{:?}\n",
                "Number of bit for precision == 0".red().bold(),
                Backtrace::new()
            ),
        };
    };
}

macro_rules! MinMaxError {
    ($min: expr, $max: expr) => {
        ProAPIError::MinMaxError {
            min: $min,
            max: $max,
            description: format!(
                "min (= {}) <= max (= {})\n{:#?}\n",
                $min,
                $max,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! RadiusError {
    ($radius: expr) => {
        ProAPIError::RadiusError {
            radius: $radius,
            description: format!(
                "{}: {}\n{:#?}\n",
                "Invalid radius".red().bold(),
                $radius,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! MonomialError {
    ($polynomial_size: expr, $monomial: expr) => {
        ProAPIError::MonomialError {
            polynomial_size: $polynomial_size,
            monomial: $monomial,
            description: format!(
                "{}: polynomial_size (= {}) <= monomial index (= {})\n{:#?}\n",
                "Can't access the monomial coefficient".red().bold(),
                $polynomial_size,
                $monomial,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! NotPowerOfTwoError {
    ($polynomial_size: expr) => {
        ProAPIError::NotPowerOfTwoError {
            polynomial_size: $polynomial_size,
            description: format!(
                "polynomial_size (= {}) {}\n{:#?}\n",
                $polynomial_size,
                "must be a power of 2".red().bold(),
                Backtrace::new()
            ),
        };
    };
}

macro_rules! ZeroCiphertextsInStructureError {
    ($nb_ciphertexts: expr) => {
        ProAPIError::ZeroCiphertextsInStructureError {
            nb_ciphertexts: $nb_ciphertexts,
            description: format!(
                "There should be {}: nb_ciphertexts provided is {}\n{:#?}\n",
                "at least one ciphertext in the structure".red().bold(),
                $nb_ciphertexts,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! WrongSizeError {
    ($size: expr) => {
        ProAPIError::WrongSizeError {
            size: $size,
            description: format!(
                "The {}: {}\n{:#?}\n",
                "size is wrong".red().bold(),
                $size,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! NotEnoughValidEncoderError {
    ($nb_valid_encoders: expr,$nb_actions:expr) => {
        ProAPIError::NotEnoughValidEncoderError {
            nb_valid_encoders: $nb_valid_encoders,
            nb_actions: $nb_actions,
            description: format!(
                "There are only {} {} but it was asked to work on {}\n{:#?}\n",
                $nb_valid_encoders,
                "valid encoders".red().bold(),
                $nb_actions,
                Backtrace::new()
            ),
        };
    };
}

macro_rules! LweToRlweError {
    ($dimension: expr,$polynomial_size:expr) => {
        ProAPIError::LweToRlweError {
            dimension: $dimension,
            polynomial_size: $polynomial_size,
            description: format!(
                "{} with dimension = {} {} with polynomial_size = {}\n{:#?}\n",
                "Can't cast a Lwe".red().bold(),
                $dimension,
                "into a Rlwe".red().bold(),
                $polynomial_size,
                Backtrace::new()
            ),
        };
    };
}
