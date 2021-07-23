use std::error::Error;
use std::fmt;

pub enum CryptoAPIError {
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
impl fmt::Display for CryptoAPIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CryptoAPIError::PolynomialSizeError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::NoNoiseInCiphertext { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::DimensionError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotEnoughPaddingError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::InvalidEncoderError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::MessageOutsideIntervalError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::MessageTooBigError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::DeltaError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::PaddingError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::IndexError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::ConstantMaximumError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::ZeroInIntervalError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::NbCTError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::PrecisionError { description } => writeln!(f, "\n{}", description),
            CryptoAPIError::MinMaxError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::RadiusError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::MonomialError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotPowerOfTwoError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::ZeroCiphertextsInStructureError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::WrongSizeError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotEnoughValidEncoderError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::LweToRlweError { description, .. } => writeln!(f, "\n{}", description),
        }
    }
}

impl fmt::Debug for CryptoAPIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CryptoAPIError::PolynomialSizeError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::NoNoiseInCiphertext { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::DimensionError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotEnoughPaddingError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::InvalidEncoderError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::MessageOutsideIntervalError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::MessageTooBigError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::DeltaError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::PaddingError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::IndexError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::ConstantMaximumError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::ZeroInIntervalError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::NbCTError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::PrecisionError { description } => writeln!(f, "\n{}", description),
            CryptoAPIError::MinMaxError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::RadiusError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::MonomialError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotPowerOfTwoError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::ZeroCiphertextsInStructureError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::WrongSizeError { description, .. } => writeln!(f, "\n{}", description),
            CryptoAPIError::NotEnoughValidEncoderError { description, .. } => {
                writeln!(f, "\n{}", description)
            }
            CryptoAPIError::LweToRlweError { description, .. } => writeln!(f, "\n{}", description),
        }
    }
}

impl Error for CryptoAPIError {
    fn description(&self) -> &str {
        match self {
            CryptoAPIError::PolynomialSizeError { description, .. } => description,
            CryptoAPIError::NoNoiseInCiphertext { description, .. } => description,
            CryptoAPIError::DimensionError { description, .. } => description,
            CryptoAPIError::NotEnoughPaddingError { description, .. } => description,
            CryptoAPIError::InvalidEncoderError { description, .. } => description,
            CryptoAPIError::MessageOutsideIntervalError { description, .. } => description,
            CryptoAPIError::MessageTooBigError { description, .. } => description,
            CryptoAPIError::DeltaError { description, .. } => description,
            CryptoAPIError::PaddingError { description, .. } => description,
            CryptoAPIError::IndexError { description, .. } => description,
            CryptoAPIError::ConstantMaximumError { description, .. } => description,
            CryptoAPIError::ZeroInIntervalError { description, .. } => description,
            CryptoAPIError::NbCTError { description, .. } => description,
            CryptoAPIError::PrecisionError { description } => description,
            CryptoAPIError::MinMaxError { description, .. } => description,
            CryptoAPIError::RadiusError { description, .. } => description,
            CryptoAPIError::MonomialError { description, .. } => description,
            CryptoAPIError::NotPowerOfTwoError { description, .. } => description,
            CryptoAPIError::ZeroCiphertextsInStructureError { description, .. } => description,
            CryptoAPIError::WrongSizeError { description, .. } => description,
            CryptoAPIError::NotEnoughValidEncoderError { description, .. } => description,
            CryptoAPIError::LweToRlweError { description, .. } => description,
        }
    }
}

#[macro_export]
macro_rules! PolynomialSizeError {
    ($size_1: expr, $size_2: expr) => {
        CryptoAPIError::PolynomialSizeError {
            size_1: $size_1,
            size_2: $size_2,
            description: format!(
                "{}: {} != {} \n{:#?}\n ",
                "Different polynomial sizes: ".red().bold(),
                $size_1,
                $size_2,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! NoNoiseInCiphertext {
    ($var: expr) => {
        CryptoAPIError::NoNoiseInCiphertext {
            var: $var,
            description: format!(
                "{} {} {} \n{:#?}\n ",
                "The integer representation has not enough precision to represent error samples from the normal law of variance".red().bold(),
                $var,
                "so the ciphertext does not contain any noise!\n{:#?}\n",
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! DimensionError {
    ($dim_1: expr, $dim_2:expr) => {
        CryptoAPIError::DimensionError {
            dim_1: $dim_1,
            dim_2: $dim_2,
            description: format!(
                "{}: {} != {}\n{:#?}\n",
                "Different dimensions".red().bold(),
                $dim_1,
                $dim_2,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! InvalidEncoderError {
    ($nb_bit_precision: expr, $delta: expr) => {
        CryptoAPIError::InvalidEncoderError {
            nb_bit_precision: $nb_bit_precision,
            delta: $delta,
            description: format!(
                "{}: nb_bit_precision = {}, delta = {}\n{:#?}\n",
                "Invalid Encoder".red().bold(),
                $nb_bit_precision,
                $delta,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! MessageOutsideIntervalError {
    ($message: expr, $o: expr, $delta: expr) => {
        CryptoAPIError::MessageOutsideIntervalError {
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
        }
    };
}

#[macro_export]
macro_rules! MessageTooBigError {
    ($message: expr, $delta: expr) => {
        CryptoAPIError::MessageTooBigError {
            message: $message,
            delta: $delta,
            description: format!(
                "The absolute value of the message {} is {} = {}\n{:#?}\n",
                "bigger than delta".red().bold(),
                $message.abs(),
                $delta,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! DeltaError {
    ($delta_1: expr, $delta_2: expr) => {
        CryptoAPIError::DeltaError {
            delta_1: $delta_1,
            delta_2: $delta_2,
            description: format!(
                "{} : {} != {}\n{:#?}\n",
                "Deltas should be the same".red().bold(),
                $delta_1,
                $delta_2,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! PaddingError {
    ($p_1: expr, $p_2: expr) => {
        CryptoAPIError::PaddingError {
            p_1: $p_1,
            p_2: $p_2,
            description: format!(
                "{}: {} != {}\n{:#?}\n",
                "Number of bits of padding should be the same".red().bold(),
                $p_1,
                $p_2,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! NotEnoughPaddingError {
    ($p: expr, $min_p: expr) => {
        CryptoAPIError::NotEnoughPaddingError {
            p: $p,
            min_p: $min_p,
            description: format!(
                "{} we need at least {} bits of padding, and we only have {}\n{:#?}\n",
                "Not enough padding:".red().bold(),
                $min_p,
                $p,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! IndexError {
    ($nb_ct: expr, $n: expr) => {
        CryptoAPIError::IndexError {
            nb_ct: $nb_ct,
            n: $n,
            description: format!(
                "{}: number of ciphertexts = {} <= index = {}\n{:#?}\n",
                "Can't access the ciphertext".red().bold(),
                $nb_ct,
                $n,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! ConstantMaximumError {
    ($cst: expr, $max: expr) => {
        CryptoAPIError::ConstantMaximumError {
            cst: $cst,
            max: $max,
            description: format!(
                "Absolute value of the constant (= {}) is {} (= {})\n{:#?}\n",
                $cst,
                "bigger than the maximum".red().bold(),
                $max,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! ZeroInIntervalError {
    ($o: expr, $delta: expr) => {
        CryptoAPIError::ZeroInIntervalError {
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
        }
    };
}

#[macro_export]
macro_rules! NbCTError {
    ($nb_ct1: expr, $nb_ct2: expr) => {
        CryptoAPIError::NbCTError {
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
        }
    };
}

#[macro_export]
macro_rules! PrecisionError {
    () => {
        CryptoAPIError::PrecisionError {
            description: format!(
                "{}\n{:?}\n",
                "Number of bit for precision == 0".red().bold(),
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! MinMaxError {
    ($min: expr, $max: expr) => {
        CryptoAPIError::MinMaxError {
            min: $min,
            max: $max,
            description: format!(
                "min (= {}) <= max (= {})\n{:#?}\n",
                $min,
                $max,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! RadiusError {
    ($radius: expr) => {
        CryptoAPIError::RadiusError {
            radius: $radius,
            description: format!(
                "{}: {}\n{:#?}\n",
                "Invalid radius".red().bold(),
                $radius,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! MonomialError {
    ($polynomial_size: expr, $monomial: expr) => {
        CryptoAPIError::MonomialError {
            polynomial_size: $polynomial_size,
            monomial: $monomial,
            description: format!(
                "{}: polynomial_size (= {}) <= monomial index (= {})\n{:#?}\n",
                "Can't access the monomial coefficient".red().bold(),
                $polynomial_size,
                $monomial,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! NotPowerOfTwoError {
    ($polynomial_size: expr) => {
        CryptoAPIError::NotPowerOfTwoError {
            polynomial_size: $polynomial_size,
            description: format!(
                "polynomial_size (= {}) {}\n{:#?}\n",
                $polynomial_size,
                "must be a power of 2".red().bold(),
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! ZeroCiphertextsInStructureError {
    ($nb_ciphertexts: expr) => {
        CryptoAPIError::ZeroCiphertextsInStructureError {
            nb_ciphertexts: $nb_ciphertexts,
            description: format!(
                "There should be {}: nb_ciphertexts provided is {}\n{:#?}\n",
                "at least one ciphertext in the structure".red().bold(),
                $nb_ciphertexts,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! WrongSizeError {
    ($size: expr) => {
        CryptoAPIError::WrongSizeError {
            size: $size,
            description: format!(
                "The {}: {}\n{:#?}\n",
                "size is wrong".red().bold(),
                $size,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! NotEnoughValidEncoderError {
    ($nb_valid_encoders: expr,$nb_actions:expr) => {
        CryptoAPIError::NotEnoughValidEncoderError {
            nb_valid_encoders: $nb_valid_encoders,
            nb_actions: $nb_actions,
            description: format!(
                "There are only {} {} but it was asked to work on {}\n{:#?}\n",
                $nb_valid_encoders,
                "valid encoders".red().bold(),
                $nb_actions,
                Backtrace::new()
            ),
        }
    };
}

#[macro_export]
macro_rules! LweToRlweError {
    ($dimension: expr,$polynomial_size:expr) => {
        CryptoAPIError::LweToRlweError {
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
        }
    };
}
