//! Secret Key Tensor Operations
//! * Contains every function only related to secret key tensors
//! * Keys are only binary for now
//! * Keys are stored into slices of Torus elements

use crate::Types;

pub trait SecretKey: Sized {
    fn get_bit(binary_key: &[Self], n: usize) -> bool;
    fn get_bit_monomial(
        binary_key: &[Self],
        polynomial_size: usize,
        mask_index: usize,
        monomial_index: usize,
    ) -> bool;
    fn convert_key_to_boolean_slice(boolean_slice: &mut [bool], sk: &[Self], sk_nb_bits: usize);
    fn print(sk: &[Self], dimension: usize, polynomial_size: usize);
    fn print_ring(sk: &[Self], dimension: usize, polynomial_size: usize);
    fn get_secret_key_length(dimension: usize, polynomial_size: usize) -> usize;
}

macro_rules! impl_trait_secret_key {
    ($T:ty,$DOC:expr) => {
        impl SecretKey for $T {
            /// Returns the n-th bit (a boolean) of a binary key viewed as a key for a LWE sample
            /// # Arguments
            /// * `binary_key` - a Torus slice representing the binary key
            /// * `n` - the index of the wanted bit
            /// # Output
            /// * the n-th bit of k as a boolean
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::SecretKey;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of a secret key
            /// let mut sk: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the secret key with random)
            ///
            /// // we want the 7-th (counting from zero) bit of the scret key
            /// let bit: bool = SecretKey::get_bit(&sk, 7);
            /// ```
            fn get_bit(binary_key: &[$T], n: usize) -> bool {
                // finds the right case of the slice
                let i: usize = (n / <$T as Types>::TORUS_BIT) as usize;
                let cell: $T = binary_key[i];

                // finds the right bit in the Torus element
                let j: usize = (n % <$T as Types>::TORUS_BIT) as usize;
                let bit: $T = (cell >> (((<$T as Types>::TORUS_BIT - 1) as $T) - j as $T)) & 1;

                // returns a boolean
                return bit == 1;
            }

            /// Returns the bit coefficient of the monomial of degree monomial_degree, from the mask_index-th polynomial in the binary polynomial of the RLWE secret key binary_key
            /// # Example
            /// * the binary polynomial RLWE secret key binary_key is: (0 + 1*X, 1 + 1*X, 1 + 0*X, 0 + 1*X)
            /// * we can see that polynomial_size := 2
            /// * get_bit_monomial(binary_key,polynomial_size,3,0) outputs False because binary_key\[3\]=0 + 1*X, and the coefficient of X^0 is 0
            /// # Arguments
            /// * `binary_key` - a Torus slice representing a binary polynomial RLWE secret key
            /// * `polynomial_size` - the number of coefficients in polynomials
            /// * `mask_index` - a mask index
            /// * `monomial_degree` - a degree of a monomial
            /// # Output
            /// * the bit coefficient (as a boolean) of the degree monomial_degree monomial of the mask_index-th polynomial in binary_key
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::SecretKey;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            /// let polynomial_size: usize = 16;
            ///
            /// // allocation of a secret key
            /// let mut sk: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the secret key with random)
            ///
            /// // we want the constant coefficient of the 3-th (counting from zero) polynomial
            /// // stored in the scret key if we look at it as a list of degree < 16 polynomials
            /// let bit: bool = SecretKey::get_bit_monomial(&sk, polynomial_size, 3, 0);
            /// ```
            fn get_bit_monomial(
                binary_key: &[$T],
                polynomial_size: usize,
                mask_index: usize,
                monomial_index: usize,
            ) -> bool {
                // compute which bit it is that we are looking for
                let i: usize = polynomial_size * mask_index + monomial_index;

                // calls get_bit
                return Self::get_bit(binary_key, i);
            }

            /// Convert a binary key stored in a Torus slice into a boolean slice
            /// # Arguments
            /// * `boolean_slice` - a boolean slice (output)
            /// * `sk` - a Torus slice storing a secret key
            /// * `sk_nb_bits` - the number of bits in the secret key
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::{secret_key, SecretKey};
            #[doc = $DOC]
            ///
            /// // settings
            /// let dimension: usize = 64;
            /// let polynomial_size: usize = 256;
            /// let sk_size: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            ///
            /// // allocation of a secret key
            /// let mut sk: Vec<Torus> = vec![0; sk_size];
            ///
            /// // ... (fill the secret key with random)
            ///
            /// // allocation of the result
            /// let mut bs: Vec<bool> = vec![false; dimension * polynomial_size];
            ///
            /// // conversion
            /// SecretKey::convert_key_to_boolean_slice(&mut bs, &sk, dimension * polynomial_size);
            /// ```
            fn convert_key_to_boolean_slice(
                boolean_slice: &mut [bool],
                sk: &[$T],
                sk_nb_bits: usize,
            ) {
                for i in 0..sk_nb_bits {
                    boolean_slice[i] = Self::get_bit(sk, i);
                }
            }

            /// Print the key as a string of bits, convenient for seeing a key as a LWE keys
            /// there is a space every 8 bits
            /// # Arguments
            /// * `sk` - a Torus slice representing a binary key
            /// * `dimension` - size of the mask
            /// * `polynomial_size` - number of coefficients in polynomials
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::{secret_key, SecretKey};
            /// use concrete::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let dimension: usize = 3;
            /// let polynomial_size: usize = 2;
            /// let sk_size: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            ///
            /// // creation of a random polynomial
            /// let mut sk: Vec<Torus> = vec![0; sk_size];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // print
            /// SecretKey::print(&sk, dimension, polynomial_size);
            ///
            /// // stdout:
            /// // [BINARY_KEY] key: 101101
            /// ```
            fn print(sk: &[$T], dimension: usize, polynomial_size: usize) {
                print!("[BINARY_KEY] key:");
                for i in 0..(dimension * polynomial_size) {
                    if i % 8 == 0 {
                        print!(" ");
                    }
                    if Self::get_bit(sk, i as usize) {
                        print!("1");
                    } else {
                        print!("0");
                    }
                }
                print!("\n");
            }

            /// Print the key as a string of binary polynomials, convenient for seeing a key as a RLWE keys
            /// # Arguments
            /// * `sk` - a Torus slice representing a binary key
            /// * `dimension` - size of the mask
            /// * `polynomial_size` - number of coefficients in polynomials
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::{secret_key, SecretKey};
            /// use concrete::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let dimension: usize = 3;
            /// let polynomial_size: usize = 2;
            /// let sk_size: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            ///
            /// // creation of a random polynomial
            /// let mut sk: Vec<Torus> = vec![0; sk_size];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // print
            /// SecretKey::print_ring(&sk, dimension, polynomial_size);
            ///
            /// // stdout:
            /// // [BINARY_KEY] key:
            /// //                   1 + 1 X
            /// //                   1 + 0 X
            /// //                   1 + 0 X
            /// ```
            fn print_ring(sk: &[$T], dimension: usize, polynomial_size: usize) {
                println!("[BINARY_KEY] key:");
                for i in 0..(dimension) {
                    // mask
                    print!("                  ");
                    for j in 0..polynomial_size {
                        // polynomial
                        if Self::get_bit(sk, (i * polynomial_size + j) as usize) {
                            print!("1");
                        } else {
                            print!("0");
                        }
                        if j > 0 {
                            print!(" X");
                        }
                        if j > 1 {
                            print!("^{}", j);
                        }
                        if j != polynomial_size - 1 {
                            print!(" + ");
                        }
                    }
                    print!("\n");
                }
            }
            /// Returns the number of Torus element needed to represent a binary key for LWE / RLWE samples
            /// according to the dimension of the mask and the size of the polynomials
            /// when dealing with LWE keys, set polynomial_size to 1
            /// # Arguments
            /// * `dimension` - size of the mask
            /// * `polynomial_size` - number of coefficients in polynomials
            /// # Output
            /// * the length of the Torus slice we need to store this kind of binary key
            /// # Example
            /// ```rust
            /// use concrete::core_api::crypto::SecretKey;
            ///
            /// type Torus = u32; // or u64
            ///
            /// // settings
            /// let dimension: usize = 128;
            /// let polynomial_size: usize = 1024;
            ///
            /// let length = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// ```
            fn get_secret_key_length(dimension: usize, polynomial_size: usize) -> usize {
                ((dimension * polynomial_size) as f32 / <Self as Types>::TORUS_BIT as f32).ceil() as usize
            }
        }
    };
}

impl_trait_secret_key!(u32, "type Torus = u32;");
impl_trait_secret_key!(u64, "type Torus = u64;");
