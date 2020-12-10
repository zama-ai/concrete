//! Part 3 of the Concrete guide where you learn how to compute some simple homomorphic operations with LWE and RLWE ciphertexts.
//!
//! This guide assumes that your already know how to **encode, encrypt** and **decrypt** LWE/RLWE ciphertexts with the Concrete library.
//! If not, [this part](super::guide_part2_encrypt_decrypt) is designed for you ;-)
//!
//! In this part we will compute **simple homomorphic operations**.
//! With "simple", we refer to any homomoprhic operation that does not need an extra key to be executed.
//!
//! It is **important** to notice that some homomorphic operations guarantees that the **result
//! is correct** whatever the input messages where but other homomorphic operations requires you
//! to somehow **know a little bit more about the distribution of the input or the output
//! messages**.
//! For each homomorphic operation, it will be **mentioned** when there is a **risk of losing correctness** or not.
//!
//! # Step 1: Learn about the relationship between LWE and RLWE ciphertexts
//!
//! We mentioned it in the previous part, **RLWE is an extension of LWE**.
//! An interesting thing to notice, is that you can publicly **extract** a coefficient of the polynomial plaintext inside an RLWE ciphertext, as an LWE ciphertexts (with "almost the same secret key").
//!
//! We first see how to extract an LWE ciphertext from an RLWE ciphertext, then we see how to convert an RLWE secret key into an "almost the same LWE secret key", and finally how to convert an LWE secret key into an "almost the same RLWE secret key".
//!
//! ## 1-A. Extract an RLWE coefficient as an LWE
//!
//! Since there are **several RLWE ciphertexts** inside an [VectorRLWE](super::super::crypto_api::VectorRLWE) instance, we will need to specify in **which ciphertexts** we want to extract in addition of **which coefficient** we are looking for.
//! To do so, we use the [extract_1_lwe](super::super::crypto_api::VectorRLWE::extract_1_lwe) function that extract one coefficient from one RLWE ciphertext contained in the RLWE structure.
//! Here is an example.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate a fresh secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let rlwe_ct = VectorRLWE::encrypt_packed(&secret_key, &plaintext).unwrap();
//!
//!     // extraction of the coefficient indexed by 2 (0.12)
//!     // in the RLWE ciphertext indexed by 0
//!     let lwe_ct = rlwe_ct.extract_1_lwe(2, 0);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Note that there is **no risk** of losing correctness with this operation.
//!
//! Now you must feel frustrated.
//! Indeed, you don't have the **LWE secret key** to decrypt the LWE ciphertext we just extracted!
//! Don't worry it's very easy :-)
//!
//! ## 1-B. Convert an RLWE secret key into an LWE secret key
//!
//! An RLWE secret key is a vector of polynomials (with binary coefficients in the Concrete library).
//! If you build a long vector of bits, with **each coefficient in the natural order** of the polynomials of the RLWE secret key, you get an LWE secret key!
//! Isn't it magic?
//! This way of converting the RLWE secret key into an LWE secret key must be followed by the RLWE ciphertext extraction algorithm.
//!
//! You can easily convert an [RLWESecretKey](super::super::crypto_api::RLWESecretKey) instance into an [LWESecretKey](super::super::crypto_api::LWESecretKey) with the [to_lwe_secret_key](super::super::crypto_api::RLWESecretKey::to_lwe_secret_key) method.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // generate a fresh secret key
//!     let rlwe_sk = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // derive the RLWE secret key into an LWE secret key
//!     let lwe_sk = rlwe_sk.to_lwe_secret_key();
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Let's **decrypt** our extracted LWE ciphertext now.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate secret keys
//!     let rlwe_sk = RLWESecretKey::new(&RLWE128_1024_1);
//!     let lwe_sk = rlwe_sk.to_lwe_secret_key();
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let rlwe_ct = VectorRLWE::encrypt_packed(&rlwe_sk, &plaintext).unwrap();
//!
//!     // extraction of the coefficient indexed by 2 (0.12)
//!     // in the RLWE ciphertext indexed by 0
//!     let lwe_ct = rlwe_ct.extract_1_lwe(2, 0).unwrap();
//!
//!     // decryption
//!     let decryptions: Vec<f64> = lwe_ct.decrypt_decode(&lwe_sk).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     if (decryptions[0] - messages[2]).abs() > encoder.get_granularity() / 2. {
//!         panic!();
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Pretty easy :-) right?
//!
//! ## 1-C. Convert an LWE secret key into an RLWE secret key
//!
//! It works almost the same, the only difference is that you have to **specify the polynomial size** when you use the [to_rlwe_secret_key](super::super::crypto_api::LWESecretKey::to_rlwe_secret_key) method.
//! Obviously it has to **divide the dimension** of the LWE secret key and be a **power of two**.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // generate a fresh secret key
//!     let lwe_sk = LWESecretKey::new(&LWE128_1024);
//!
//!     // derive the LWE secret key into an RLWE secret key
//!     let rlwe_sk = lwe_sk.to_rlwe_secret_key(512);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Step 2: Compute an addition between a ciphertext and a constant
//!
//! Because **we're not doing modular arithmetic computation** but rather some computation on **approximated real numbers**, there are different ways to compute a simple addition.
//!
//! ## 2-A. Compute an addition between a ciphertext and a small constant by staying in the same interval
//!
//! With an LWE ciphertext it is pretty simple to compute an **homomorphic addition** between an **LWE ciphertext** and a **message**, that **output a new LWE ciphertext**.
//! Indeed, the only thing to do, is to add a particular plaintext to the body of the ciphertext!
//!
//! Note that are working in [min,max], so the messages that we add should be **small enough** to not go beyond min and max.
//! It means that we can actually **lose correctness** with this operation if we go beyond the interval.
//! Here is an example of how to compute this kind of addition in the interval [100,110], and for instance, we will start with an LWE ciphertext of the message 106.276, and add to it the message -4.9, so **the result is still in the interval**.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_1024);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
//!
//!     // encode and encrypt
//!     let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
//!     let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext.add_constant_static_encoder_inplace(&messages_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > encoder.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! As you can see, we used the [add_constant_static_encoder_inplace](super::super::crypto_api::VectorLWE::add_constant_static_encoder_inplace) method that only takes for argument the messages.
//!
//! It works exactly the same with **RLWE ciphertexts** with the [add_constant_static_encoder_inplace](super::super::crypto_api::VectorRLWE::add_constant_static_encoder_inplace) method.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
//!
//!     // encode and encrypt
//!     let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
//!     let mut ciphertext = VectorRLWE::encrypt_packed(&secret_key, &plaintext_1).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext.add_constant_static_encoder_inplace(&messages_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > encoder.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 2-B. Compute an addition between a ciphertext and any constant by changing the interval
//!
//! There is also another way to compute this addition **without having to even modify the ciphertext**...
//! It can be useful if the message is **big** for instance.
//! Indeed we simply **change the Encoder** and replace [min,max] by [min+m;max+m] so when we will decrypt we end up with the sum between the original message in the LWE ciphertext and m.
//! Note that there is **no risk** of losing correctness with this operation.
//! Here is an example.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let messages_2: Vec<f64> = vec![3017.3, -49.1, -93.33, 86., -3.2];
//!
//!     // encode and encrypt
//!     let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
//!     let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext.add_constant_dynamic_encoder_inplace(&messages_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > encoder.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! As you can see, we used the [add_constant_dynamic_encoder_inplace](super::super::crypto_api::VectorLWE::add_constant_dynamic_encoder_inplace) method that takes as input only the messages.
//!
//! Now with **RLWE** ciphertext, we also use the [add_constant_dynamic_encoder_inplace](super::super::crypto_api::VectorRLWE::add_constant_dynamic_encoder_inplace) method.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let messages_2: Vec<f64> = vec![3017.3, -49.1, -93.33, 86., -3.2];
//!
//!     // encode and encrypt
//!     let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
//!     let mut ciphertext = VectorRLWE::encrypt_packed(&secret_key, &plaintext_1).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext.add_constant_dynamic_encoder_inplace(&messages_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > encoder.get_granularity() {
//!             panic!(
//!                 "(before_1 + before_2 - after).abs() = {} ;  encoder.get_granularity() = {}",
//!                 (before_1 + before_2 - after).abs(),
//!                 encoder.get_granularity()
//!             );
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Step 3: Compute an homomorphic opposite
//!
//! A simple homomorphic operation is the computation of the **opposite**.
//! Meaning that if you have a ciphertext of a message m in the interval [min,max], you can **easily convert** it into a new ciphertext of a message -m in the interval [-max,-min] :-)
//! To do so we have the [opposite_nth_inplace](super::super::crypto_api::VectorLWE::opposite_nth_inplace) method that takes as argument the index of the ciphertext you want to compute the opposite and modify directly in the structure.
//! Let's see now with an example.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(50., 100., 8, 2).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_1024);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![66.65, 84.87, 95.46];
//!
//!     // encode and encrypt
//!     let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
//!
//!     // compute the opposite of the second ciphertext
//!     ciphertext.opposite_nth_inplace(1);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     if (messages[1] + decryptions[1]).abs() > encoder.get_granularity() / 2. {
//!         panic!();
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Step 4: Compute an homomorphic addition
//!
//! As with the constant addition we have **different ways** to compute an homomorphic addition.
//!
//! ## 4-A. Compute an addition between two ciphertexts when you know the output interval
//!
//! Now, we're going to **add two LWE ciphertexts** (of some messages m1 and m2) together and the result is going to be an LWE ciphertext (of the m1+m2).
//! There are **some constraints** when you want to do that, indeed we need for the two LWE input ciphertexts to be working in two intervals with the **same size**.
//! It means that if we use [min1,max1] as an encoder for m1 and [min2,max2] as an encoder for m2, we need to have **max1-min1=max2-min2**.
//! Then, the result m1+m2 will be encoded in [min,max] with max-min=max1-min1=max2-min2 and the min value **has to be provided**.
//! Those constraints force you to know what's going on in the computation, at least to have an idea of the intervals.
//! It also means that we can actually **lose correctness** with this operation if the provided output interval [min,max] does not include the result.
//! Here is an example, please note that in order to not lose any precision we set the same number of bits of precision for both Encoders.
//!
//! ```rust
//! use itertools::izip;
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoders
//!     let encoder_1 = Encoder::new_centered(105., 5., 8, 0).unwrap();
//!     let encoder_2 = Encoder::new_centered(-17., 5., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_1024);
//!
//!     // two lists of messages
//!     // 3 independent ciphertexts from the normal distribution of mean 105 and standard derivation 1
//!     let messages_1: Vec<f64> = vec![106.65286348661301, 104.87845375069587, 105.46354804688922];
//!     // 3 independent ciphertexts from the normal distribution of mean -17 and standard
//! derivation 2
//!     let messages_2: Vec<f64> =
//!         vec![-18.00952927821479, -15.095226956064629, -16.952167875620457];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
//!     let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
//!
//!     // addition between ciphertext_1 and ciphertext_2
//!     ciphertext_1.add_centered_inplace(&ciphertext_2); // the new interval will be centered in 105-17=88
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > ciphertext_1.encoders[0].get_granularity()  {
//!             panic!("(before_1 + before_2 - after).abs() = {} ; ciphertext_1.encoders[0].get_granularity()  = {}",
//!             (before_1 + before_2 - after).abs(),
//!             ciphertext_1.encoders[0].get_granularity());
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Here we used the [add_centered](super::super::crypto_api::VectorLWE::add_centered) method that will **automatically** compute a new interval defined by (new_center,radius) from the two input intervals (center1,radius) and (center2,radius) as follow: new_center=center1+center2.
//! Note that this way of adding ciphertexts is convenient when you deal with normal distribution
//! as
//! messages :-)
//!
//! Obviously they all **share the same radius** since the length of the interval has to be the same.
//! There are **other ways** to compute this sum with more flexibility regarding the new interval such as the [add_with_new_min](super::super::crypto_api::VectorLWE::add_with_new_min) method, where you directly provide **new min values** for the output interval.
//!
//! It actually works pretty the same with the **RLWE ciphertexts** with the [add_centered_inplace](super::super::crypto_api::VectorRLWE::add_centered_inplace) method.
//!
//! ```rust
//! use itertools::izip;
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoders
//!     let encoder_1 = Encoder::new_centered(105., 5., 8, 0).unwrap();
//!     let encoder_2 = Encoder::new_centered(-17., 5., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     // 3 independent ciphertexts from the normal distribution of mean 105 and std 1
//!     let messages_1: Vec<f64> = vec![106.65286348661301, 104.87845375069587, 105.46354804688922];
//!     // 3 independent ciphertexts from the normal distribution of mean -17 and standard
//! derivation 2
//!     let messages_2: Vec<f64> =
//!         vec![-18.00952927821479, -15.095226956064629, -16.952167875620457];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages_1, &encoder_1).unwrap();
//!     let ciphertext_2 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages_2, &encoder_2).unwrap();
//!
//!     // addition between ciphertext_1 and ciphertext_2
//!     ciphertext_1.add_centered_inplace(&ciphertext_2).unwrap(); // the new interval will be centered in 105-17=88
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after,enc1) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter(),ciphertext_1.encoders.iter())
//!     {
//!         if (before_1 + before_2 - after).abs() > enc1.get_granularity() {
//!             panic!(
//!                 "{} {} {} {} {}",
//!                 before_1,
//!                 before_2,
//!                 after,
//!                 (before_1 + before_2 - after).abs(),
//!                 encoder_1.get_granularity()
//!             );
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 4-B. Compute an addition between two ciphertexts by consuming padding
//!
//! We are here using the **padding to compute the addition**.
//! It means that we have to have two LWE ciphertexts of two messages m1 and m2 such that they both belong to **intervals with the same size**, and with the **same number of padding bits**.
//! If m1 was in [min1,max1] and m2 was in [min2,max2] then the output message is in [min1+min2,max1+max2], and the padding for the output is **decremented by 1**.
//! Note that there is **no risk** of losing correctness with this operation.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 7, 1).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // two lists of messages
//!     let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
//!     let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext_1.add_with_padding_inplace(&ciphertext_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages.iter(),
//!         messages.iter(),
//!         decryptions.iter(),
//!         ciphertext_1.encoders.iter()
//!     ) {
//!         if (before_1 + before_2 - after).abs() > enc.get_granularity() {
//!             panic!(
//!                 "(before_1 + before_2 - after).abs() = {} ; enc.get_granularity() = {}",
//!                 (before_1 + before_2 - after).abs(),
//!                 enc.get_granularity()
//!             );
//!         }
//!     }
//! }
//! ```
//!
//! In the previous example, we encrypted twice the same messages, and add them together thanks to the [add_with_padding_inplace](super::super::crypto_api::VectorLWE::add_with_padding_inplace) method.
//!
//! It works the same with **RLWE** ciphertexts and the [add_with_padding_inplace](super::super::crypto_api::VectorRLWE::add_with_padding_inplace) method.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 7, 1).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
//!     let ciphertext_2 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
//!
//!     // addition between ciphertext and messages_2
//!     ciphertext_1.add_with_padding_inplace(&ciphertext_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages.iter(),
//!         messages.iter(),
//!         decryptions.iter(),
//!         ciphertext_1.encoders.iter()
//!     ) {
//!         if (before_1 + before_2 - after).abs() > enc.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 4-C. Compute a subtraction between two ciphertexts by consuming padding
//!
//! In the exact same way as with the addition above, we are using the **padding to compute the subtraction**.
//! It means that we have to have two LWE ciphertexts of two messages m1 and m2 such that they both belong to **intervals with the same size**, and with the **same number of padding bits**.
//! If m1 was in [min1,max1] and m2 was in [min2,max2] then the output message m1-m2 is in [min1-max2,max1-min2], and the padding for the output is **decremented by 1**.
//! Note that there is **no risk** of losing correctness with this operation.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder_1 = Encoder::new(100., 110., 7, 1).unwrap();
//!     let encoder_2 = Encoder::new(-30., -20., 7, 1).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let messages_2: Vec<f64> = vec![-22., -27.5, -21.2, -29., -25.];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 =
//!         VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
//!     let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
//!
//!     // subtraction between ciphertext and messages_2
//!     ciphertext_1.sub_with_padding_inplace(&ciphertext_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages_1.iter(),
//!         messages_2.iter(),
//!         decryptions.iter(),
//!         ciphertext_1.encoders.iter()
//!     ) {
//!         if (before_1 - before_2 - after).abs() > enc.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! In the previous example, we encrypted twice the same messages, and add them together thanks to the [add_with_padding_inplace](super::super::crypto_api::VectorRLWE::add_with_padding_inplace) method.
//!
//! It works the same with **RLWE** ciphertexts and the [sub_with_padding_inplace](super::super::crypto_api::VectorRLWE::sub_with_padding_inplace) method.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 7, 1).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!
//!     // encode and encrypt
//!     let mut ciphertext_1 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
//!     let ciphertext_2 =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
//!
//!     // subtraction between ciphertext and messages_2
//!     ciphertext_1.sub_with_padding_inplace(&ciphertext_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages.iter(),
//!         messages.iter(),
//!         decryptions.iter(),
//!         ciphertext_1.encoders.iter()
//!     ) {
//!         if (before_1 - before_2 - after).abs() > enc.get_granularity() / 2. {
//!             panic!("decryption: {}", after);
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Step 5: Compute a multiplication between a ciphertext and a constant
//!
//! It is pretty easy to **multiply each element of both mask and body ** of an LWE/RLWE ciphertext by an **integer**.
//! This way we can multiply ciphertexts by integer messages.
//! Because of the encodings, we also have **different ways** to deal with that.
//!
//! ## 5-A. Compute a multiplication between a ciphertext and a small constant and stay in the same interval
//!
//! First, we will consider that the result of the multiplication is in the **same interval** as the LWE/RLWE input.
//! The constant has to be small to stay in the same interval, so it means that we can **lose correctness** with this operation if we actually go beyond the interval.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-30., 30., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_1024);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![6.2, 4.3, -1.1, -12.3, 7.7];
//!     let messages_2: Vec<i32> = vec![-4, 6, 20, -2, 3];
//!
//!     // encode and encrypt
//!     let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
//!     let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
//!
//!     // multiplication between ciphertext and messages_2
//!     ciphertext.mul_constant_static_encoder_inplace(&messages_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after) in
//!         izip!(messages_1.iter(), messages_2.iter(), decryptions.iter())
//!     {
//!         if (before_1 * (*before_2 as f64) - after).abs() > encoder.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! In the previous example, we use the [mul_constant_static_encoder_inplace](super::super::crypto_api::VectorRLWE::mul_constant_static_encoder_inplace) method.
//!
//! It works the same with **RLWE** ciphertexts and the [mul_constant_static_encoder_inplace](super::super::crypto_api::VectorRLWE::mul_constant_static_encoder_inplace) method.
//!
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-30., 30., 6, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![6.2, 4.3, -1.1, -12.3, 7.7];
//!     let message_2: Vec<i32> = vec![-2];
//!
//!     // encode and encrypt
//!     let mut ciphertext =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages_1, &encoder).unwrap();
//!
//!     // multiplication between ciphertexts and an integers
//!     ciphertext.mul_constant_static_encoder_inplace(&message_2);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, after) in izip!(messages_1.iter(), decryptions.iter()) {
//!         if (before_1 * (message_2[0] as f64) - after).abs() > encoder.get_granularity() / 2. {
//!             panic!("decryption: {}", after);
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 5-B. Compute a multiplication between a ciphertext and a bigger constant by changing the encoding
//!
//! We can also compute a **multiplication** between a ciphertext and any real constant (up to a certain **precision**) by eating bits of padding.
//! The number of bits of padding consumed **represents the precision** for the real constant.
//! We use the [mul_constant_with_padding_inplace](super::super::crypto_api::VectorLWE::mul_constant_with_padding_inplace) method to compute such multiplication, and it takes as upper bound of the constant, in absolute value.
//! This upper bound will be used to determine the **new interval** for the output encoding, so if we have several input with the same encoding and if we use the same upper bound each time, we will end up with several identical output interval :-) useful for adding them together for instance.
//! Obviously, if the constant is negative we have the opposite interval, but still with the **same size**!
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-2., 6., 4, 4).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_1024);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![-1., 2., 0., 5., -0.5];
//!     let messages_2: Vec<f64> = vec![-2., -1., 3., 2.5, 1.5];
//!
//!     // encode and encrypt
//!     let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
//!
//!     // multiplication between ciphertext and messages_2
//!     let max_constant: f64 = 3.;
//!     let scalar_precision: usize = 4;
//!     ciphertext
//!         .mul_constant_with_padding_inplace(&messages_2, max_constant, scalar_precision)
//!         .unwrap();
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages_1.iter(),
//!         messages_2.iter(),
//!         decryptions.iter(),
//!         ciphertext.encoders.iter()
//!     ) {
//!         if (before_1 * before_2 - after).abs() > enc.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! It works pretty the same with RLWE.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     use itertools::izip;
//!     // encoder
//!     let encoder = Encoder::new(-2., 6., 4, 4).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![-1., 2., 0., 5., -0.5];
//!     let messages_2: Vec<f64> = vec![-2., -1., 3., 2.5, 1.5];
//!
//!     // encode and encrypt
//!     let mut ciphertext =
//!         VectorRLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
//!
//!     // multiplication between ciphertext and messages_2
//!     let max_constant: f64 = 3.;
//!     let scalar_precision: usize = 4;
//!     ciphertext
//!         .mul_constant_with_padding_inplace(&messages_2, max_constant, scalar_precision)
//!         .unwrap();
//!
//!     // decryption
//!     let (decryptions, dec_encoders) = ciphertext.decrypt_with_encoders(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before_1, before_2, after, enc) in izip!(
//!         messages_1.iter(),
//!         messages_2.iter(),
//!         decryptions.iter(),
//!         dec_encoders.iter()
//!     ) {
//!         if (before_1 * before_2 - after).abs() > enc.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Conclusion
//!
//! In this part, you saw:
//! - how to extract an LWE ciphertext from an RLWE ciphertext and deal with keys
//! - how to compute an addition between a ciphertext and a constant message
//! - how to compute the opposite of a ciphertext
//! - how to compute an addition between two ciphertexts
//! - how to compute a multiplication between a ciphertext and a constant message
//!
//! You're all set for the [next part](super::guide_part4_complex_operations) where you will learn how to compute pretty complex homomorphic operations :-o
