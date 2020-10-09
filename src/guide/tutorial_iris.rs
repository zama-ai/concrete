//! A tutorial to evaluate a trained logistic regression with Concrete library.
//!
//! In this tutorial, we will use the **[Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)** and train a logistic regression on it.
//! This dataset is sorting irises according to **4 lengths measured on flowers**.
//! There are **3 different types** of irises.
//! With this tutorial we want to showcase how to use some of the Concrete's functions.
//! The goal of this tutorial is to train a logistic regression with python3 and implement in Rust a program using the Concrete library to **homomorphically predict the class of an encrypted input iris**.
//!
//! # 1. Set up
//!
//! To complete this tutorial, you have to **install** the following tools:
//! - python3
//! - sklearn
//! - numpy
//! - Rust
//!
//! # 2. Start a new Rust project
//!
//! To create a new **Rust project**, you can run this simple command:
//! ```bash
//! cargo new iris_log_reg
//! ```
//!
//! There is a ``main.rs`` file in this new project and we are going to **edit it** during this tutorial :-)
//!
//! # 3. Train a logistic regression over the Iris dataset
//!
//! We are going to use python and sklearn to **train** our logistic regression and we can do that as follow.
//!
//! ```python
//! # Python code
//!
//! import numpy as np
//! from sklearn.linear_model import LogisticRegression
//! from sklearn import datasets
//!
//! # import the Iris dataset
//! dataset_iris = datasets.load_iris()
//!
//! # concatenate the features with their labels and shuffle them
//! data_and_labels = np.c_[(dataset_iris.data,dataset_iris.target)]
//! np.random.shuffle(data_and_labels)
//!
//! # split into a training set (120 elements) and a validation set (30 elements)
//! training_set = data_and_labels[:120,:-1]
//! training_labels = data_and_labels[:120,-1]
//! testing_set = data_and_labels[120:,:-1]
//! testing_labels = data_and_labels[120:,-1]
//!
//! # create an instance of a logistic regression
//! logreg = LogisticRegression(C=1e5)
//!
//! # train the logistic regression
//! logreg.fit(training_set,training_labels)
//!
//! # print the score over the validation set
//! print(f"score: {logreg.score(testing_set,testing_labels)*100}")
//! ```
//!
//! With this particular example we have 100% of accuracy :p
//!
//! # 4. Take a look at the weights of the trained logistic regression
//!
//! The coefficients are stored in ``coef_``.
//! We have in our logistic regression **3 perceptrons** because we have 3 classes possible.
//! Each perceptron takes as input **4 feature values** since our dataset has 4 features.
//! In consequence, we have 4*3=**12 coefficients** in our logistic regression.
//! We can already write some Rust code:
//!
//!
//! ```rust
//! // Rust code
//!
//! // topology of the logistic regression
//! const NB_PERCEPTRON: usize = 3;
//! const NB_FEATURE: usize = 4;
//! ```
//!
//! Let's have a look to the minimum and maximum values of the coefficients.
//!
//! ```python
//! # Python code
//!
//! print(f"min: {logreg.coef_.min()}; max: {logreg.coef_.max()}")
//! ```
//!
//! They are (in absolute value) **lesser than 11**.
//! Now we can print them and convert them into Rust code.
//!
//! ```python
//! # Python code
//!
//! import functools
//! print(f"const WEIGHTS: [f64; {logreg.coef_.flatten().shape[0]}] = [{functools.reduce(lambda x,y : str(x)+', '+str(y),logreg.coef_.flatten())}];")
//! print(f"const BIASES: [f64; {logreg.intercept_.shape[0]}] = [{functools.reduce(lambda x,y : str(x)+', '+str(y),logreg.intercept_)}];")
//! ```
//!
//! We get something like that that we can immediately copy paste into our Rust project.
//!
//! ```rust
//! // Rust code
//!
//! // weights
//! const MAX_WEIGHT: f64 = 11.;
//! const WEIGHTS: [f64; 12] = [
//!     3.3758082444336295,
//!     7.9196458076215945,
//!     -10.774548167193158,
//!     -5.487605086345206,
//!     -1.1369996360241739,
//!     -0.7553735705885082,
//!     1.4312527569766305,
//!     -5.064955905902094,
//!     -2.238808608432861,
//!     -7.164272237046656,
//!     9.343295410205023,
//!     10.552560992243956,
//! ];
//! const BIASES: [f64; 3] = [1.666627227331721, 18.918783641421552, -20.585410868758302];
//! ```
//!
//! We will need the **validation set**, both the input and their labels in our Rust project to really play :-)
//!
//! ```python
//! # Python code
//!
//! import functools
//! print(f"const VALIDATION_SET: [f64; {testing_set.shape[0]*testing_set.shape[1]}] = [{functools.reduce(lambda x,y : str(x)+', '+str(y),testing_set.flatten())}];")
//! print(f"const VALIDATION_LABELS: [f64; {testing_labels.shape[0]}] = [{functools.reduce(lambda x,y : str(x)+', '+str(y),testing_labels)}];")
//! ```
//!
//! ```rust
//! // Rust code
//!
//! // validation set
//! const VALIDATION_SET: [f64; 120] = [
//!     5.3, 3.7, 1.5, 0.2, 5.7, 3.8, 1.7, 0.3, 5.2, 4.1, 1.5, 0.1, 7.4, 2.8, 6.1, 1.9, 7.0, 3.2,
//!     4.7, 1.4, 5.2, 3.5, 1.5, 0.2, 6.4, 2.7, 5.3, 1.9, 5.6, 2.8, 4.9, 2.0, 5.5, 4.2, 1.4, 0.2,
//!     6.7, 3.0, 5.0, 1.7, 7.7, 3.0, 6.1, 2.3, 5.5, 2.4, 3.8, 1.1, 6.1, 2.8, 4.0, 1.3, 6.1, 3.0,
//!     4.9, 1.8, 4.5, 2.3, 1.3, 0.3, 6.4, 2.8, 5.6, 2.1, 5.4, 3.9, 1.3, 0.4, 5.7, 2.6, 3.5, 1.0,
//!     6.7, 3.1, 4.4, 1.4, 7.6, 3.0, 6.6, 2.1, 5.2, 3.4, 1.4, 0.2, 5.8, 2.6, 4.0, 1.2, 5.1, 3.8,
//!     1.5, 0.3, 7.2, 3.2, 6.0, 1.8, 4.8, 3.0, 1.4, 0.1, 6.5, 3.0, 5.5, 1.8, 6.0, 3.4, 4.5, 1.6,
//!     5.1, 3.8, 1.9, 0.4, 6.3, 2.5, 4.9, 1.5, 5.9, 3.0, 5.1, 1.8,
//! ];
//! const VALIDATION_LABELS: [f64; 30] = [
//!     0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0,
//!     1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
//! ];
//! ```
//!
//! # 5. Find good encodings for our input
//!
//! Let have a look to our input and find a good interval for each feature.
//! We want to see each minimum value and each maximum value.
//! ```python
//! # Python code
//!
//! for col in training_set.T:
//!     print(f"min: {col.min()}; max: {col.max()}")
//! ```
//!
//! It make sens to use the interval [0,8].
//! We can add other constant to our implementation in Rust, such as the minimum of the encoding interval and the its size.
//! We will also need **other parameters**: the number of bits of precision and the number of bits of precision for the weights and also the number of bits of padding.
//!
//! ```rust
//! // Rust code
//!
//! // encoding settings
//! const MIN_INTERVAL: f64 = 0.;
//! const MAX_INTERVAL: f64 = 8.;
//! const NB_BIT_PRECISION: usize = 5;
//! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
//! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
//! ```
//!
//! We took 8 bits of padding because we are going to use 5 of them for the multiplication by a constant, 2 for summing 4 LWE ciphertexts, and a last one to compute a bootstrap evaluating the activation function.
//!
//! Now your Rust code should look like [this](full_code_1).
//!
//! # 6. Setup the crypto
//!
//! We have to write a function that **generates everything we need for the crypto**: a [secret key](super::super::crypto_api::LWESecretKey) to encrypt/decrypt and a [bootstrapping key](super::super::crypto_api::LWEBSK) to compute complex homomorphic operations.
//!
//! Once again we have to **pick some settings**: the LWE parameters, a base_log and a level.
//! They will all have an **impact on the precision** we can have at most, so they are very important.
//!
//! The creation of the bootstrapping key takes some time so for now we used the [zero](super::super::crypto_api::LWEBSK::zero) function which only allocates instead of the [new](super::super::crypto_api::LWEBSK::new) function.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! // lwe settings
//! const LWE_PARAMS: LWEParams = LWE128_688;
//! const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
//!
//! // bootstrapping settings
//! const BASE_LOG: usize = 8;
//! const LEVEL: usize = 5;
//!
//! fn setup_gen() -> (LWESecretKey, LWEBSK, LWESecretKey) {
//!     // generation of the keys
//!     let lwe_key_input = LWESecretKey::new(&LWE_PARAMS);
//!     let rlwe_secret_key = RLWESecretKey::new(&RLWE_PARAMS);
//!     let lwe_key_output = rlwe_secret_key.to_lwe_secret_key();
//!     let bootstrapping_key = LWEBSK::new(&lwe_key_input, &rlwe_secret_key, BASE_LOG, LEVEL);
//!
//!     // we write the keys in files
//!     lwe_key_output.save("lwe_key_output.json").unwrap();
//!     lwe_key_input.save("lwe_key_input.json").unwrap();
//!     bootstrapping_key.write_in_file_bytes("bsk_bytes.txt");
//!
//!     (lwe_key_input, bootstrapping_key, lwe_key_output)
//! }
//! ```
//!
//! Since it can takes a long time to generate a bootstrapping key we wrote them into files and we can add a setup function that loads those files.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! fn setup_load() -> (LWESecretKey, LWEBSK, LWESecretKey) {
//!     let bootstrapping_key = LWEBSK::read_in_file_bytes("bsk_bytes.txt");
//!     let lwe_key_input = LWESecretKey::load("lwe_key_input.json").unwrap();
//!     let lwe_key_output = LWESecretKey::load("lwe_key_output.json").unwrap();
//!
//!     (lwe_key_input, bootstrapping_key, lwe_key_output)
//! }
//! ```
//!
//! # 7. Write an encrypt function
//!
//! Now that we have a way to generate our secret key, we can use it to **encrypt an input**.
//! In our use-case, an input is an f64 slice of size 4.
//! Let's write an encrypt function that will encrypt our input.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! // encoding settings
//! const MIN_INTERVAL: f64 = 0.;
//! const MAX_INTERVAL: f64 = 8.;
//! const NB_BIT_PRECISION: usize = 5;
//! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
//! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
//!
//! fn encrypt_input(input: &[f64], lwe_secret_key: &LWESecretKey) -> VectorLWE {
//!     // generate an encoder
//!     let encoder =
//!         Encoder::new(MIN_INTERVAL, MAX_INTERVAL, NB_BIT_PRECISION, NB_BIT_PADDING).unwrap();
//!
//!     // encode and encrypt
//!     let ciphertexts = VectorLWE::encode_encrypt(lwe_secret_key, input, &encoder).unwrap();
//!
//!     ciphertexts
//! }
//! ```
//!
//! # 8. Write a decrypt and argmax function
//!
//! In the same way we will write a function that will decrypt and compute the argmax function obtain the inference.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! fn decrypt_inference(ciphertexts: &VectorLWE, lwe_secret_key: &LWESecretKey) -> usize {
//!     // decrypt
//!     let decryptions = ciphertexts.decrypt_decode(lwe_secret_key).unwrap();
//!
//!     // find the argmax
//!     let mut res: usize = 0;
//!     for (i, val) in decryptions.iter().enumerate() {
//!         if decryptions[res] < *val {
//!             res = i;
//!         }
//!     }
//!
//!     res
//! }
//! ```
//!
//! Now the last thing we have to code, and the most interesting is the **homomorphic computation**!
//! But first we can check that **our code works** for now with a simple example where we encrypt some values and get the argmax.
//! You can try on your own and when you feel like it you chan check a possible answer right [here](full_code_2)
//!
//! # 9. Computation of the products between LWE ciphertexts and some constants
//!
//! We now start to code the homomorphic computation code!
//! The first thing for our inference is to **multiply each features with a constant**, and that for each perceptron.
//! We are going to write a function that homomorphically computes what a single preceptron needs to compute.
//! It takes as input for now some LWE ciphertexts, the bootstrapping key, the weights, the bias and it output nothing (for now :p).
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! const NB_BIT_PRECISION: usize = 5;
//! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
//!
//! // topology of the logistic regression
//! const NB_PERCEPTRON: usize = 3;
//! const NB_FEATURE: usize = 4;
//!
//! // weights
//! const MAX_WEIGHT: f64 = 11.;
//!
//! fn homomorphic_perceptron(
//!     ciphertexts: &mut VectorLWE,
//!     key_switching_key: &LWEKSK,
//!     bootstrapping_key: &LWEBSK,
//!     weights: &[f64],
//!     bias: f64,
//! ) {
//!     // clone the ciphertexts
//!     let mut ct = ciphertexts.clone();
//!
//!     // multiply each ciphertext with a weight
//!     ct.mul_constant_with_padding_inplace(&weights, MAX_WEIGHT, NB_BIT_PRECISION)
//!         .unwrap();
//!
//!     // more homomorphic operations...
//! }
//! ```
//!
//! # 10. Homomorphic computation a sum of LWE ciphertexts
//!
//! We need to **compute a sum** now between the 4 resulting ciphertexts obtained after multiplication.
//! We are going to **write a quick function** that will take as argument a LWE structure with four LWE ciphertexts in it, and output one LWE structure with one LWE ciphertext in it that is the sum of the four input ones.
//! The sum will be done by **consuming 2 bits of padding**.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//!
//! fn add_4(ciphertexts: &VectorLWE) -> VectorLWE {
//!     // first addition
//!     let mut c0 = ciphertexts.extract_nth(0).unwrap();
//!     let c1 = ciphertexts.extract_nth(1).unwrap();
//!     c0.add_with_padding_inplace(&c1);
//!
//!     // second addition
//!     let mut c2 = ciphertexts.extract_nth(2).unwrap();
//!     let c3 = ciphertexts.extract_nth(3).unwrap();
//!     c2.add_with_padding_inplace(&c3);
//!
//!     // last addition
//!     c0.add_with_padding_inplace(&c2);
//!     c0
//! }
//! ```
//!
//! Now we can **plug** this function in our ``homomorphic_perceptron`` function and finally return an LWE (even if we have to compute others things in between).
//!
//! Your code should look like [that](full_code_3).
//!
//! # 11. Add the bias and compute the sigmoid function
//!
//! There are two last things our homomorphic perceptron has to compute, the **addition with the bias** and the **computation of the sigmoid function**.
//! We can do both at the same time with a **bootstrap**!
//! In the following code we are going to code a simpler function just to show you how to add a bootstrap to your ``homomorphic_perceptron`` function.
//! We have to use a key switching procedure first to make the bootstrapping key smaller and the bootstrap faster.
//!
//! ```rust
//! use concrete_lib::*;
//!
//! const NB_BIT_PRECISION: usize = 5;
//! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
//!
//! fn mockup_bootstrap(ciphertext_input: &mut VectorLWE, bootstrapping_key: &LWEBSK, bias: f64) {
//!     // output interval
//!     let encoder_output = Encoder::new(0., 1., NB_BIT_PRECISION, 0).unwrap();
//!
//!     // bootstrap
//!     let ciphertext_output = ciphertext_input
//!         .bootstrap_nth_with_function(
//!             bootstrapping_key,
//!             |x| 1. / (1. + (-x - bias).exp()), // the sigmoid function
//!             &encoder_output,                   // contains the output of the sigmoid function
//!             0,                                 // the 0th LWE ciphertext in the structure
//!         )
//!         .unwrap();
//! }
//! ```
//!
//! # 12. Wrap ``homomorphic_perceptron`` into a ``homomorphic_inference`` function
//!
//! We have to write a function that will call ``homomorphic_perceptron`` as many times as we have preceptrons and aggregate the results.
//!
//! ```rust
//! /// file: main.rs
//! use concrete_lib::*;
//! use itertools::izip;
//!
//! const NB_PERCEPTRON: usize = 3;
//! const NB_FEATURE: usize = 4;
//! const WEIGHTS: [f64; 12] = [0.; 12];
//! const BIASES: [f64; 3] = [0.; 3];
//!
//! fn homomorphic_inference(input_ciphertexts: &mut VectorLWE, bootstrapping_key: &LWEBSK) -> VectorLWE {
//!     // allocation of the result with zeros
//!     let mut result = VectorLWE::zero(
//!         bootstrapping_key.dimension * bootstrapping_key.polynomial_size,
//!         NB_PERCEPTRON,
//!     )
//!     .unwrap();
//!
//!     // compute for each perceptron
//!     for (weights, bias, i) in izip!(WEIGHTS.chunks(NB_FEATURE), BIASES.iter(), 0..NB_PERCEPTRON)
//!     {
//!         let mut tmp =
//!             homomorphic_perceptron(input_ciphertexts, bootstrapping_key, weights, *bias);
//!
//!         // copy the output of the perceptron inside result
//!         result.copy_in_nth_nth_inplace(i, &tmp, 0);
//!     }
//!
//!     return result;
//! }
//!
//! fn homomorphic_perceptron(
//!     ciphertexts: &mut VectorLWE,
//!     bootstrapping_key: &LWEBSK,
//!     weights: &[f64],
//!     bias: f64,
//! ) -> VectorLWE {
//!     // some code
//!     return VectorLWE::zero(1, 1).unwrap();
//! }
//! ```
//!
//! # 13. Put everything together
//!
//! Now you have all you need to combine everything.
//! We end up with an accuracy of 0.67.
//!
//! The code is available [here](full_code_4).

pub mod full_code_1 {
    //! The full code step by step: step 1
    //!
    //! Back to the [tutorial](super).
    //!
    //! ```rust
    //! /// file: main.rs
    //! use concrete_lib::*;
    //!
    //! // encoding settings
    //! const MIN_INTERVAL: f64 = 0.;
    //! const MAX_INTERVAL: f64 = 8.;
    //! const NB_BIT_PRECISION: usize = 5;
    //! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
    //! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
    //!
    //! // topology of the logistic regression
    //! const NB_PERCEPTRON: usize = 3;
    //! const NB_FEATURE: usize = 4;
    //!
    //! // weights
    //! const MAX_WEIGHT: f64 = 11.;
    //! const WEIGHTS: [f64; 12] = [
    //!     3.3758082444336295,
    //!     7.9196458076215945,
    //!     -10.774548167193158,
    //!     -5.487605086345206,
    //!     -1.1369996360241739,
    //!     -0.7553735705885082,
    //!     1.4312527569766305,
    //!     -5.064955905902094,
    //!     -2.238808608432861,
    //!     -7.164272237046656,
    //!     9.343295410205023,
    //!     10.552560992243956,
    //! ];
    //! const BIASES: [f64; 3] = [1.666627227331721, 18.918783641421552, -20.585410868758302];
    //!
    //! // validation set
    //! const VALIDATION_SET: [f64; 120] = [
    //!     5.3, 3.7, 1.5, 0.2, 5.7, 3.8, 1.7, 0.3, 5.2, 4.1, 1.5, 0.1, 7.4, 2.8, 6.1, 1.9, 7.0, 3.2,
    //!     4.7, 1.4, 5.2, 3.5, 1.5, 0.2, 6.4, 2.7, 5.3, 1.9, 5.6, 2.8, 4.9, 2.0, 5.5, 4.2, 1.4, 0.2,
    //!     6.7, 3.0, 5.0, 1.7, 7.7, 3.0, 6.1, 2.3, 5.5, 2.4, 3.8, 1.1, 6.1, 2.8, 4.0, 1.3, 6.1, 3.0,
    //!     4.9, 1.8, 4.5, 2.3, 1.3, 0.3, 6.4, 2.8, 5.6, 2.1, 5.4, 3.9, 1.3, 0.4, 5.7, 2.6, 3.5, 1.0,
    //!     6.7, 3.1, 4.4, 1.4, 7.6, 3.0, 6.6, 2.1, 5.2, 3.4, 1.4, 0.2, 5.8, 2.6, 4.0, 1.2, 5.1, 3.8,
    //!     1.5, 0.3, 7.2, 3.2, 6.0, 1.8, 4.8, 3.0, 1.4, 0.1, 6.5, 3.0, 5.5, 1.8, 6.0, 3.4, 4.5, 1.6,
    //!     5.1, 3.8, 1.9, 0.4, 6.3, 2.5, 4.9, 1.5, 5.9, 3.0, 5.1, 1.8,
    //! ];
    //! const VALIDATION_LABELS: [f64; 30] = [
    //!     0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0,
    //!     1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
    //! ];
    //!
    //! fn main() {
    //!     // work in progress :-p
    //! }
    //! ```
}

pub mod full_code_2 {
    //! The full code step by step: step 2
    //!
    //! Back to the [tutorial](super).
    //!
    //! ```rust
    //! /// file: main.rs
    //! use concrete_lib::*;
    //!
    //! // encoding settings
    //! const MIN_INTERVAL: f64 = 0.;
    //! const MAX_INTERVAL: f64 = 8.;
    //! const NB_BIT_PRECISION: usize = 5;
    //! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
    //! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
    //!
    //! // lwe settings
    //! const LWE_PARAMS: LWEParams = LWE128_688;
    //! const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
    //!
    //! // bootstrapping settings
    //! const BASE_LOG: usize = 8;
    //! const LEVEL: usize = 5;
    //!
    //! // topology of the logistic regression
    //! const NB_PERCEPTRON: usize = 3;
    //! const NB_FEATURE: usize = 4;
    //!
    //! // weights
    //! const MAX_WEIGHT: f64 = 11.;
    //! const WEIGHTS: [f64; 12] = [
    //!     3.3758082444336295,
    //!     7.9196458076215945,
    //!     -10.774548167193158,
    //!     -5.487605086345206,
    //!     -1.1369996360241739,
    //!     -0.7553735705885082,
    //!     1.4312527569766305,
    //!     -5.064955905902094,
    //!     -2.238808608432861,
    //!     -7.164272237046656,
    //!     9.343295410205023,
    //!     10.552560992243956,
    //! ];
    //! const BIASES: [f64; 3] = [1.666627227331721, 18.918783641421552, -20.585410868758302];
    //!
    //! // validation set
    //! const VALIDATION_SET: [f64; 120] = [
    //!     5.3, 3.7, 1.5, 0.2, 5.7, 3.8, 1.7, 0.3, 5.2, 4.1, 1.5, 0.1, 7.4, 2.8, 6.1, 1.9, 7.0, 3.2,
    //!     4.7, 1.4, 5.2, 3.5, 1.5, 0.2, 6.4, 2.7, 5.3, 1.9, 5.6, 2.8, 4.9, 2.0, 5.5, 4.2, 1.4, 0.2,
    //!     6.7, 3.0, 5.0, 1.7, 7.7, 3.0, 6.1, 2.3, 5.5, 2.4, 3.8, 1.1, 6.1, 2.8, 4.0, 1.3, 6.1, 3.0,
    //!     4.9, 1.8, 4.5, 2.3, 1.3, 0.3, 6.4, 2.8, 5.6, 2.1, 5.4, 3.9, 1.3, 0.4, 5.7, 2.6, 3.5, 1.0,
    //!     6.7, 3.1, 4.4, 1.4, 7.6, 3.0, 6.6, 2.1, 5.2, 3.4, 1.4, 0.2, 5.8, 2.6, 4.0, 1.2, 5.1, 3.8,
    //!     1.5, 0.3, 7.2, 3.2, 6.0, 1.8, 4.8, 3.0, 1.4, 0.1, 6.5, 3.0, 5.5, 1.8, 6.0, 3.4, 4.5, 1.6,
    //!     5.1, 3.8, 1.9, 0.4, 6.3, 2.5, 4.9, 1.5, 5.9, 3.0, 5.1, 1.8,
    //! ];
    //! const VALIDATION_LABELS: [f64; 30] = [
    //!     0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0,
    //!     1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
    //! ];
    //!
    //! fn setup_gen() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     // generation of the keys
    //!     let lwe_key_input = LWESecretKey::new(&LWE_PARAMS);
    //!     let rlwe_secret_key = RLWESecretKey::new(&RLWE_PARAMS);
    //!     let lwe_key_output = rlwe_secret_key.to_lwe_secret_key();
    //!     let bootstrapping_key = LWEBSK::new(&lwe_key_input, &rlwe_secret_key, BASE_LOG, LEVEL);
    //!
    //!     // we write the keys in files
    //!     lwe_key_output.save("lwe_key_output.json").unwrap();
    //!     lwe_key_input.save("lwe_key_input.json").unwrap();
    //!     bootstrapping_key.write_in_file_bytes("bsk_bytes.txt");
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn setup_load() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     let bootstrapping_key = LWEBSK::read_in_file_bytes("bsk_bytes.txt");
    //!     let lwe_key_input = LWESecretKey::load("lwe_key_input.json").unwrap();
    //!     let lwe_key_output = LWESecretKey::load("lwe_key_output.json").unwrap();
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn encrypt_input(input: &[f64], lwe_secret_key: &LWESecretKey) -> VectorLWE {
    //!     // generate an encoder
    //!     let encoder =
    //!         Encoder::new(MIN_INTERVAL, MAX_INTERVAL, NB_BIT_PRECISION, NB_BIT_PADDING).unwrap();
    //!
    //!     // encode and encrypt
    //!     let ciphertexts = VectorLWE::encode_encrypt(lwe_secret_key, input, &encoder).unwrap();
    //!
    //!     ciphertexts
    //! }
    //!
    //! fn decrypt_inference(ciphertexts: &VectorLWE, lwe_secret_key: &LWESecretKey) -> usize {
    //!     // decrypt
    //!     let decryptions = ciphertexts.decrypt_decode(lwe_secret_key).unwrap();
    //!
    //!     // find the argmax
    //!     let mut res: usize = 0;
    //!     for (i, val) in decryptions.iter().enumerate() {
    //!         println!("dec {}", val);
    //!         if decryptions[res] < *val {
    //!             res = i;
    //!         }
    //!     }
    //!
    //!     res
    //! }
    //!
    //! fn main() {
    //!     // work in progress :-p
    //!
    //!     // but we want to test what we already have!
    //!
    //!     let (lwe_key, bootstrapping_key, lwe_key_output) = setup_gen();
    //!     // let (lwe_key, bootstrapping_key, lwe_key_output) = setup_load();
    //!
    //!     let input_id: usize = 1; // the 1th input from the validation set
    //!
    //!     let input = VALIDATION_SET // [5.7, 3.8, 1.7, 0.3]
    //!         .get((input_id * NB_FEATURE)..((input_id + 1) * NB_FEATURE))
    //!         .unwrap();
    //!
    //!     let ciphertexts = encrypt_input(&input, &lwe_key);
    //!
    //!     let argmax = decrypt_inference(&ciphertexts, &lwe_key);
    //!
    //!     assert_eq!(argmax, 0);
    //! }
    //! ```
}

pub mod full_code_3 {
    //! The full code step by step: step 3
    //!
    //! Back to the [tutorial](super).
    //!
    //! ```rust
    //! /// file: main.rs
    //! use concrete_lib::*;
    //!
    //! // encoding settings
    //! const MIN_INTERVAL: f64 = 0.;
    //! const MAX_INTERVAL: f64 = 8.;
    //! const NB_BIT_PRECISION: usize = 5;
    //! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
    //! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
    //!
    //! // lwe settings
    //! const LWE_PARAMS: LWEParams = LWE128_688;
    //! const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
    //!
    //! // bootstrapping settings
    //! const BASE_LOG: usize = 8;
    //! const LEVEL: usize = 5;
    //!
    //! // topology of the logistic regression
    //! const NB_PERCEPTRON: usize = 3;
    //! const NB_FEATURE: usize = 4;
    //!
    //! // weights
    //! const MAX_WEIGHT: f64 = 11.;
    //! const WEIGHTS: [f64; 12] = [
    //!     3.3758082444336295,
    //!     7.9196458076215945,
    //!     -10.774548167193158,
    //!     -5.487605086345206,
    //!     -1.1369996360241739,
    //!     -0.7553735705885082,
    //!     1.4312527569766305,
    //!     -5.064955905902094,
    //!     -2.238808608432861,
    //!     -7.164272237046656,
    //!     9.343295410205023,
    //!     10.552560992243956,
    //! ];
    //! const BIASES: [f64; 3] = [1.666627227331721, 18.918783641421552, -20.585410868758302];
    //!
    //! // validation set
    //! const VALIDATION_SET: [f64; 120] = [
    //!     5.3, 3.7, 1.5, 0.2, 5.7, 3.8, 1.7, 0.3, 5.2, 4.1, 1.5, 0.1, 7.4, 2.8, 6.1, 1.9, 7.0, 3.2,
    //!     4.7, 1.4, 5.2, 3.5, 1.5, 0.2, 6.4, 2.7, 5.3, 1.9, 5.6, 2.8, 4.9, 2.0, 5.5, 4.2, 1.4, 0.2,
    //!     6.7, 3.0, 5.0, 1.7, 7.7, 3.0, 6.1, 2.3, 5.5, 2.4, 3.8, 1.1, 6.1, 2.8, 4.0, 1.3, 6.1, 3.0,
    //!     4.9, 1.8, 4.5, 2.3, 1.3, 0.3, 6.4, 2.8, 5.6, 2.1, 5.4, 3.9, 1.3, 0.4, 5.7, 2.6, 3.5, 1.0,
    //!     6.7, 3.1, 4.4, 1.4, 7.6, 3.0, 6.6, 2.1, 5.2, 3.4, 1.4, 0.2, 5.8, 2.6, 4.0, 1.2, 5.1, 3.8,
    //!     1.5, 0.3, 7.2, 3.2, 6.0, 1.8, 4.8, 3.0, 1.4, 0.1, 6.5, 3.0, 5.5, 1.8, 6.0, 3.4, 4.5, 1.6,
    //!     5.1, 3.8, 1.9, 0.4, 6.3, 2.5, 4.9, 1.5, 5.9, 3.0, 5.1, 1.8,
    //! ];
    //! const VALIDATION_LABELS: [f64; 30] = [
    //!     0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0,
    //!     1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
    //! ];
    //!
    //! fn setup_gen() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     // generation of the keys
    //!     let lwe_key_input = LWESecretKey::new(&LWE_PARAMS);
    //!     let rlwe_secret_key = RLWESecretKey::new(&RLWE_PARAMS);
    //!     let lwe_key_output = rlwe_secret_key.to_lwe_secret_key();
    //!     let bootstrapping_key = LWEBSK::new(&lwe_key_input, &rlwe_secret_key, BASE_LOG, LEVEL);
    //!
    //!     // we write the keys in files
    //!     lwe_key_output.save("lwe_key_output.json").unwrap();
    //!     lwe_key_input.save("lwe_key_input.json").unwrap();
    //!     bootstrapping_key.write_in_file_bytes("bsk_bytes.txt");
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn setup_load() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     let bootstrapping_key = LWEBSK::read_in_file_bytes("bsk_bytes.txt");
    //!     let lwe_key_input = LWESecretKey::load("lwe_key_input.json").unwrap();
    //!     let lwe_key_output = LWESecretKey::load("lwe_key_output.json").unwrap();
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn encrypt_input(input: &[f64], lwe_secret_key: &LWESecretKey) -> VectorLWE {
    //!     // generate an encoder
    //!     let encoder =
    //!         Encoder::new(MIN_INTERVAL, MAX_INTERVAL, NB_BIT_PRECISION, NB_BIT_PADDING).unwrap();
    //!
    //!     // encode and encrypt
    //!     let ciphertexts = VectorLWE::encode_encrypt(lwe_secret_key, input, &encoder).unwrap();
    //!
    //!     ciphertexts
    //! }
    //!
    //! fn decrypt_inference(ciphertexts: &VectorLWE, lwe_secret_key: &LWESecretKey) -> usize {
    //!     // decrypt
    //!     let decryptions = ciphertexts.decrypt_decode(lwe_secret_key).unwrap();
    //!
    //!     // find the argmax
    //!     let mut res: usize = 0;
    //!     for (i, val) in decryptions.iter().enumerate() {
    //!         println!("dec {}", val);
    //!         if decryptions[res] < *val {
    //!             res = i;
    //!         }
    //!     }
    //!
    //!     res
    //! }
    //!
    //! fn homomorphic_perceptron(
    //!     ciphertexts: &mut VectorLWE,
    //!     key_switching_key: &LWEKSK,
    //!     bootstrapping_key: &LWEBSK,
    //!     weights: &[f64],
    //!     bias: f64,
    //! ) -> VectorLWE {
    //!     // clone the ciphertexts
    //!     let mut ct = ciphertexts.clone();
    //!
    //!     // multiply each ciphertext with a weight
    //!     ct.mul_constant_with_padding_inplace(&weights, MAX_WEIGHT, NB_BIT_PRECISION)
    //!         .unwrap();
    //!
    //!     // sum them together
    //!     let mut sum = add_4(&ct);
    //!
    //!     // more homomorphic operations...
    //!
    //!     sum
    //! }
    //!
    //! fn add_4(ciphertexts: &VectorLWE) -> VectorLWE {
    //!     // first addition
    //!     let mut c0 = ciphertexts.extract_nth(0).unwrap();
    //!     let c1 = ciphertexts.extract_nth(1).unwrap();
    //!     c0.add_with_padding_inplace(&c1);
    //!
    //!     // second addition
    //!     let mut c2 = ciphertexts.extract_nth(2).unwrap();
    //!     let c3 = ciphertexts.extract_nth(3).unwrap();
    //!     c2.add_with_padding_inplace(&c3);
    //!
    //!     // last addition
    //!     c0.add_with_padding_inplace(&c2);
    //!
    //!     c0
    //! }
    //!
    //! fn main() {
    //!     // work in progress :-p
    //! }
    //! ```
}

pub mod full_code_4 {
    //! The full code step by step: step 4
    //!
    //! Back to the [tutorial](super).
    //!
    //! ```rust
    //! use concrete_lib::*;
    //! use itertools::izip;
    //!
    //! // encoding settings
    //! const MIN_INTERVAL: f64 = 0.;
    //! const MAX_INTERVAL: f64 = 8.;
    //! const NB_BIT_PRECISION: usize = 5;
    //! const NB_BIT_PRECISION_WEIGHTS: usize = 5;
    //! const NB_BIT_PADDING: usize = NB_BIT_PRECISION_WEIGHTS + 2 + 1;
    //!
    //! // lwe settings const
    //! const LWE_PARAMS: LWEParams = LWE128_688;
    //! const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
    //!
    //! // bootstrapping settings
    //! const BASE_LOG: usize = 8;
    //! const LEVEL: usize = 5;
    //!
    //! // topology of the logistic regression
    //! const NB_PERCEPTRON: usize = 3;
    //! const NB_FEATURE: usize = 4;
    //!
    //! // weights
    //! const MAX_WEIGHT: f64 = 10.8;
    //! const WEIGHTS: [f64; 12] = [
    //!     3.3758082444336295,
    //!     7.9196458076215945,
    //!     -10.774548167193158,
    //!     -5.487605086345206,
    //!     -1.1369996360241739,
    //!     -0.7553735705885082,
    //!     1.4312527569766305,
    //!     -5.064955905902094,
    //!     -2.238808608432861,
    //!     -7.164272237046656,
    //!     9.343295410205023,
    //!     10.552560992243956,
    //! ];
    //! const BIASES: [f64; 3] = [1.666627227331721, 18.918783641421552, -20.585410868758302];
    //!
    //! // validation set
    //! const VALIDATION_SET: [f64; 120] = [
    //!     5.3, 3.7, 1.5, 0.2, 5.7, 3.8, 1.7, 0.3, 5.2, 4.1, 1.5, 0.1, 7.4, 2.8, 6.1, 1.9, 7.0, 3.2,
    //!     4.7, 1.4, 5.2, 3.5, 1.5, 0.2, 6.4, 2.7, 5.3, 1.9, 5.6, 2.8, 4.9, 2.0, 5.5, 4.2, 1.4, 0.2,
    //!     6.7, 3.0, 5.0, 1.7, 7.7, 3.0, 6.1, 2.3, 5.5, 2.4, 3.8, 1.1, 6.1, 2.8, 4.0, 1.3, 6.1, 3.0,
    //!     4.9, 1.8, 4.5, 2.3, 1.3, 0.3, 6.4, 2.8, 5.6, 2.1, 5.4, 3.9, 1.3, 0.4, 5.7, 2.6, 3.5, 1.0,
    //!     6.7, 3.1, 4.4, 1.4, 7.6, 3.0, 6.6, 2.1, 5.2, 3.4, 1.4, 0.2, 5.8, 2.6, 4.0, 1.2, 5.1, 3.8,
    //!     1.5, 0.3, 7.2, 3.2, 6.0, 1.8, 4.8, 3.0, 1.4, 0.1, 6.5, 3.0, 5.5, 1.8, 6.0, 3.4, 4.5, 1.6,
    //!     5.1, 3.8, 1.9, 0.4, 6.3, 2.5, 4.9, 1.5, 5.9, 3.0, 5.1, 1.8,
    //! ];
    //! const VALIDATION_LABELS: [f64; 30] = [
    //!     0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0,
    //!     1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
    //! ];
    //!
    //! fn setup_gen() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     // generation of the keys
    //!     let lwe_key_input = LWESecretKey::new(&LWE_PARAMS);
    //!     let rlwe_secret_key = RLWESecretKey::new(&RLWE_PARAMS);
    //!     let lwe_key_output = rlwe_secret_key.to_lwe_secret_key();
    //!     let bootstrapping_key = LWEBSK::new(&lwe_key_input, &rlwe_secret_key, BASE_LOG, LEVEL);
    //!
    //!     // we write the keys in files
    //!     lwe_key_output.save("lwe_key_output.json").unwrap();
    //!     lwe_key_input.save("lwe_key_input.json").unwrap();
    //!     bootstrapping_key.write_in_file_bytes("bsk_bytes.txt");
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn setup_load() -> (LWESecretKey, LWEBSK, LWESecretKey) {
    //!     let bootstrapping_key = LWEBSK::read_in_file_bytes("bsk_bytes.txt");
    //!     let lwe_key_input = LWESecretKey::load("lwe_key_input.json").unwrap();
    //!     let lwe_key_output = LWESecretKey::load("lwe_key_output.json").unwrap();
    //!
    //!     (lwe_key_input, bootstrapping_key, lwe_key_output)
    //! }
    //!
    //! fn encrypt_input(input: &[f64], lwe_secret_key: &LWESecretKey) -> VectorLWE {
    //!     // generate an encoder
    //!     let encoder =
    //!         Encoder::new(MIN_INTERVAL, MAX_INTERVAL, NB_BIT_PRECISION, NB_BIT_PADDING).unwrap();
    //!
    //!     // encode and encrypt
    //!     let ciphertexts = VectorLWE::encode_encrypt(lwe_secret_key, input, &encoder).unwrap();
    //!
    //!     ciphertexts
    //! }
    //!
    //! fn decrypt_inference(ciphertexts: &VectorLWE, lwe_secret_key: &LWESecretKey) -> usize {
    //!     // decrypt
    //!     let decryptions = ciphertexts.decrypt_decode(lwe_secret_key).unwrap();
    //!
    //!     // find the argmax
    //!     let mut res: usize = 0;
    //!     for (i, val) in decryptions.iter().enumerate() {
    //!         if decryptions[res] < *val {
    //!             res = i;
    //!         }
    //!     }
    //!
    //!     res
    //! }
    //!
    //! fn homomorphic_perceptron(
    //!     ciphertexts: &VectorLWE,
    //!     bootstrapping_key: &LWEBSK,
    //!     weights: &[f64],
    //!     bias: f64,
    //! ) -> VectorLWE {
    //!     // clone the ciphertexts
    //!     let mut ct = ciphertexts.clone();
    //!
    //!     // multiply each ciphertext with a weight
    //!     ct.mul_constant_with_padding_inplace(&weights, MAX_WEIGHT, NB_BIT_PRECISION)
    //!         .unwrap();
    //!
    //!     // sum them together
    //!     let sum = add_4(&ct);
    //!
    //!     // output interval
    //!     let encoder_output = Encoder::new(0., 1., NB_BIT_PRECISION, 0).unwrap();
    //!
    //!     // bootstrap
    //!     let result = sum
    //!         .bootstrap_nth_with_function(
    //!             bootstrapping_key,
    //!             |x| 1. / (1. + (-x - bias).exp()), // the sigmoid function
    //!             &encoder_output,                   // contains the output of the sigmoid function
    //!             0,                                 // the 0th LWE ciphertext in the structure
    //!         )
    //!         .unwrap();
    //!
    //!     result
    //! }
    //!
    //! fn add_4(ciphertexts: &VectorLWE) -> VectorLWE {
    //!     // first addition
    //!     let mut c0 = ciphertexts.extract_nth(0).unwrap();
    //!     let c1 = ciphertexts.extract_nth(1).unwrap();
    //!     c0.add_with_padding_inplace(&c1).unwrap();
    //!
    //!     // second addition
    //!     let mut c2 = ciphertexts.extract_nth(2).unwrap();
    //!     let c3 = ciphertexts.extract_nth(3).unwrap();
    //!     c2.add_with_padding_inplace(&c3).unwrap();
    //!
    //!     // last addition
    //!     c0.add_with_padding_inplace(&c2).unwrap();
    //!
    //!     c0
    //! }
    //!
    //! fn homomorphic_inference(input_ciphertexts: &VectorLWE, bootstrapping_key: &LWEBSK) -> VectorLWE {
    //!     // allocation of the result with zeros
    //!     let mut result = VectorLWE::zero(
    //!         bootstrapping_key.dimension * bootstrapping_key.polynomial_size,
    //!         NB_PERCEPTRON,
    //!     )
    //!     .unwrap();
    //!
    //!     // compute for each perceptron
    //!     for (weights, bias, i) in izip!(WEIGHTS.chunks(NB_FEATURE), BIASES.iter(), 0..NB_PERCEPTRON)
    //!     {
    //!         let tmp = homomorphic_perceptron(input_ciphertexts, bootstrapping_key, weights, *bias);
    //!
    //!         // copy the output of the perceptron inside result
    //!         result.copy_in_nth_nth_inplace(i, &tmp, 0).unwrap();
    //!     }
    //!
    //!     result
    //! }
    //!
    //! fn main() {
    //!     // start to count seconds
    //!     use std::time::SystemTime;
    //!     let mut now = SystemTime::now();
    //!
    //!     // secret key generation
    //!     let (lwe_key, bootstrapping_key, lwe_key_output) = setup_gen();
    //!     // let (lwe_key, bootstrapping_key, lwe_key_output) = setup_load();
    //!
    //!     // print the time to generate or load the key
    //!     println!(
    //!         "time to generate/load the keys: {}",
    //!         now.elapsed().unwrap().as_secs()
    //!     );
    //!
    //!     for test_id in 0..10 {
    //!         // reset the second counter
    //!         now = SystemTime::now();
    //!
    //!         // prediction of each element of the test set
    //!         let mut accuracy: usize = 0;
    //!         for input_id in 0..VALIDATION_LABELS.len() {
    //!             // select an input
    //!             // let input_id: usize = 1; // the 1th input from the validation set
    //!             let input =
    //!                VALIDATION_SET // [5.7, 3.8, 1.7, 0.3]
    //!                    .get((input_id * NB_FEATURE)..((input_id + 1) * NB_FEATURE))
    //!                    .unwrap();
    //!
    //!             // encrypt the input
    //!             let ciphertexts = encrypt_input(&input, &lwe_key);
    //!
    //!             // homomorphicinference
    //!             let output = homomorphic_inference(&ciphertexts, &bootstrapping_key);
    //!
    //!             // decrypt and argmax
    //!             let argmax: usize = decrypt_inference(&output, &lwe_key_output);
    //!
    //!             if (VALIDATION_LABELS[input_id] as usize) == argmax {
    //!                 accuracy += 1;
    //!             }
    //!         }
    //!
    //!         // print the time needed to predict the whole test set
    //!         println!(
    //!             "time to predict each element of the test set: {}",
    //!             now.elapsed().unwrap().as_secs()
    //!         );
    //!
    //!         // print the accuracy
    //!         println!(
    //!             "[{}]accuracy {}",
    //!             test_id,
    //!             (accuracy as f64) / (VALIDATION_LABELS.len() as f64)
    //!         );
    //!     }
    //! }
    //! ```
}
