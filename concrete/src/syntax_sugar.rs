//! This module contains macros used to provide some [syntactic-sugar]
//!
//!
//! [syntactic-sugar]: https://en.wikipedia.org/wiki/Syntactic_sugar

/// **experimental** syntax sugar macro to write conditions with booleans
///
/// It allows the use of symbols like `&&`, `||`, `==`,`!=`.
///
/// Since `&&` and `||` are not overload-able and `==`, `!=` requires to return a [bool]
/// (not a FHE boolean), it is the only way these symbols can be used with our FHE types.
#[cfg_attr(doc, cfg(feature = "experimental_syntax_sugar"))]
#[macro_export]
macro_rules! condition {
    // Rule that handles &&
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( && $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* & )
            rest = ( $($rest)* )
        )
    };

    // Rule that handles ||
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( || $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* | )
            rest = ( $($rest)* )
        )
    };

    // Rule that handles !=
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( $lhs_cond:ident != $rhs_cond:ident $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* $lhs_cond.neq(&$rhs_cond) )
            rest = ( $($rest)* )
        )
    };

    // Rule that handles ==
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( $lhs_cond:ident == $rhs_cond:ident $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* $lhs_cond.eq(&$rhs_cond) )
            rest = ( $($rest)* )
        )
    };
    // Handle parenthesis `(..)` by recursively calling our macro
    //
    // whats better, this or having 2 rules, one for `(` and the
    // other `)`
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( ($($a:tt)*) $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* (condition!( $($a)* )) )
            rest = ( $($rest)* )
        )
    };

    // If we have an ident, we stack a reference to it
    // in the processed thing.
    //
    // The goal is that we want ops to take by ref and not move the values
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( $a:ident $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* &$a )
            rest = ( $($rest)* )
        )
    };

    // Rule that advances by one token
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ( $a:tt $($rest:tt)* )
    ) => {
        condition!(
            @internal
            processed = ( $($processed)* $a )
            rest = ( $($rest)* )
        )
    };

    // Termination rule
    // where we expand the tokens we processed
    (
        @internal
        processed = ( $($processed:tt)* )
        rest = ()
    ) => {
        $($processed)*
    };

    // Entry point of the macro, must be the last rule as this one matches
    // any input possible.
    (
        $($tokens:tt)*
    ) => {
        condition!(
            @internal
            processed = ()
            rest = ( $($tokens)* )
        )
    };
}
/// **experimental** syntax sugar macro to write if-else branches with booleans
///
/// As data is encrypted, no real branching can happen, and so both of the code in the
/// `if` and `else` branches are executed.
///
/// # Example
///
/// ```
/// # #![cfg(all(feature = "booleans", feature = "experimental_syntax_sugar"))]
/// # {
/// use concrete::prelude::*;
/// use concrete::{branch, condition, generate_keys, set_server_key, ConfigBuilder, FheBool};
///
/// let config = ConfigBuilder::all_disabled().enable_default_bool().build();
///
/// let (client_key, server_key) = generate_keys(config);
///
/// set_server_key(server_key);
/// let a = FheBool::encrypt(true, &client_key);
/// let b = FheBool::encrypt(false, &client_key);
/// let c = FheBool::encrypt(true, &client_key);
///
/// let r = branch!(if a { b } else { c });
///
/// let r_decrypted = r.decrypt(&client_key);
/// assert_eq!(r_decrypted, false); // b
/// # }
/// ```
#[cfg_attr(doc, cfg(feature = "experimental_syntax_sugar"))]
#[macro_export]
macro_rules! branch {
    // Terminal rule
    (
        @internal
        condition = ( $($cond_token:tt)* )
        rest = ( { $($then:tt)* } else { $($else:tt)* } )
    ) => {
        ::concrete::if_then_else(
            $crate::condition!($($cond_token)*),
            { $($then)* },
            { $($else)* }
        )
    };

    // rule that munches the next token
    (
        @internal
        condition = ( $($cond_token:tt)* )
        rest = ( $next:tt $($rest:tt)* )
    ) => {
        branch!(
            @internal
            condition = ( $($cond_token)* $next )
            rest = ( $($rest)* )
        );
    };

    // Entry point rule
    (
        if $($rest:tt)*
    ) => {
        branch!(
            @internal
            condition = ()
            rest = ( $($rest)* )
        )
    };
}
