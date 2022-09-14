use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "internal-keycache")]
use concrete_integer::keycache::KEY_CACHE;
use concrete_integer::ServerKey;

use super::client_key::GenericIntegerClientKey;
use super::parameters::IntegerParameter;
use super::types::GenericInteger;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct GenericIntegerServerKey<P: IntegerParameter> {
    key: ServerKey,
    _marker: PhantomData<P>,
}

impl<P> GenericIntegerServerKey<P>
where
    P: IntegerParameter,
{
    pub(super) fn new(client_key: &GenericIntegerClientKey<P>) -> Self {
        #[cfg(feature = "internal-keycache")]
        {
            use super::IntegerParameterSet;
            match client_key.params {
                IntegerParameterSet::Radix(_) => {
                    let key = KEY_CACHE.get_from_params(client_key.key.parameters()).1;
                    Self {
                        key,
                        _marker: Default::default(),
                    }
                }
            }
        }
        #[cfg(not(feature = "internal-keycache"))]
        {
            Self {
                key: ServerKey::new(&client_key.key),
                _marker: Default::default(),
            }
        }
    }

    pub(super) fn smart_neg(&self, lhs: &GenericInteger<P>) -> GenericInteger<P> {
        let ciphertext = self.key.smart_neg(&mut lhs.ciphertext.borrow_mut());
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_add(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_add(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_sub(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_sub(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_mul(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_mul(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_bitand(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_bitand(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_bitor(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_bitor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_bitxor(
        &self,
        lhs: &GenericInteger<P>,
        rhs: &GenericInteger<P>,
    ) -> GenericInteger<P> {
        let ciphertext = self.key.smart_bitxor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_add_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_add_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_sub_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_sub_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_mul_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_mul_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_bitand_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_bitand_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_bitor_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_bitor(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_bitxor_assign(&self, lhs: &GenericInteger<P>, rhs: &GenericInteger<P>) {
        self.key.smart_bitxor_assign(
            &mut lhs.ciphertext.borrow_mut(),
            &mut rhs.ciphertext.borrow_mut(),
        );
    }

    pub(super) fn smart_scalar_add(&self, lhs: &GenericInteger<P>, rhs: u64) -> GenericInteger<P> {
        let ciphertext = self
            .key
            .smart_scalar_add(&mut lhs.ciphertext.borrow_mut(), rhs);
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_scalar_sub(&self, lhs: &GenericInteger<P>, rhs: u64) -> GenericInteger<P> {
        let ciphertext = self
            .key
            .smart_scalar_sub(&mut lhs.ciphertext.borrow_mut(), rhs);
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_scalar_mul(&self, lhs: &GenericInteger<P>, rhs: u64) -> GenericInteger<P> {
        let ciphertext = self
            .key
            .smart_scalar_mul(&mut lhs.ciphertext.borrow_mut(), rhs);
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn unchecked_scalar_left_shift(
        &self,
        lhs: &GenericInteger<P>,
        rhs: u64,
    ) -> GenericInteger<P> {
        let ciphertext = self
            .key
            .unchecked_scalar_left_shift(&lhs.ciphertext.borrow(), rhs as usize);
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn unchecked_scalar_right_shift(
        &self,
        lhs: &GenericInteger<P>,
        rhs: u64,
    ) -> GenericInteger<P> {
        let ciphertext = self
            .key
            .unchecked_scalar_right_shift(&lhs.ciphertext.borrow(), rhs as usize);
        GenericInteger::<P>::new(ciphertext, lhs.id)
    }

    pub(super) fn smart_scalar_add_assign(&self, lhs: &GenericInteger<P>, rhs: u64) {
        self.key
            .smart_scalar_add_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    pub(super) fn smart_scalar_sub_assign(&self, lhs: &GenericInteger<P>, rhs: u64) {
        self.key
            .smart_scalar_sub_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    pub(super) fn smart_scalar_mul_assign(&self, lhs: &GenericInteger<P>, rhs: u64) {
        self.key
            .smart_scalar_mul_assign(&mut lhs.ciphertext.borrow_mut(), rhs);
    }

    pub(super) fn unchecked_scalar_left_shift_assign(&self, lhs: &GenericInteger<P>, rhs: u64) {
        self.key
            .unchecked_scalar_left_shift_assign(&mut lhs.ciphertext.borrow_mut(), rhs as usize);
    }

    pub(super) fn unchecked_scalar_right_shift_assign(&self, lhs: &GenericInteger<P>, rhs: u64) {
        self.key
            .unchecked_scalar_right_shift_assign(&mut lhs.ciphertext.borrow_mut(), rhs as usize);
    }
}
