use super::client_key::GenericBoolClientKey;
use super::parameters::BooleanParameterSet;
use super::types::GenericBool;
use concrete_boolean::server_key::{BinaryBooleanGates, ServerKey};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(doc, cfg(feature = "booleans"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct GenericBoolServerKey<P>
where
    P: BooleanParameterSet,
{
    pub(in crate::booleans) key: ServerKey,
    _marker: std::marker::PhantomData<P>,
}

impl<P> GenericBoolServerKey<P>
where
    P: BooleanParameterSet,
{
    pub(crate) fn new(key: &GenericBoolClientKey<P>) -> Self {
        Self {
            key: ServerKey::new(&key.key),
            _marker: Default::default(),
        }
    }

    pub(in crate::booleans) fn and(
        &mut self,
        lhs: &GenericBool<P>,
        rhs: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.and(&lhs.ciphertext, &rhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn or(
        &mut self,
        lhs: &GenericBool<P>,
        rhs: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.or(&lhs.ciphertext, &rhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn xor(
        &mut self,
        lhs: &GenericBool<P>,
        rhs: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.xor(&lhs.ciphertext, &rhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn xnor(
        &mut self,
        lhs: &GenericBool<P>,
        rhs: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.xnor(&lhs.ciphertext, &rhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn nand(
        &mut self,
        lhs: &GenericBool<P>,
        rhs: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.nand(&lhs.ciphertext, &rhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn not(&mut self, lhs: &GenericBool<P>) -> GenericBool<P> {
        let ciphertext = self.key.not(&lhs.ciphertext);
        GenericBool::<P>::new(ciphertext, lhs.id)
    }

    pub(in crate::booleans) fn mux(
        &mut self,
        condition: &GenericBool<P>,
        then_result: &GenericBool<P>,
        else_result: &GenericBool<P>,
    ) -> GenericBool<P> {
        let ciphertext = self.key.mux(
            &condition.ciphertext,
            &then_result.ciphertext,
            &else_result.ciphertext,
        );
        GenericBool::<P>::new(ciphertext, condition.id)
    }
}
