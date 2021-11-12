use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::AbstractEntity;

engine_error! {
    DestructionError for DestructionEngine @
}

/// A trait for engines destructing entities.
///
/// # Semantics
///
/// This operation consumes and destroys the `entity` entity.
///
/// # Note on `Drop`
///
/// One may rightfully ask why not using the `Drop` trait instead of an explicit destructor like
/// this. The reason is that for backends handling special hardwares, the allocator is expected to
/// be contained in the engine, and as such, only the engine can properly dispose of the data
/// contained in the entity.
///
/// As a consequence, even if simply dropping an entity is not unsafe, forgetting to call `destroy`
/// will likely result in memory leaks.
pub trait DestructionEngine<Entity>: AbstractEngine
where
    Entity: AbstractEntity,
{
    /// Destroys an entity.
    fn destroy(&mut self, entity: Entity) -> Result<(), DestructionError<Self::EngineError>>;

    /// Unsafely destroys an entity.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`DestructionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn destroy_unchecked(&mut self, entity: Entity);
}
