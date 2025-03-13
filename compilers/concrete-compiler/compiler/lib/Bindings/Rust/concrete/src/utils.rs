pub trait GetInfo{
    type Output;
    fn get_info(&self) -> Self::Output;
}
