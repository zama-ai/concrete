#[cfg(feature = "seeder_x86_64_rdseed")]
mod rdseed;
#[cfg(feature = "seeder_x86_64_rdseed")]
pub use rdseed::RdseedSeeder;

#[cfg(feature = "seeder_unix")]
mod unix;
#[cfg(feature = "seeder_unix")]
pub use unix::UnixSeeder;
