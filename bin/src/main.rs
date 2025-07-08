mod app;

use anyhow::{Result, anyhow};
use clap::Parser;
use winit::event_loop::EventLoop;

use crate::app::App;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path
    #[arg(short, long, default_value = "assets/final-one-weekend.json")]
    path: String,
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&cli.path)?;
    event_loop.run_app(&mut app).map_err(|e| anyhow!("{e:?}"))
}
