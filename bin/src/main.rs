mod app;

use clap::Parser;
use winit::{error::EventLoopError, event_loop::EventLoop};

use crate::app::App;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path
    #[arg(short, long, default_value = "assets/final-one-weekend.json")]
    path: String,
}

fn main() -> Result<(), EventLoopError> {
    env_logger::init();

    let cli = Cli::parse();

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop, false, &cli.path);
    event_loop.run_app(&mut app)
}
