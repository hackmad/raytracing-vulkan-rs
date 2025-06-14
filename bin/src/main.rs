use app::App;
use winit::{error::EventLoopError, event_loop::EventLoop};

mod app;

fn main() -> Result<(), EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop, false);
    event_loop.run_app(&mut app)
}
