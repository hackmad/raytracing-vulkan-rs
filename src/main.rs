use app::App;
use winit::{error::EventLoopError, event_loop::EventLoop};

mod app;
mod gui;
mod raytracer;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app)
}
