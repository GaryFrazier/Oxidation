mod engine;

fn main() {
    pollster::block_on(engine::init::run());
}

