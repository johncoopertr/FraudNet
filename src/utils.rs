/// Simple pseudo-random number generator (LCG)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    pub fn next(&mut self) -> u64 {
        // Linear congruential generator constants
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        // Generate a float between 0 and 1
        (self.next() as f64) / (u64::MAX as f64)
    }
}
