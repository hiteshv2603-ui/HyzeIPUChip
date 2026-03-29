//! Hyze Prompt Guard - Hardware-accelerated prompt injection detection.

pub enum GuardResult {
    Safe(Vec<u8>),
    Blocked(&'static str),
}

pub struct IpuFilter;
impl IpuFilter {
    pub fn scan(&self, bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
        Ok(bytes.to_vec())
    }
}

pub struct HyzePromptGuard {
    injection_patterns: Vec<String>,
    token_entropy: f32,
    ipu_filter: IpuFilter,
}

impl HyzePromptGuard {
    pub fn new(patterns: Vec<String>) -> Self {
        Self { injection_patterns: patterns, token_entropy: 0.0, ipu_filter: IpuFilter }
    }

    pub fn scan_injection(&mut self, prompt: &str) -> GuardResult {
        for pattern in &self.injection_patterns {
            if prompt.contains(pattern.as_str()) {
                return GuardResult::Blocked("Injection pattern matched");
            }
        }
        let entropy = calculate_shannon_entropy(prompt);
        self.token_entropy = entropy;
        if entropy > 4.5 {
            return GuardResult::Blocked("High entropy evasion");
        }
        // BUG FIX: original called `self.ipu_filter(prompt.as_bytes())?` but
        // ipu_filter is a field not a method, and ? requires a Result return type.
        match self.ipu_filter.scan(prompt.as_bytes()) {
            Ok(safe_tokens) => GuardResult::Safe(safe_tokens),
            Err(reason) => GuardResult::Blocked(reason),
        }
    }
}

fn calculate_shannon_entropy(s: &str) -> f32 {
    if s.is_empty() { return 0.0; }
    let mut counts = [0u32; 256];
    for b in s.bytes() { counts[b as usize] += 1; }
    let len = s.len() as f32;
    counts.iter().filter(|&&c| c > 0).map(|&c| { let p = c as f32 / len; -p * p.log2() }).sum()
}
