// Llama 3.2 3B inference engine.
//
// Loads model weights + tokenizer from HuggingFace, then runs autoregressive
// generation with KV-cache acceleration.
//
// Prerequisites:
//   1. `huggingface-cli login` to authenticate (the model requires a license grant)
//   2. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B
//
// The model is ~6 GB. First run downloads it (~1-5 min depending on connection).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks},
};
use hf_hub::api::sync::{Api, ApiRepo};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// ── shared config ────────────────────────────────────────────────────────────

const MODEL_ID: &str = "meta-llama/Llama-3.2-3B";

struct GenConfig {
    max_new_tokens: usize,
    seed: u64,
    temperature: f64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self { max_new_tokens: 100, seed: 42, temperature: 0.8 }
    }
}

// ── model loading ────────────────────────────────────────────────────────────

struct LoadedModel {
    llama: Llama,
    cache: Cache,
    tokenizer: Tokenizer,
    eos_token_id: Option<LlamaEosToks>,
    device: Device,
}

/// Download config/weights/tokenizer from HuggingFace, build the model + KV cache.
///
/// On first call: downloads ~6 GB and caches under ~/.cache/huggingface/hub/.
/// On subsequent calls: loads from the local cache almost instantly.
/// Load all safetensors shards listed in `model.safetensors.index.json`.
///
/// Many 3B+ models split weights across multiple files.
/// The index.json maps each weight name to its file (e.g. "model-00001-of-00002.safetensors").
/// We collect the unique file names and download each one.
fn load_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    let index_path = repo.get("model.safetensors.index.json")?;
    let json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(index_path)?)?;

    let weight_map = json["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("no weight_map in safetensors index"))?;

    let mut files: HashSet<&str> = HashSet::new();
    for value in weight_map.values() {
        if let Some(name) = value.as_str() {
            files.insert(name);
        }
    }

    files.iter().map(|name| Ok(repo.get(name)?)).collect()
}

fn load_model() -> Result<LoadedModel> {
    let device = Device::Cpu; // switch to Device::Cuda(0) once CUDA is wired up
    let dtype = DType::F16; // half-precision: 6 GB instead of 12 GB of VRAM

    println!("downloading model from HuggingFace ({MODEL_ID})...");

    let api = Api::new()?;
    let repo = api.model(MODEL_ID.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;

    // try sharded weights first (most 3B+ models), fall back to a single file
    let model_paths = load_safetensors_shards(&repo)
        .unwrap_or_else(|_| vec![repo.get("model.safetensors").unwrap()]);

    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let config = config.into_config(false);

    println!("loading model weights into memory...");

    // mmap-backed: weights demand-paged by the OS, no upfront copy
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_paths, dtype, &device)? };
    let llama = Llama::load(vb, &config)?;

    // pre-allocate KV cache buffers for the full context window
    let cache = Cache::new(true, dtype, &config, &device)?;
    let eos_token_id = config.eos_token_id.clone();

    println!("loading tokenizer...");

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

    Ok(LoadedModel { llama, cache, tokenizer, eos_token_id, device })
}

// ── generation loop ──────────────────────────────────────────────────────────

/// Autoregressive generation with KV-cache acceleration.
///
/// Prefill (1st step): feeds the entire prompt through the model; KV cache is populated.
/// Decode (remaining steps): only the single newest token passes through;
/// past keys and values are served from the cache — O(n) per step instead of O(n²).
fn generate(model: &mut LoadedModel, prompt: &str, gen_cfg: &GenConfig) -> Result<()> {
    let mut tokens = model
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?
        .get_ids()
        .to_vec();

    // temperature scaling + top-p / argmax sampling
    let mut sampler = LogitsProcessor::new(gen_cfg.seed, Some(gen_cfg.temperature), None);

    print!("{prompt}");
    std::io::stdout().flush()?;

    // absolute position in the sequence (used as the KV cache index)
    let mut pos = 0;

    for step in 0..gen_cfg.max_new_tokens {
        let (context_len, cache_index) = if step == 0 {
            (tokens.len(), 0) // prefill: process full prompt, cache at index 0
        } else {
            (1, pos) // decode: only the newest token, cache at current position
        };

        let recent = &tokens[tokens.len() - context_len..];
        let input = Tensor::new(recent, &model.device)?.unsqueeze(0)?; // [1, seq]

        let logits = model.llama.forward(&input, cache_index, &mut model.cache)?;
        let logits = logits.squeeze(0)?; // [seq, vocab] → [vocab]
        pos += context_len;

        let next_token = sampler.sample(&logits)?;
        tokens.push(next_token);

        // halt on EOS
        match &model.eos_token_id {
            Some(LlamaEosToks::Single(id)) if next_token == *id => break,
            Some(LlamaEosToks::Multiple(ids)) if ids.contains(&next_token) => break,
            _ => {}
        }

        // decode one token → text and stream to stdout
        let text = model
            .tokenizer
            .decode(&[next_token], false)
            .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))?;
        print!("{text}");
        std::io::stdout().flush()?;
    }

    println!();
    Ok(())
}

// ── entrypoint ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let mut model = load_model()?;
    let gen_cfg = GenConfig::default();
    generate(&mut model, "Once upon a time, ", &gen_cfg)
}
