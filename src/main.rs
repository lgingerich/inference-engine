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
//
// Modes:
//   cargo run                       — single prompt, prints timing
//   cargo run -- --serve            — OpenAI-compatible HTTP API on :3000
//   cargo run -- --serve --port 8080

mod api;
mod llama;

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use tokenizers::Tokenizer;

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features `cuda` and `metal` are mutually exclusive");

// ── shared config ────────────────────────────────────────────────────────────

pub(crate) const MODEL_ID: &str = "meta-llama/Llama-3.2-3B";

#[derive(Parser)]
#[command(name = "infer")]
struct Cli {
    /// Run as an HTTP server instead of a single prompt
    #[arg(long)]
    serve: bool,

    /// Port for the HTTP server
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Prompt to generate from (CLI mode only)
    #[arg(long, default_value = "Once upon a time, ")]
    prompt: String,
}

#[derive(Clone)]
pub(crate) struct GenConfig {
    pub max_new_tokens: usize,
    pub seed: u64,
    pub temperature: f64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            seed: 42,
            temperature: 0.8,
        }
    }
}

// ── model loading ────────────────────────────────────────────────────────────

pub(crate) struct LoadedModel {
    llama: Llama,
    model_config: llama::Config,
    dtype: DType,
    tokenizer: Tokenizer,
    eos_token_id: Option<LlamaEosToks>,
    device: Device,
}

impl LoadedModel {
    /// Create a fresh KV cache for a new generation request.
    pub(crate) fn new_cache(&self) -> Result<Cache> {
        Cache::new(true, self.dtype, &self.model_config, &self.device).map_err(anyhow::Error::from)
    }
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
    let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;

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

fn hf_token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

pub(crate) fn load_model(device: &Device) -> Result<LoadedModel> {
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16 // meta-llama/Llama-3.2-3B weights are BF16; CPU matmul needs F32.
    };

    println!("downloading model from HuggingFace ({MODEL_ID})...");

    let api = ApiBuilder::new().with_token(hf_token_from_env()).build()?;
    let repo = api.model(MODEL_ID.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;

    // try sharded weights first (most 3B+ models), fall back to a single file
    let model_paths = load_safetensors_shards(&repo)
        .unwrap_or_else(|_| vec![repo.get("model.safetensors").unwrap()]);

    let model_config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let config = model_config.into_config(false);

    println!("loading model weights into memory...");

    // mmap-backed: weights demand-paged by the OS, no upfront copy
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_paths, dtype, device)? };
    let llama = Llama::load(vb, &config)?;

    let eos_token_id = config.eos_token_id.clone();

    println!("loading tokenizer...");

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

    Ok(LoadedModel {
        llama,
        model_config: config,
        dtype,
        tokenizer,
        eos_token_id,
        device: device.clone(),
    })
}

// ── generation ───────────────────────────────────────────────────────────────

pub(crate) struct GenMetrics {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub ttft: Duration,
    pub decode_tokens: u32,
    pub total_time: Duration,
}

/// Autoregressive generation with KV-cache acceleration.
///
/// Prefill (1st step): feeds the entire prompt through the model; KV cache is populated.
/// Decode (remaining steps): only the single newest token passes through;
/// past keys and values are served from the cache — O(n) per step instead of O(n²).
///
/// Calls `on_token` with each decoded text fragment as it streams.
pub(crate) fn generate(
    model: &mut LoadedModel,
    cache: &mut Cache,
    prompt: &str,
    gen_cfg: &GenConfig,
    on_token: &mut dyn FnMut(&str),
) -> Result<GenMetrics> {
    let mut tokens = model
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?
        .get_ids()
        .to_vec();
    let prompt_tokens = tokens.len() as u32;

    // temperature scaling + top-p / argmax sampling
    let mut sampler = LogitsProcessor::new(gen_cfg.seed, Some(gen_cfg.temperature), None);

    // absolute position in the sequence (used as the KV cache index)
    let mut pos = 0;

    let start = Instant::now();
    let mut ttft = None;
    let mut generated_tokens = 0u32;
    let mut decode_tokens = 0u32;

    for step in 0..gen_cfg.max_new_tokens {
        let (context_len, cache_index) = if step == 0 {
            (tokens.len(), 0) // prefill: process full prompt, cache at index 0
        } else {
            (1, pos) // decode: only the newest token, cache at current position
        };

        let recent = &tokens[tokens.len() - context_len..];
        let input = Tensor::new(recent, &model.device)?.unsqueeze(0)?; // [1, seq]

        let logits = model.llama.forward(&input, cache_index, cache)?;
        let logits = logits.squeeze(0)?; // [seq, vocab] → [vocab]
        pos += context_len;

        let next_token = sampler.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        // record TTFT after prefill + first token sample
        if step == 0 {
            ttft = Some(start.elapsed());
        } else {
            decode_tokens += 1;
        }

        // halt on EOS
        match &model.eos_token_id {
            Some(LlamaEosToks::Single(id)) if next_token == *id => break,
            Some(LlamaEosToks::Multiple(ids)) if ids.contains(&next_token) => break,
            _ => {}
        }

        // decode one token → text and stream to caller
        let text = model
            .tokenizer
            .decode(&[next_token], false)
            .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))?;
        on_token(&text);
    }

    let total = start.elapsed();
    Ok(GenMetrics {
        prompt_tokens,
        generated_tokens,
        ttft: ttft.unwrap_or_default(),
        decode_tokens,
        total_time: total,
    })
}

fn select_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }

    #[cfg(feature = "metal")]
    {
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        Ok(Device::Cpu)
    }
}

// ── entrypoint ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = select_device()?;
    println!("device: {device:?}");

    let mut model = load_model(&device)?;

    if cli.serve {
        api::serve(model, cli.port).await?;
    } else {
        // ── CLI single-prompt mode ──────────────────────────────────────────
        print!("{}", cli.prompt);
        std::io::stdout().flush()?;

        let gen_cfg = GenConfig::default();
        let mut cache = model.new_cache()?;
        let metrics = generate(&mut model, &mut cache, &cli.prompt, &gen_cfg, &mut |text| {
            print!("{text}");
            std::io::stdout().flush().ok();
        })?;

        // ── benchmark printout ──────────────────────────────────────────────
        println!();
        println!("──── metrics ──────────────────────────────");
        println!("  TTFT        {:8.1?}", metrics.ttft);
        if metrics.decode_tokens > 0 {
            let decode_time = metrics.total_time - metrics.ttft;
            let tok_s = metrics.decode_tokens as f64 / decode_time.as_secs_f64();
            println!(
                "  decode      {:5} tokens in {:8.1?}  ({:.1} tok/s)",
                metrics.decode_tokens, decode_time, tok_s
            );
        }
        println!("  total wall  {:8.1?}", metrics.total_time);
        println!("─────────────────────────────────────────────");
    }

    Ok(())
}
