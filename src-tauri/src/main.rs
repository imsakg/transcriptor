// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_core as candle;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer::{
    Config as MixFormerConfig, MixFormerSequentialForCausalLM as MixFormer,
};
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use candle_transformers::models::whisper::{self as m, audio, Config};
use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::executor::block_on;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{distributions::Distribution, SeedableRng};
use std::iter;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;
use tracing::info;
use tracing_subscriber;

mod llm;
mod multilingual;

struct AsyncProcInputTx {
    inner: TokioMutex<mpsc::Sender<String>>,
}

struct AsyncWhistperInputTx {
    inner: TokioMutex<mpsc::Sender<String>>,
}

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    tx_pipe: AsyncWhistperInputTx,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
        tx_pipe: AsyncWhistperInputTx,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            tx_pipe,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    async fn run(&mut self, mel: &Tensor, times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                match times {
                    Some((start, end)) => {
                        println!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text);
                        self.tx_pipe
                            .inner
                            .lock()
                            .await
                            .send(format!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text))
                            .await
                            .unwrap();
                    }
                    None => {
                        println!(
                            "{:.1}s -- {:.1}s: {}",
                            segment.start,
                            segment.start + segment.duration,
                            segment.dr.text,
                        )
                    }
                }
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }

    fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    #[allow(dead_code)]
    fn reset_kv_cache(&mut self) {
        match &mut self.model {
            Model::Normal(m) => m.reset_kv_cache(),
            Model::Quantized(m) => m.reset_kv_cache(),
        }
    }

    fn model(&mut self) -> &mut Model {
        &mut self.model
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
    #[value(name = "distil-large-v3")]
    DistilLargeV3,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::DistilLargeV2
            | Self::DistilLargeV3 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::DistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
        }
    }
}

#[derive(Clone, Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "distil-medium.en")]
    model: WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
}

pub fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;
    tracing_subscriber::fmt::init();

    let (async_proc_input_tx, async_proc_input_rx) = mpsc::channel(1);
    let (async_proc_output_tx, mut async_proc_output_rx) = mpsc::channel(1);
    let (in_whisper_mpsc_tx, mut in_whisper_mpsc_rx) = mpsc::channel(1);
    let (out_whisper_mpsc_tx, mut out_whisper_mpsc_rx) = mpsc::channel(1);
    let tx_pipe = AsyncWhistperInputTx {
        inner: TokioMutex::new(in_whisper_mpsc_tx),
    };
    let out_tx_pipe = AsyncWhistperInputTx {
        inner: TokioMutex::new(out_whisper_mpsc_tx),
    };

    let args = Args::parse();
    let out_args = args.clone();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let in_device = candle_examples::device(args.cpu)?;
    let out_device = candle_examples::device(args.cpu)?;
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();

    let out_default_model = default_model.to_string();
    let out_default_revision = default_revision.to_string();

    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let (out_model_id, out_revision) = match (out_args.model_id, out_args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (out_default_model, revision),
        (None, None) => (out_default_model, out_default_revision),
    };

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };

    let (out_config_filename, out_tokenizer_filename, out_weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            out_model_id,
            RepoType::Model,
            out_revision,
        ));
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };

    // Input Definations
    let in_config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let in_tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let in_model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_filename,
            &in_device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, in_config.clone())?)
    } else {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &in_device)?
        };
        Model::Normal(m::model::Whisper::load(&vb, in_config.clone())?)
    };
    let in_language_token = None;
    let mut in_dc = Decoder::new(
        in_model,
        in_tokenizer.clone(),
        args.seed,
        &in_device,
        in_language_token,
        args.task,
        args.timestamps,
        args.verbose,
        tx_pipe,
    )?;
    let in_mel_bytes = match in_config.num_mel_bins {
        80 => include_bytes!("whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("whisper/melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };

    let mut in_mel_filters = vec![0f32; in_mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        in_mel_bytes,
        &mut in_mel_filters,
    );

    // Output Definations

    let out_config: Config = serde_json::from_str(&std::fs::read_to_string(out_config_filename)?)?;
    let out_tokenizer = Tokenizer::from_file(out_tokenizer_filename).map_err(E::msg)?;
    let out_model = if out_args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &out_weights_filename,
            &out_device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, out_config.clone())?)
    } else {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[out_weights_filename], m::DTYPE, &out_device)?
        };
        Model::Normal(m::model::Whisper::load(&vb, out_config.clone())?)
    };
    let out_language_token = None;
    let mut out_dc = Decoder::new(
        out_model,
        out_tokenizer.clone(),
        args.seed,
        &out_device,
        out_language_token,
        args.task,
        args.timestamps,
        args.verbose,
        out_tx_pipe,
    )?;
    let out_mel_bytes = match out_config.num_mel_bins {
        80 => include_bytes!("whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("whisper/melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };

    let mut out_mel_filters = vec![0f32; out_mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        out_mel_bytes,
        &mut out_mel_filters,
    );

    // Set up the input device and stream with the default input config.
    let host = cpal::default_host();

    // Input Device
    let _in_device = "default";

    if args.verbose {
        println!("{:?}", host.default_input_device().unwrap().name());
    }

    let _in_device = if _in_device == "default" {
        host.default_input_device()
    } else {
        host.input_devices()?
            .find(|x| x.name().map(|y| y == _in_device).unwrap_or(false))
    }
    .expect("failed to find input device");
    let _in_config = _in_device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("in device is: {:?}", _in_device.name());

    let in_channel_count = _in_config.channels() as usize;

    let audio_in_ring_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_in_ring_buffer_2 = audio_in_ring_buffer.clone();

    std::thread::spawn(move || loop {
        let data = record_audio(&_in_device, &_in_config, 300).unwrap();
        audio_in_ring_buffer
            .lock()
            .unwrap()
            .extend_from_slice(&data);
        let max_len = data.len() * 16;
        let data_len = data.len();
        let len = audio_in_ring_buffer.lock().unwrap().len();
        if len > max_len {
            let mut data = audio_in_ring_buffer.lock().unwrap();
            let new_data = data[data_len..].to_vec();
            *data = new_data;
        }
    });

    // Output device

    // let _out_device = "Loopback Audio";
    //
    //  let _out_device=   host.default_input_device()
    // let _out_device = if _out_device == "default" {
    //     host.default_input_device()
    // } else {
    //     host.input_devices()?.find(|x| {
    //         x.name()
    //             .map(|y| {
    //                 println!("{:?}", y);
    //                 y == _out_device
    //             })
    //             .unwrap_or(false)
    //     })
    // }
    // .expect("failed to find output device");
    // let _out_config = _out_device
    //     .default_output_config()
    //     .expect("Failed to get default output config");
    // println!("out device is: {:?}", _out_device.name());
    //
    // let out_channel_count = _out_config.channels() as usize;
    //
    // let audio_out_ring_buffer = Arc::new(Mutex::new(Vec::new()));
    // let audio_out_ring_buffer_2 = audio_out_ring_buffer.clone();
    //
    // std::thread::spawn(move || loop {
    //     let data = record_audio(&_out_device, &_out_config, 300).unwrap();
    //     audio_out_ring_buffer
    //         .lock()
    //         .unwrap()
    //         .extend_from_slice(&data);
    //     let max_len = data.len() * 16;
    //     let data_len = data.len();
    //     let len = audio_out_ring_buffer.lock().unwrap().len();
    //     if len > max_len {
    //         let mut data = audio_out_ring_buffer.lock().unwrap();
    //         let new_data = data[data_len..].to_vec();
    //         *data = new_data;
    //     }
    // });
    //
    // // Input Whisper
    // std::thread::spawn(move || {
    //     // loop to process the audio data forever (until the user stops the program)
    //     println!("input loop started {:?}", in_config);
    //
    //     for (i, _) in iter::repeat(()).enumerate() {
    //         std::thread::sleep(std::time::Duration::from_millis(1000));
    //         let data = audio_in_ring_buffer_2.lock().unwrap().clone();
    //         let pcm_data: Vec<_> = data[..data.len() / in_channel_count as usize]
    //             .iter()
    //             .map(|v| *v as f32 / 32768.)
    //             .collect();
    //         let mel = audio::pcm_to_mel(&in_config, &pcm_data, &in_mel_filters);
    //         let mel_len = mel.len();
    //         let mel = Tensor::from_vec(
    //             mel,
    //             (1, in_config.num_mel_bins, mel_len / in_config.num_mel_bins),
    //             &in_device,
    //         )?;
    //
    //         // on the first iteration, we detect the language and set the language token.
    //         if i == 0 {
    //             println!("is_multilingual {}", args.model.is_multilingual());
    //             let language_token = match (args.model.is_multilingual(), &args.language) // args.language
    //              {
    //                 (true, None) => Some(multilingual::detect_language(
    //                     in_dc.model(),
    //                     &in_tokenizer,
    //                     &mel,
    //                 )?),
    //                 (false, None) => None,
    //                 (true, Some(language)) => {
    //                     println!("language setting to: {:?}", language);
    //                     match token_id(&in_tokenizer, &format!("<|{language}|>")) {
    //                         Ok(token_id) => Some(token_id),
    //                         Err(_) => anyhow::bail!("language {language} is not supported"),
    //                     }
    //                 }
    //                 (false, Some(_)) => {
    //                     anyhow::bail!("a language cannot be set for non-multilingual models")
    //                 }
    //             };
    //             println!("language_token: {:?}", language_token);
    //             in_dc.set_language_token(language_token);
    //         }
    //         block_on(in_dc.run(
    //             &mel,
    //             Some((
    //                 i as f64,
    //                 i as f64 + data.len() as f64 / m::SAMPLE_RATE as f64,
    //             )),
    //         ))?;
    //         in_dc.reset_kv_cache();
    //     }
    //
    //     Ok(())
    // });
    //
    // // Output Whisper
    // let out_config = out_config.clone();
    // let out_tokenizer = out_tokenizer.clone();
    //
    // std::thread::spawn(move || {
    //     // loop to process the audio data forever (until the user stops the program)
    //     println!("output loop started {:?}", out_config);
    //     for (i, _) in iter::repeat(()).enumerate() {
    //         std::thread::sleep(std::time::Duration::from_millis(1000));
    //         let data = audio_out_ring_buffer_2.lock().unwrap().clone();
    //         let pcm_data: Vec<_> = data[..data.len() / out_channel_count as usize]
    //             .iter()
    //             .map(|v| *v as f32 / 32768.)
    //             .collect();
    //         let mel = audio::pcm_to_mel(&out_config, &pcm_data, &out_mel_filters);
    //         let mel_len = mel.len();
    //         let mel = Tensor::from_vec(
    //             mel,
    //             (
    //                 1,
    //                 out_config.num_mel_bins,
    //                 mel_len / out_config.num_mel_bins,
    //             ),
    //             &out_device,
    //         )?;
    //
    //         // on the first iteration, we detect the language and set the language token.
    //         if i == 0 {
    //             println!("is_multilingual {}", out_args.model.is_multilingual());
    //             let language_token = match (out_args.model.is_multilingual(), None::<String>) // args.language
    //             {
    //                 (true, None) => Some(multilingual::detect_language(
    //                     out_dc.model(),
    //                     &out_tokenizer,
    //                     &mel,
    //                 )?),
    //                 (false, None) => None,
    //                 (true, Some(language)) => {
    //                     match token_id(&out_tokenizer, &format!("<|{language}|>")) {
    //                         Ok(token_id) => Some(token_id),
    //                         Err(_) => anyhow::bail!("language {language} is not supported"),
    //                     }
    //                 }
    //                 (false, Some(_)) => {
    //                     anyhow::bail!("a language cannot be set for non-multilingual models")
    //                 }
    //             };
    //             println!("language_token: {:?}", language_token);
    //             out_dc.set_language_token(language_token);
    //         }
    //         block_on(out_dc.run(
    //             &mel,
    //             Some((
    //                 i as f64,
    //                 i as f64 + data.len() as f64 / m::SAMPLE_RATE as f64,
    //             )),
    //         ))?;
    //         out_dc.reset_kv_cache();
    //     }
    //     Ok(())
    // });
    //
    std::thread::spawn(|| {
        let mut llm_pipeline = llm::main().unwrap();
        llm_pipeline
            .run(
                r##"
            Person 1: How was your day?
            Person 2: It was fine tho? How about you?
            Answer:"##,
                5000,
            )
            .unwrap();

        println!("llm pipeline done");
    });

    // app_lib::run();

    tauri::Builder::default()
        .manage(AsyncProcInputTx {
            inner: TokioMutex::new(async_proc_input_tx),
        })
        // .manage(AsyncWhistperInputTx {
        //     inner: TokioMutex::new(whisper_mpsc_tx),
        // })
        .invoke_handler(tauri::generate_handler![js2rs])
        .setup(move |app| {
            tauri::async_runtime::spawn(async move {
                async_process_model(async_proc_input_rx, async_proc_output_tx).await
            });
            let app_handle = app.handle().clone();
            let whisper_in_app_handle = app.handle().clone();
            let whisper_out_app_handle = app.handle().clone();

            tauri::async_runtime::spawn(async move {
                while let Some(input) = in_whisper_mpsc_rx.recv().await {
                    whisper_in_app_handle
                        .emit("whisper_in", input.to_string())
                        .unwrap();
                }
            });

            tauri::async_runtime::spawn(async move {
                while let Some(input) = out_whisper_mpsc_rx.recv().await {
                    whisper_out_app_handle
                        .emit("whisper_out", input.to_string())
                        .unwrap();
                }
            });

            tauri::async_runtime::spawn(async move {
                loop {
                    if let Some(output) = async_proc_output_rx.recv().await {
                        rs2js(output, &app_handle);
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    Ok(())
}

fn record_audio(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    milliseconds: u64,
) -> Result<Vec<i16>> {
    let writer = Arc::new(Mutex::new(Vec::new()));
    let writer_2 = writer.clone();
    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let processed = data
                .iter()
                .map(|v| (v * 32768.0) as i16)
                .collect::<Vec<i16>>();
            writer_2.lock().unwrap().extend_from_slice(&processed);
        },
        move |err| {
            eprintln!("an error occurred on stream: {}", err);
        },
        None,
    )?;
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(milliseconds));
    drop(stream);
    let data = writer.lock().unwrap().clone();
    let step = 3;
    let data: Vec<i16> = data.iter().step_by(step).copied().collect();
    Ok(data)
}

fn rs2js<R: tauri::Runtime>(message: String, manager: &AppHandle<R>) {
    info!(?message, "rs2js");
    manager.emit("rs2js", format!("rs: {}", message)).unwrap();
}

#[tauri::command]
async fn js2rs(message: String, state: tauri::State<'_, AsyncProcInputTx>) -> Result<(), String> {
    info!(?message, "js2rs");
    let async_proc_input_tx = state.inner.lock().await;
    async_proc_input_tx
        .send(message)
        .await
        .map_err(|e| e.to_string())
}

async fn async_process_model(
    mut input_rx: mpsc::Receiver<String>,
    output_tx: mpsc::Sender<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    while let Some(input) = input_rx.recv().await {
        let output = input;
        output_tx.send(output).await?;
    }

    Ok(())
}
