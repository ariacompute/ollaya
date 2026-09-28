//! ONNX Runtime engine for encoder decision models (`laya-markers-v1`, `afm-de-latest`).
//!
//! A model directory holds the layers of one manifest:
//!
//! ```text
//! model.onnx [+ model.onnx.data]   encoder + decision head + marker gather, one graph
//! tokenizer.json                   HF tokenizers file
//! decision.json                    layout, lengths, special tokens, tensor names
//! calibration.json                 temperatures
//! arch.json                        (optional) the network for the MLX engine; see crate::mlx
//! ```

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ollaya_decision::{
    AFM_DE_LATEST, Calibration, CalibrationFile, LAYA_MARKERS_V1, LayaLayout, Questions,
    SpecialTokens, StateTruncation, TokenEncoder,
};
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use serde::Deserialize;
use serde_json::Value;

use crate::net::{Batch, Head, Net};
use crate::{Error, Output, QuestionOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device ordinal.
    Cuda(i32),
    /// The Apple GPU, through MLX (the `mlx` feature), not ONNX Runtime.
    Metal,
}

/// Embedding shortlist for high-cardinality Choice (`afm-de-latest`).
#[derive(Debug, Clone, Deserialize)]
pub struct ShortlistConfig {
    #[serde(default = "shortlist_threshold")]
    pub threshold: usize,
    #[serde(default = "shortlist_k")]
    pub k: usize,
}

fn shortlist_threshold() -> usize {
    40
}
fn shortlist_k() -> usize {
    20
}

/// The `decision` layer.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    pub engine: String,
    pub layout: String,
    pub max_len: usize,
    pub head_max_len: usize,
    pub special_tokens: SpecialTokens,
    /// The graph takes at least this many marker slots (extra slots are masked).
    #[serde(default = "one")]
    pub min_markers: usize,
    /// Override Laya's 48 / AFM-D's 96 when set.
    #[serde(default)]
    pub option_desc_max: Option<usize>,
    #[serde(default)]
    pub state_truncation: Option<StateTruncation>,
    #[serde(default)]
    pub shortlist: Option<ShortlistConfig>,
}

fn one() -> usize {
    1
}

/// Encoder input for one request: every question against the shared state.
#[derive(Debug, Clone)]
pub struct Encoding {
    pub questions: Vec<ollaya_decision::Encoded>,
    /// Tokens in the serialized state, before any truncation.
    pub state_tokens: usize,
}

pub struct OnnxModel {
    net: Net,
    tokenizer: Tokenizer,
    layout: LayaLayout,
    min_markers: usize,
    shortlist: Option<ShortlistConfig>,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

/// An ONNX Runtime session on `device`. Every engine builds its sessions here, so all of them
/// get the same execution-provider settings.
pub fn session(
    graph: &Path,
    device: Device,
    intra_threads: Option<usize>,
) -> Result<Session, Error> {
    session_with(graph, device, intra_threads, Ok)
}

/// [`session`], with `configure` applied to the builder just before the graph loads: session
/// options that one family's graphs need.
pub fn session_with(
    graph: &Path,
    device: Device,
    intra_threads: Option<usize>,
    configure: impl FnOnce(SessionBuilder) -> Result<SessionBuilder, Error>,
) -> Result<Session, Error> {
    session_for(graph, device, intra_threads, CudaArena::Default, configure)
}

/// How the CUDA provider's memory arena grows when a run needs more than it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArena {
    /// ONNX Runtime's default: each extension is twice the previous one. Few allocations, but
    /// up to twice the memory that is live.
    Default,
    /// Each extension is exactly the request. The decoder graphs allocate large buffers of many
    /// sizes on every run (attention scores grow with the square of the row length; with
    /// `weights_in_memory: bf16`, one layer's widened weights), and doubling left decider-4b's
    /// arena bigger than a 24 GB GPU: the NVIDIA driver on Windows (WSL) then spilled 4 GB into
    /// system memory and a parity run slowed down more than tenfold. Exact growth held decider-2b
    /// at 8.4 GB instead of 10.2 GB at the same speed
    /// (`docs/decisions/0002-decoder-weights-in-memory.md`).
    SameAsRequested,
}

/// [`session_with`], with the CUDA arena growing as `arena` says.
pub fn session_for(
    graph: &Path,
    device: Device,
    intra_threads: Option<usize>,
    arena: CudaArena,
    configure: impl FnOnce(SessionBuilder) -> Result<SessionBuilder, Error>,
) -> Result<Session, Error> {
    if device == Device::Metal {
        return Err(Error::Model(
            "the Metal device runs on MLX, not ONNX Runtime".into(),
        ));
    }
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    if let Some(n) = intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    if let Device::Cuda(id) = device {
        builder = with_cuda(builder, id, arena)?;
    }
    Ok(configure(builder)?.commit_from_file(graph)?)
}

#[cfg(feature = "cuda")]
fn with_cuda(
    builder: ort::session::builder::SessionBuilder,
    device_id: i32,
    arena: CudaArena,
) -> Result<ort::session::builder::SessionBuilder, Error> {
    // TF32 matmuls keep 10 mantissa bits, which moves calibrated probabilities by ~1e-3 and
    // flips close decisions. fp32 graphs run in true fp32; speed comes from fp16 graphs.
    let mut ep = ort::ep::CUDA::default()
        .with_device_id(device_id)
        .with_tf32(false);
    if arena == CudaArena::SameAsRequested {
        ep = ep.with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
    }
    Ok(builder.with_execution_providers([ep.build().error_on_failure()])?)
}

#[cfg(not(feature = "cuda"))]
fn with_cuda(
    _builder: ort::session::builder::SessionBuilder,
    _device_id: i32,
    _arena: CudaArena,
) -> Result<ort::session::builder::SessionBuilder, Error> {
    Err(Error::Model(
        "this build of ollaya has no CUDA support".into(),
    ))
}

/// A model's `tokenizer.json`, with any truncation or padding it ships switched off: layouts
/// place every token themselves, and Python's tokenizer calls (the references) ignore those
/// settings too. Some upstream files bake in truncation (e.g. 512) that would silently cut states.
pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer, Error> {
    let mut tok = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    tok.with_truncation(None)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    tok.with_padding(None);
    Ok(tok)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

/// The files one model loads from. In the blob store they are `sha256-<hex>` blobs; in a
/// development export they are the named files of one directory.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub graph: PathBuf,
    /// A vision model's image graph (`vision.onnx`).
    pub vision: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub decision: PathBuf,
    pub calibration: Option<PathBuf>,
    /// The `arch` layer: what the MLX engine builds its network from.
    pub arch: Option<PathBuf>,
    /// The author's weights file, for MLX. Unset: the file the arch layer names, next to it.
    pub weights: Option<PathBuf>,
}

impl ModelFiles {
    /// `model.onnx`, `tokenizer.json`, `decision.json`, `calibration.json` and, when present,
    /// `arch.json` and `vision.onnx` in one directory.
    pub fn dir(dir: &Path) -> Self {
        let arch = dir.join("arch.json");
        let vision = dir.join("vision.onnx");
        ModelFiles {
            graph: dir.join("model.onnx"),
            vision: vision.is_file().then_some(vision),
            tokenizer: dir.join("tokenizer.json"),
            decision: dir.join("decision.json"),
            calibration: Some(dir.join("calibration.json")),
            arch: arch.is_file().then_some(arch),
            weights: None,
        }
    }
}

impl OnnxModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx"
            || (config.layout != LAYA_MARKERS_V1 && config.layout != AFM_DE_LATEST)
        {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this runner serves onnx/{LAYA_MARKERS_V1} and onnx/{AFM_DE_LATEST}",
                config.engine, config.layout
            )));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let net = crate::net::load(files, device, intra_threads, Head::Laya)?;

        let mut layout = if config.layout == AFM_DE_LATEST {
            LayaLayout::afm_de(config.max_len, config.head_max_len, config.special_tokens)
        } else {
            LayaLayout::laya(config.max_len, config.head_max_len, config.special_tokens)
        };
        if let Some(n) = config.option_desc_max {
            layout.option_desc_max = n;
        }
        if let Some(t) = config.state_truncation {
            layout.state_truncation = t;
        }

        Ok(OnnxModel {
            net,
            tokenizer: Tokenizer(tokenizer),
            layout,
            min_markers: config.min_markers,
            shortlist: config.shortlist,
            calibration,
            device,
        })
    }

    /// Encode every question against the shared state (token ids and marker positions).
    pub fn encode(&self, state: &Value, questions: &Questions) -> Result<Encoding, Error> {
        let state_ids = self
            .layout
            .encode_state(&self.tokenizer, &ollaya_decision::serialize_state(state))?;
        let questions = questions
            .iter()
            .map(|(qid, q)| {
                self.layout
                    .encode(&self.tokenizer, &state_ids, q)
                    .map_err(|e| Error::Decision(e.for_question(qid)))
            })
            .collect::<Result<_, _>>()?;
        Ok(Encoding {
            questions,
            state_tokens: state_ids.len(),
        })
    }

    /// Answer every question in one forward pass.
    ///
    /// For `afm-de-latest` with a shortlist config, Choice questions at or above the threshold
    /// are narrowed by bag-of-token cosine (stand-in until an embed.onnx layer ships) before
    /// packing; logits are scattered back to the full option axis and confidence should be
    /// discounted by the caller via [`ollaya_decision::shortlist::confidence_discount`].
    pub fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        if let Some(sl) = &self.shortlist {
            return self.run_with_shortlist(state, questions, sl);
        }
        let encoded = self.encode(state, questions)?;
        self.run_encoded(&encoded, questions)
    }

    fn run_with_shortlist(
        &self,
        state: &Value,
        questions: &Questions,
        sl: &ShortlistConfig,
    ) -> Result<Output, Error> {
        use ollaya_decision::question::{Criteria, QType};
        use ollaya_decision::shortlist::{self, bag_embed};

        let state_text = ollaya_decision::serialize_state(state);
        let state_ids = self.layout.encode_state(&self.tokenizer, &state_text)?;
        let mut narrowed = Questions::new();
        let mut maps: Vec<Option<(Vec<usize>, usize)>> = Vec::new();

        for (qid, q) in questions {
            if q.qtype == QType::Choice {
                let opts = q.render_options();
                if shortlist::needs_shortlist(opts.len(), sl.threshold) {
                    let mut vectors = vec![bag_embed(&state_ids, 256)];
                    for opt in &opts {
                        let ids = self.tokenizer.encode(opt).map_err(Error::Decision)?;
                        vectors.push(bag_embed(&ids, 256));
                    }
                    let keep = shortlist::shortlist_indices(&vectors, sl.k, &[])
                        .map_err(Error::Decision)?;
                    let k_full = opts.len();
                    let Criteria::Choice(ref crit) = q.criteria else {
                        return Err(Error::Model("choice without Choice criteria".into()));
                    };
                    let pairs: Vec<_> = crit.iter().collect();
                    let mut sub = crit.clone();
                    sub.clear();
                    for &i in &keep {
                        let (k, v) = pairs.get(i).ok_or_else(|| {
                            Error::Model(format!("shortlist index {i} out of range"))
                        })?;
                        sub.insert((*k).clone(), (*v).clone());
                    }
                    let mut nq = q.clone();
                    nq.criteria = Criteria::Choice(sub);
                    narrowed.insert(qid.clone(), nq);
                    maps.push(Some((keep, k_full)));
                    continue;
                }
            }
            narrowed.insert(qid.clone(), q.clone());
            maps.push(None);
        }

        let encoded = self.encode(state, &narrowed)?;
        let mut out = self.run_encoded(&encoded, &narrowed)?;
        for (i, map) in maps.into_iter().enumerate() {
            if let Some((keep, k_full)) = map {
                let slim = std::mem::take(&mut out.questions[i].logits);
                let mut full = vec![f32::NEG_INFINITY; k_full];
                for (j, &src) in keep.iter().enumerate() {
                    if j < slim.len() {
                        full[src] = slim[j];
                    }
                }
                out.questions[i].logits = full;
                let _ = shortlist::confidence_discount(keep.len(), k_full);
            }
        }
        Ok(out)
    }

    pub fn run_encoded(&self, encoding: &Encoding, questions: &Questions) -> Result<Output, Error> {
        let encoded = &encoding.questions;
        let qtypes: Vec<i64> = questions.values().map(|q| q.qtype.index() as i64).collect();
        let lens: Vec<usize> = encoded.iter().map(|e| e.ids.len()).collect();
        let mut outputs = Vec::with_capacity(encoded.len());
        for range in crate::engine::batches(&lens, crate::engine::TOKEN_BUDGET, usize::MAX) {
            outputs.extend(self.run_batch(&encoded[range.clone()], &qtypes[range])?);
        }
        Ok(Output {
            questions: outputs,
            input_tokens: lens.iter().sum(),
            state_tokens: encoding.state_tokens,
            state_truncated: encoded.iter().any(|e| e.state_truncated),
        })
    }

    /// One forward pass over `encoded` (one row per question), padded to its longest row.
    fn run_batch(
        &self,
        encoded: &[ollaya_decision::Encoded],
        qtypes: &[i64],
    ) -> Result<Vec<QuestionOutput>, Error> {
        let n = encoded.len();
        let seq = encoded.iter().map(|e| e.ids.len()).max().unwrap_or(0);
        let k = encoded
            .iter()
            .map(|e| e.markers.len())
            .max()
            .unwrap_or(0)
            .max(self.min_markers);

        let pad = i64::from(self.layout.special.pad);
        let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
        let mut attention = Array2::<i64>::zeros((n, seq));
        let mut marker_pos = Array2::<i64>::zeros((n, k));
        let mut marker_mask = Array2::<bool>::from_elem((n, k), false);
        for (r, e) in encoded.iter().enumerate() {
            for (c, &id) in e.ids.iter().enumerate() {
                input_ids[[r, c]] = i64::from(id);
                attention[[r, c]] = 1;
            }
            for (c, &m) in e.markers.iter().enumerate() {
                marker_pos[[r, c]] = m as i64;
                marker_mask[[r, c]] = true;
            }
        }

        let [logits, act] = <[_; 2]>::try_from(self.net.run(
            Batch {
                input_ids,
                attention,
                markers: Some((marker_pos, marker_mask)),
                qtype: Some(qtypes.to_vec()),
            },
            &["logits", "act_logits"],
        )?)
        .map_err(|_| Error::Model("expected two outputs".into()))?;
        Ok(encoded
            .iter()
            .enumerate()
            .map(|(r, e)| QuestionOutput {
                logits: (0..e.markers.len()).map(|c| logits[[r, c]]).collect(),
                act_logits: Some(act.slice(ndarray::s![r, ..]).to_vec()),
            })
            .collect())
    }
}
