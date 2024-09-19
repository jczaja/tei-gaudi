//use ndarray::{s, Axis};
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use std::ops::{Div, Mul};
use std::path::PathBuf;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

use tch::{nn, nn::Module, nn::OptimizerConfig, nn::VarStore};//, Device, Hpu}; // TODO: make HPU after CPU

/// This enum is needed to be able to differentiate between jina models that also use
/// the `bert` model type and valid Bert models.
/// We use the `_name_or_path` field in the config to do so. This might not be robust in the long
/// run but is still better than the other options...
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "_name_or_path")]
pub enum BertConfigWrapper {
    #[serde(rename = "jinaai/jina-bert-implementation")]
    JinaBert(BertConfig),
    #[serde(rename = "jinaai/jina-bert-v2-qk-post-norm")]
    JinaCodeBert(BertConfig),
    #[serde(untagged)]
    Bert(BertConfig),
}

#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
enum Config {
    Bert(BertConfigWrapper),
    XlmRoberta(BertConfig),
    Camembert(BertConfig),
    Roberta(BertConfig),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
    #[serde(rename(deserialize = "nomic_bert"))]
    NomicBert(NomicConfig),
    Mistral(MistralConfig),
    #[serde(rename = "new")]
    Gte(GTEConfig),
    Qwen2(Qwen2Config),
}

// TODO: make for CPU


//use tokio::runtime::Runtime; // Add tokio

// What is pooling mode and is it supported?
// what is backend thread?

pub struct TchBackend {
}

impl TchBackend {
    pub fn new(
        model_path: PathBuf,
        dtype: String,
        model_type: ModelType,
    ) -> Result<Self, BackendError> {

        let default_safetensors = model_path.join("model.safetensors");
        let default_pytorch = model_path.join("pytorch_model.bin");

        // Single Files
        let model_files = if default_safetensors.exists() {
            vec![default_safetensors]
        } else if default_pytorch.exists() {
            vec![default_pytorch]
        }
        // Sharded weights
        else {
            // Get index file
            let index_file = model_path.join("model.safetensors.index.json");

            // Parse file
            let index_file_string: String = std::fs::read_to_string(&index_file)
                .map_err(|err| BackendError::Start(err.to_string()))?;
            let json: serde_json::Value = serde_json::from_str(&index_file_string)
                .map_err(|err| BackendError::Start(err.to_string()))?;

            let weight_map = match json.get("weight_map") {
                None => {
                    return Err(BackendError::Start(format!(
                        "no weight map in {index_file:?}"
                    )));
                }
                Some(serde_json::Value::Object(map)) => map,
                Some(_) => {
                    return Err(BackendError::Start(format!(
                        "weight map in {index_file:?} is not a map"
                    )));
                }
            };
            let mut safetensors_files = std::collections::HashSet::new();
            for value in weight_map.values() {
                if let Some(file) = value.as_str() {
                    safetensors_files.insert(file.to_string());
                }
            }

            // Collect paths
            safetensors_files
                .iter()
                .map(|n| model_path.join(n))
                .collect()
        };

        // Load config
        let config: String = std::fs::read_to_string(model_path.join("config.json"))
            .context("Unable to read config file")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;
        let config: Config = serde_json::from_str(&config)
            .context("Model is not supported")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;


        // Set Device (CPU then HPU)
        // Set data type (f32)

        // Make model files into model
        // TODO: How to make safetensors lloaded
        //
        //

        let vs = VarStore::new(Device::CPU);
        // TODO: Load model desc
        // TODO: miniGPT tutorial
        //vs.load(model_files)?; // TODO: multiple files
        vs.load("MODEL//models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb
536a557fa166f842b0e09/model.safetensor")?;

        Ok(Self {
        })
    }
}

impl Backend for TchBackend {

    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let pooled_indices = batch.pooled_indices.clone();
        let raw_indices = batch.raw_indices.clone();
        let batch_size = batch.len();

        // Used for indexing in the raw_embeddings tensor
        let input_lengths: Vec<usize> = (0..batch.len())
            .map(|i| {
                (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize
            })
            .collect();

        // Run forward
        let (pooled_embeddings, raw_embeddings) = self.model.embed(batch).e()?;

        // Get from device to CPU

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        Ok(embeddings)
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {

        let batch_size = batch.len();
        let mut predictions =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        Ok(predictions)
    }
}
/*
pub trait WrapErr<O> {
    fn s(self) -> Result<O, BackendError>;
    fn e(self) -> Result<O, BackendError>;
}

impl<O> WrapErr<O> for Result<O, ort::Error> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}

impl<O> WrapErr<O> for Result<O, ndarray::ShapeError> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}
*/
