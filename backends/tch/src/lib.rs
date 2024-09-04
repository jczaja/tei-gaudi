//use ndarray::{s, Axis};
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use std::ops::{Div, Mul};
use std::path::PathBuf;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

use tch::{nn, nn::Module, nn::OptimizerConfig};//, Device, Hpu}; // TODO: make HPU after CPU


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

        match model_type {
            ModelType::Classifier => {}
            ModelType::Embedding(pool) => {
                if pool != Pool::Cls {
                    return Err(BackendError::Start(format!("{pool:?} is not supported")));
                }
            }
        };


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
        if !batch.raw_indices.is_empty() {
            return Err(BackendError::Inference(
                "raw embeddings are not supported for the TCH backend.".to_string(),
            ));
        }
        let batch_size = batch.len();
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
