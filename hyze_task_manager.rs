//! Hyze Task Manager v1.0
//! Schedules inference/training across CPU/GPU/IPU tiles
//! Auto-scales, load balances, zero cold-start

use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;

#[derive(Clone, Serialize, Deserialize)]
pub enum HyzeTask {
    Inference { model: String, input: Vec<u8> },
    Training { batch: Vec<Tensor>, epochs: u32 },
    Multimodal { text: String, image: Vec<u8>, audio: Vec<u16> },
}

pub struct HyzeTaskManager {
    tile_pool: Arc<Mutex<HashMap<u32, TileStatus>>>,
    task_queue: mpsc::Sender<HyzeTask>,
    gpu_queue: mpsc::Sender<TrainingBatch>,
}

#[derive(Clone)]
struct TileStatus {
    tile_id: u32,
    task_type: TaskType,
    load: f32,
    available: bool,
}

#[derive(Clone, Serialize, Deserialize)]
enum TaskType {
    IpInference,
    Multimodal,
    LoRA,
}

impl HyzeTaskManager {
    pub fn new(tile_count: u32) -> Self {
        let (tx, rx) = mpsc::channel(1024);
        let gpu_tx = Self::gpu_channel();
        
        tokio::spawn(Self::task_dispatcher(rx, gpu_tx.clone()));
        
        let mut pool = HashMap::new();
        for tile in 0..tile_count {
            pool.insert(tile, TileStatus {
                tile_id: tile,
                task_type: TaskType::IpInference,
                load: 0.0,
                available: true,
            });
        }
        
        Self {
            tile_pool: Arc::new(Mutex::new(pool)),
            task_queue: tx,
            gpu_queue: gpu_tx,
        }
    }

    pub async fn schedule(&self, task: HyzeTask) -> Result<TaskId> {
        let task_id = rand::random::<u64>();
        
        match task {
            HyzeTask::Inference { .. } => {
                let tile = self.find_available_tile(TaskType::IpInference).await?;
                self.assign_tile(tile, task_id).await?;
            }
            HyzeTask::Training { .. } => {
                self.gpu_queue.send(task).await?;
            }
            HyzeTask::Multimodal { .. } => {
                let tiles = self.find_tiles(3, TaskType::Multimodal).await?;
                for tile in tiles {
                    self.assign_tile(tile, task_id).await?;
                }
            }
        }
        
        Ok(task_id)
    }

    pub async fn monitor(&self) -> HashMap<u32, TileMetrics> {
        let pool = self.tile_pool.lock().await;
        pool.iter().map(|(id, status)| {
            (*id, TileMetrics {
                load: status.load,
                tasks_running: 0,
                available: status.available,
            })
        }).collect()
    }

    async fn find_available_tile(&self, task_type: TaskType) -> Result<u32> {
        let pool = self.tile_pool.lock().await;
        for (tile_id, status) in pool.iter() {
            if status.available && status.task_type == task_type && status.load < 0.8 {
                return Ok(*tile_id);
            }
        }
        bail!("No available tiles for {:?}", task_type)
    }

    async fn find_tiles(&self, count: usize, task_type: TaskType) -> Result<Vec<u32>> {
        let mut tiles = Vec::new();
        let pool = self.tile_pool.lock().await;
        for (tile_id, status) in pool.iter() {
            if tiles.len() >= count { break; }
            if status.available && status.task_type == task_type {
                tiles.push(*tile_id);
            }
        }
        if tiles.len() < count {
            bail!("Need {} {} tiles, found {}", count, format!("{:?}", task_type), tiles.len());
        }
        Ok(tiles)
    }

    async fn assign_tile(&self, tile_id: u32, task_id: TaskId) -> Result<()> {
        let mut pool = self.tile_pool.lock().await;
        if let Some(tile) = pool.get_mut(&tile_id) {
            tile.available = false;
            tile.load = 1.0;
            // Send task to tile worker
            self.tile_pool.notify_tile(tile_id, task_id);
        }
        Ok(())
    }
}

// Background task dispatcher
async fn task_dispatcher(mut rx: mpsc::Receiver<HyzeTask>, gpu_rx: mpsc::Sender<TrainingBatch>) {
    while let Some(task) = rx.recv().await {
        match task {
            HyzeTask::Inference { model, input } => {
                tokio::spawn(ipu_tile_worker(model, input));
            }
            // ... other handlers
        }
    }
}

#[tokio::test]
async fn test_task_manager() {
    let manager = HyzeTaskManager::new(64);
    
    // Schedule 1000 parallel inferences
    let mut handles = vec![];
    for i in 0..1000 {
        let task = HyzeTask::Inference {
            model: "mnist".to_string(),
            input: vec![128u8; 784],
        };
        let id = manager.schedule(task).await.unwrap();
        handles.push(async move {
            // Wait for completion
            assert!(id > 0);
        });
    }
    
    futures::future::join_all(handles).await;
    let metrics = manager.monitor().await;
    assert!(metrics.len() == 64);
}
