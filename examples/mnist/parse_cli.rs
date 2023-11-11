use clap::{Parser, ValueEnum};

pub struct TrainingArgs {
    pub learning_rate: f64,
    pub load: Option<String>,
    pub save: Option<String>,
    pub epochs: usize,
}

#[derive(ValueEnum, Clone)]
pub enum WhichModel {
    Linear,
    Mlp,
}

#[derive(ValueEnum, Clone)]
pub enum WhichOptim {
    Adadelta,
    Adagrad,
    Adamax,
    Sgd,
    NAdam,
    RAdam,
    Rms,
    Adam,
}

#[derive(Parser)]
pub struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Linear)]
    pub model: WhichModel,

    #[arg(long, value_enum, default_value_t = WhichOptim::Adadelta)]
    pub optim: WhichOptim,

    #[arg(long)]
    pub learning_rate: Option<f64>,

    #[arg(long, default_value_t = 200)]
    pub epochs: usize,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long)]
    pub save: Option<String>,

    /// The file where to load the trained weights from, in safetensors format.
    #[arg(long)]
    pub load: Option<String>,

    /// The directory where to load the dataset from, in ubyte format.
    #[arg(long)]
    pub local_mnist: Option<String>,
}
