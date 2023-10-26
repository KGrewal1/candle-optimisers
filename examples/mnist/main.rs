use clap::Parser;

mod models;
mod parse_cli;
mod training;

use models::{LinearModel, Mlp};
use parse_cli::{Args, TrainingArgs, WhichModel};
use training::training_loop;

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load the dataset
    let m = if let Some(directory) = args.local_mnist {
        candle_datasets::vision::mnist::load_dir(directory)?
    } else {
        candle_datasets::vision::mnist::load()?
    };
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let default_learning_rate = match args.model {
        WhichModel::Linear => 1.,
        WhichModel::Mlp => 0.05,
    };
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
        load: args.load,
        save: args.save,
    };

    match args.model {
        WhichModel::Linear => training_loop::<LinearModel>(m, &training_args),
        WhichModel::Mlp => training_loop::<Mlp>(m, &training_args),
    }
}
