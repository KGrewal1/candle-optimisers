use clap::Parser;

mod models;
mod optim;
mod parse_cli;
mod training;

use models::{LinearModel, Mlp};
use optim::AdaDelta;
use parse_cli::{Args, TrainingArgs, WhichModel, WhichOptim};
use training::training_loop;

use crate::optim::{AdaGrad, AdaMax, NsAdam, RsAdam, RMS, SGD};

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

    match args.optim {
        WhichOptim::Adadelta => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, AdaDelta>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, AdaDelta>(m, &training_args),
        },
        WhichOptim::Adagrad => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, AdaGrad>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, AdaGrad>(m, &training_args),
        },
        WhichOptim::Adamax => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, AdaMax>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, AdaMax>(m, &training_args),
        },
        WhichOptim::SGD => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, SGD>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, SGD>(m, &training_args),
        },
        WhichOptim::NAdam => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, NsAdam>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, NsAdam>(m, &training_args),
        },
        WhichOptim::RAdam => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, RsAdam>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, RsAdam>(m, &training_args),
        },
        WhichOptim::RMS => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, RMS>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, RMS>(m, &training_args),
        },
    }
}
