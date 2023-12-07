use clap::Parser;

mod models;
mod optim;
mod parse_cli;
mod training;

use models::{LinearModel, Mlp};

use candle_optimisers::adagrad::Adagrad;
use candle_optimisers::adamax::Adamax;
use candle_optimisers::esgd::SGD;
use candle_optimisers::nadam::NAdam;
use candle_optimisers::radam::RAdam;
use candle_optimisers::rmsprop::RMSprop;
use candle_optimisers::{adadelta::Adadelta, adam::Adam};

use parse_cli::{Args, TrainingArgs, WhichModel, WhichOptim};
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

    match args.optim {
        WhichOptim::Adadelta => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, Adadelta>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, Adadelta>(m, &training_args),
        },
        WhichOptim::Adagrad => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, Adagrad>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, Adagrad>(m, &training_args),
        },
        WhichOptim::Adamax => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, Adamax>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, Adamax>(m, &training_args),
        },
        WhichOptim::Sgd => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, SGD>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, SGD>(m, &training_args),
        },
        WhichOptim::NAdam => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, NAdam>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, NAdam>(m, &training_args),
        },
        WhichOptim::RAdam => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, RAdam>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, RAdam>(m, &training_args),
        },
        WhichOptim::Rms => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, RMSprop>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, RMSprop>(m, &training_args),
        },
        WhichOptim::Adam => match args.model {
            WhichModel::Linear => training_loop::<LinearModel, Adam>(m, &training_args),
            WhichModel::Mlp => training_loop::<Mlp, Adam>(m, &training_args),
        },
    }
}
