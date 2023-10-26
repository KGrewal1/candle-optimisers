use crate::{models::Model, optim::Optim, parse_cli::TrainingArgs};
use candle_core::{DType, D};
use candle_nn::{loss, ops, VarBuilder, VarMap};

pub fn training_loop<M: Model, O: Optim>(
    m: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    println!("Training on device {:?}", dev);

    // get the labels from the dataset
    let train_labels = m.train_labels;
    // get the input from the dataset and put on device
    let train_images = m.train_images.to_device(&dev)?;
    // get the training labels on the device
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // create a new variable store
    let mut varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    // create model from variables
    let model = M::new(vs.clone())?;

    // see if there are pretrained weights to load
    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        varmap.load(load)?
    }

    // create an optimizer
    let mut optimiser = O::new(varmap.all_vars(), args.learning_rate)?;
    // candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
    // load the test images
    let test_images = m.test_images.to_device(&dev)?;
    // load the test labels
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // loop for model optimization
    for epoch in 1..args.epochs {
        // get log probabilities of results
        let logits = model.forward(&train_images)?;
        // softmax the log probabilities
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // get the loss
        let loss = loss::nll(&log_sm, &train_labels)?;
        // step the tensors by backpropagating the loss
        optimiser.backward_step(&loss)?;

        // get the log probabilities of the test images
        let test_logits = model.forward(&test_images)?;
        // get the sum of the correct predictions
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        // get the accuracy on the test set
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        varmap.save(save)?
    }
    Ok(())
}
