use candle_core::Result as CResult;
use candle_datasets::vision::Dataset;
use criterion::{criterion_group, criterion_main, Criterion};
use optimisers::{
    adadelta::Adadelta, adagrad::Adagrad, adam::Adam, adamax::Adamax, esgd::MomentumEnhancedSGD,
    nadam::NAdam, radam::RAdam, rmsprop::RMSprop,
};
use training::Mlp;

// mod models;
// mod optim;
mod training;

fn load_data() -> CResult<Dataset> {
    candle_datasets::vision::mnist::load()
}

#[allow(clippy::missing_panics_doc)]
pub fn criterion_benchmark_std(c: &mut Criterion) {
    let mut group = c.benchmark_group("std-optimisers");
    let m = &load_data().expect("Failed to load data");
    // let m = Rc::new(m);

    group.significance_level(0.1).sample_size(100);
    group.bench_function("adadelta", |b| {
        b.iter(|| {
            training::run_training::<Mlp, Adadelta>(m).expect("Failed to setup training");
        });
    });
    group.bench_function("adagrad", |b| {
        b.iter(|| {
            training::run_training::<Mlp, Adagrad>(m).expect("Failed to setup training");
        });
    });
    group.bench_function("adam", |b| {
        b.iter(|| training::run_training::<Mlp, Adam>(m).expect("Failed to setup training"));
    });
    group.bench_function("adamax", |b| {
        b.iter(|| {
            training::run_training::<Mlp, Adamax>(m).expect("Failed to setup training");
        });
    });
    group.bench_function("esgd", |b| {
        b.iter(|| {
            training::run_training::<Mlp, MomentumEnhancedSGD>(m)
                .expect("Failed to setup training");
        });
    });
    group.bench_function("nadam", |b| {
        b.iter(|| {
            training::run_training::<Mlp, NAdam>(m).expect("Failed to setup training");
        });
    });
    group.bench_function("radam", |b| {
        b.iter(|| {
            training::run_training::<Mlp, RAdam>(m).expect("Failed to setup training");
        });
    });
    group.bench_function("rmsprop", |b| {
        b.iter(|| {
            training::run_training::<Mlp, RMSprop>(m).expect("Failed to setup training");
        });
    });

    group.finish();
}

#[allow(clippy::missing_panics_doc)]
pub fn criterion_benchmark_lbfgs(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgs-optimser");
    let m = load_data().expect("Failed to load data");
    // let m = Rc::new(m);

    group.significance_level(0.1).sample_size(10);

    group.bench_function("lbfgs", |b| {
        b.iter(|| training::run_lbfgs_training::<Mlp>(&m).expect("Failed to setup training"));
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark_std, criterion_benchmark_lbfgs);
criterion_main!(benches);
