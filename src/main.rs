use dfdx::optim::{Sgd, SgdConfig};
use dfdx::prelude::*;
use image::{GenericImage, ImageBuffer, Luma};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::path::Path;
use walkdir::WalkDir;

type Discriminator = (
    (Linear<784, 784>, LeakyReLU<f32>, DropoutOneIn<2>),
    (Linear<784, 128>, LeakyReLU<f32>, DropoutOneIn<2>),
    (Linear<128, 64>, LeakyReLU<f32>, DropoutOneIn<2>),
    (Linear<64, 16>, LeakyReLU<f32>),
    Linear<16, 1>,
);

type Generator = (
    (Linear<16, 256>, LeakyReLU<f32>),
    (Linear<256, 784>, LeakyReLU<f32>),
    (Linear<784, 2024>, LeakyReLU<f32>),
    Linear<2024, 784>,
);

const BATCH_SIZE: usize = 10;

const ITERATIONS: usize = 500_000;

fn main() {
    dfdx::flush_denormals_to_zero();

    let dev: Cpu = Default::default();

    let mut discriminator = Discriminator::build_on_device(&dev);
    let mut discriminator_grads = discriminator.alloc_grads();

    let mut discriminator_sgd = Sgd::new(
        &discriminator,
        SgdConfig {
            lr: 3e-2,
            momentum: None,
            weight_decay: None,
        },
    );

    let mut generator = Generator::build_on_device(&dev);
    let mut generator_grads = generator.alloc_grads();

    let mut generator_sgd = Sgd::new(
        &generator,
        SgdConfig {
            lr: 5e-3,
            momentum: None,
            weight_decay: None,
        },
    );

    let training_data = TrainingData::new();

    let fixed_latent: Tensor<Rank2<16, 16>, f32, _> = dev.sample_normal();

    let discriminator_train_labels: Vec<f32> = (0..BATCH_SIZE)
        .map(|_| 0.0)
        .chain((0..BATCH_SIZE).map(|_| 1.0))
        .collect();

    for iteration in 1..=ITERATIONS {
        // Train discriminator
        let mut x_raw = training_data.sample_n_training_entries(BATCH_SIZE);

        let latent_input: Tensor<Rank2<BATCH_SIZE, 16>, f32, _> = dev.sample_normal();

        let generated_data = generator.forward(latent_input);

        x_raw.extend(generated_data.as_vec().iter());

        let x = dev.tensor_from_vec(x_raw, (BATCH_SIZE * 2, Const::<784>));

        let y = dev.tensor_from_vec(
            discriminator_train_labels.clone(),
            (BATCH_SIZE * 2, Const::<1>),
        );

        let prediction = discriminator.forward_mut(x.traced(discriminator_grads));

        let loss = mse_loss(prediction, y);

        if iteration % 1000 == 0 {
            println!(
                "Iteration: {iteration}, Discriminator loss: {}",
                loss.array()
            );
        }

        discriminator_grads = loss.backward();
        discriminator_sgd
            .update(&mut discriminator, &discriminator_grads)
            .expect("Unable to update weights of discriminator");
        discriminator.zero_grads(&mut discriminator_grads);

        // Train generator
        let latent_input: Tensor<Rank2<BATCH_SIZE, 16>, f32, _> = dev.sample_normal();

        let generated_predictions =
            discriminator.forward_mut(generator.forward_mut(latent_input.traced(generator_grads)));

        // Add some randomness to labels of generator
        let gen_y: Tensor<Rank2<BATCH_SIZE, 1>, f32, _> = dev.zeros() + dev.sample_normal() * 0.2;

        let generated_loss = mse_loss(generated_predictions, gen_y);

        if iteration % 1000 == 0 {
            println!(
                "Iteration: {iteration}, Generator loss: {}",
                generated_loss.array()
            );
        }

        generator_grads = generated_loss.backward();
        generator_sgd
            .update(&mut generator, &generator_grads)
            .expect("Unable to update weights of generator");
        generator.zero_grads(&mut generator_grads);

        if iteration % 2000 == 0 {
            let output = generator.forward(fixed_latent.clone());
            build_image(
                &output.as_vec(),
                format!("iteration_{iteration}.jpg").as_str(),
            );
        }
    }
}

struct TrainingData {
    data: Vec<Vec<f32>>,
}

fn build_image(image_data: &[f32], file_name: &str) {
    let transformed_data: Vec<u8> = image_data.iter().map(|&d| (d * 255.0) as u8).collect();

    let mut main: ImageBuffer<Luma<u8>, _> = image::ImageBuffer::new(4 * 28, 4 * 28);

    for image_idx in 0..16 {
        main.copy_from(
            &ImageBuffer::from_raw(
                28,
                28,
                &transformed_data[(image_idx * 28 * 28)..((image_idx + 1) * 28 * 28)],
            )
            .unwrap(),
            (image_idx as u32 / 4) * 28,
            (image_idx as u32 % 4) * 28,
        )
        .unwrap();
    }

    main.save(Path::new(file_name)).unwrap();
}

impl TrainingData {
    fn new() -> Self {
        let mut data = vec![];

        for integer in 0..10 {
            for entry in WalkDir::new(Path::new(&format!("data/training/{integer}"))) {
                let dir_entry = entry.unwrap();

                if !dir_entry.path().is_file() {
                    continue;
                }

                let img = image::open(dir_entry.path()).unwrap();

                data.push(img.as_bytes().iter().map(|&d| d as f32 / 255.0).collect());
            }
        }

        data.shuffle(&mut thread_rng());

        Self { data }
    }

    fn sample_n_training_entries(&self, n: usize) -> Vec<f32> {
        self.data
            .choose_multiple(&mut thread_rng(), n)
            .flatten()
            .copied()
            .collect()
    }
}
