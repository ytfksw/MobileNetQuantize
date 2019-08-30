from utils import parse_args, create_experiment_dirs, calculate_flops
from model import MobileNet, MobileNetQuantize
from train import Train
from data_loader import DataLoader
from summarizer import Summarizer
import tensorflow as tf

from quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    if config_args.quantize == True:
        print('Quantized model created')
        # Quantized model creation
        activation_quantizer = linear_mid_tread_half_quantizer
        activation_quantizer_kwargs = {
            'bit': 2,
            'max_value': 2
        }
        weight_quantizer = binary_mean_scaling_quantizer
        weight_quantizer_kwargs = {}
        model = MobileNetQuantize(config_args,
                        activation_quantizer=activation_quantizer,
                        activation_quantizer_kwargs=activation_quantizer_kwargs,
                        weight_quantizer=weight_quantizer,
                        weight_quantizer_kwargs=weight_quantizer_kwargs)
    else:
        print('Full precision model created')
        model = MobileNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()

    if config_args.to_test:
        print("Final test!")
        trainer.test('val')
        print("Testing Finished\n\n")


if __name__ == '__main__':
    main()
