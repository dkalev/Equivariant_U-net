from models import C4UNet, SteerableCNN

def parse_hparams(parser):
    parser.add_argument('--model', type=str, default='discrete', help='Model type. One of [discrete, standard, harmonic, steerable]')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for the Adam optimizer')
    parser.add_argument('--n_features', type=int, default=32, help='Number of features per block')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--loss_func', type=str, default='dice', help='Loss function for the training. One of [dice, bce]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    parser = C4UNet.add_model_specific_args(parser)
    parser = SteerableCNN.add_model_specific_args(parser)

    hparams = parser.parse_args()
    hparams = vars(hparams)
    return hparams