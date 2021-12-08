from mnist_dm import MNISTDataModule
from lit_mnist_module import LitMNISTModule
from pytorch_lightning import Trainer

from argparse import ArgumentParser

def cli_main():
    parser = ArgumentParser()
    #parser.add_argument("--conda_env", type=str, default"") # Program level args
    parser = LitMNISTModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    mnist_dm = MNISTDataModule()
    mnist_dm.prepare_data()

    model = LitMNISTModule(args)
    trainer = Trainer.from_argparse_args(args, max_epochs=100)
    trainer.fit(model, mnist_dm)

    trainer.test()

if __name__ == '__main__':
    cli_main()