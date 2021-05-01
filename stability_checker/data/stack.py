"""STACK DataModule"""
import argparse

from torch.utils.data import random_split
from stability_checker.data.stack_dataset import STACK_dataset
from torchvision import transforms

from stability_checker.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class STACK(BaseDataModule):
    """
    Stack DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 25, 25)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(2))

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test STACK data from the collected data."""
        STACK_dataset(self.data_dir, train=True, download=True)
        STACK_dataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        stack_full = STACK_dataset(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(stack_full, [25000, 1828])  # type: ignore
        self.data_test = STACK_dataset(self.data_dir, train=False, transform=self.transform)


if __name__ == "__main__":
    load_and_print_info(STACK)
