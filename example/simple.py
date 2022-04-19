from argparse import ArgumentParser
from pprint import pprint

from t1x import Dataloader


def main(args):  # pylint: disable=redefined-outer-name
    # loop through all configurations in the data set
    dataloader = Dataloader(args.h5file)
    for configuration in dataloader:
        pprint(configuration)


    # loop through all product, reactant, transition state triplets in the dat aset
    dataloader = Dataloader(args.h5file, only_final=True)
    for configuration in dataloader:
        pprint(configuration)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file")
    args = parser.parse_args()

    main(args)
