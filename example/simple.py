from argparse import ArgumentParser
from pprint import pprint

from transition1x import Dataloader


def main(args):  # pylint: disable=redefined-outer-name
    # loop through all configurations in the data set
    dataloader = Dataloader(args.h5file)
    for i, configuration in enumerate(dataloader):
        print('\n', configuration.pop('formula'), "\n##########")

        pprint(configuration)
        if i>10:
            break

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file", nargs='?', default="data/transition1x.h5")
    args = parser.parse_args()

    main(args)
