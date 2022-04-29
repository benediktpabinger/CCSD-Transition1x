import os
from argparse import ArgumentParser
from urllib.request import urlretrieve


def main(args):  # pylint: disable=redefined-outer-name
    os.makedirs(args.dir,exist_ok=True) # , exists_ok=True)

    urlretrieve(
        "https://figshare.com/ndownloader/files/34935780",
        os.path.join(args.dir, "t1x.h5"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", default='.')
    args = parser.parse_args()

    main(args)
