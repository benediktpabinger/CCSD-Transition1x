import os
from argparse import ArgumentParser
from urllib.request import urlretrieve

import progressbar


class ProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def main(args):  # pylint: disable=redefined-outer-name
    os.makedirs(args.dir, exist_ok=True)

    print(f"Downloading Transition1x data to {args.dir}/Transition1x.h5")
    urlretrieve(
        "https://figshare.com/ndownloader/files/36035789",
        os.path.join(args.dir, "Transition1x.h5"),
        ProgressBar()
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", nargs="?", default="data")
    args = parser.parse_args()

    main(args)
