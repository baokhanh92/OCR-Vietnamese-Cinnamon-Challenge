import argparse
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--action", type=str, required=True, help='Path to the image')
    ap.add_argument("--folder", type=str, required=True, help='Path to the image')
    args = vars(ap.parse_args())

    print(args) 