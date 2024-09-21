import torch
import cv2
import os
from tqdm import tqdm
import numpy as np
import sys

param_path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(param_path))
sys.path.append(dir_path)

from src import model
from src.utility import param

args = param.args
device = torch.device(f"cuda:{args.gpu_id}") if args.gpu_id != -1 else torch.device("cpu")

def Super_resolution_img(img, network):
    img = torch.from_numpy(img.transpose(
        (2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = network(img).round()
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return output


def main():
    print("Loading model ...")
    apath = args.dir_data + "/test"
    network = model.RCAN(args)
    pre_path = args.pre_train if args.pre_train != '.' else args.dir_data + "/experiment/RCAN/model/model_best.pth"
    network.load_state_dict(
        torch.load(pre_path, weights_only=True))

    network = network.to(device)
    network.eval()

    print("Done\n")
    print("Begin Super resolution: ")

    if args.img_path == '.':

        file_names = os.listdir(os.path.join(apath, "test_picture"))

        loop = tqdm(file_names, dynamic_ncols=100)
        for file in loop:
            img = cv2.imread(os.path.join(apath, "test_picture", file))
            result = Super_resolution_img(img, network)
            cv2.imwrite(
                os.path.join(apath, "sr_picture",
                             f"{os.path.basename(file)}x4.png"), result)

            if args.is_sharp:
                sr_sharp = cv2.filter2D(result, -1, kernel)
                cv2.imwrite(
                    os.path.join(apath, "sr_picture",
                                f"{os.path.basename(file)}x4_shape.png"),
                    sr_sharp)

    else:
        img = cv2.imread(args.img_path)
        result = Super_resolution_img(img, network)

        cv2.imwrite(
            os.path.join(apath, "sr_picture",
                         f"{os.path.basename(args.img_path)}x4.png"), result)
        
        if args.is_sharp:
            sr_sharp = cv2.filter2D(result, -1, kernel)
            cv2.imwrite(
                os.path.join(apath, "sr_picture",
                            f"{os.path.basename(args.img_path)}x4_shape.png"),
                sr_sharp)

    print("Done .")


if __name__ == "__main__":
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    main()
