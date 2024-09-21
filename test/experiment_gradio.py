import gradio as gr
import torch
import cv2
import numpy as np
import time
import os
import sys

param_path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(param_path))
sys.path.append(dir_path)

from src import model
from src.utility import param

args = param.args
device = torch.device(
    f"cuda:{args.gpu_id}") if args.gpu_id != -1 else torch.device("cpu")

gradio_path = os.path.join(dir_path, "test", "gradio")

example_images = [
    "data/0970x4.png", "data/0991x4.png", "data/0939x4.png", "data/0817x4.png",
    "data/0898x4.png", "data/1100x4.png", "data/1022x4.png", "data/1199x4.png",
    "data/1111x4.png", "data/1020x4.png", "data/0449x4.png", "data/0746x4.png",
    "data/0298x4.png", "data/0752x4.png", "data/0187x4.png", "data/0604x4.png",
    "data/0594x4.png", "data/0646x4.png", "data/0590x4.png", "data/0579x4.png"
]

for i in range(len(example_images)):
    example_images[i] = f"{dir_path}/{example_images[i]}"

css_style = """
            #button-1 {
                background: linear-gradient(135deg, #F902FF, #00DBDE)
                }
            .title {
                font-family: "微软雅黑";
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                background: linear-gradient(135deg, #F902FF, #00DBDE); /*设置渐变的方向从左到右 颜色从ff0000到ffff00*/
                -webkit-background-clip: text;/*将设置的背景颜色限制在文字中*/
                -webkit-text-fill-color: transparent;/*给文字设置成透明*/
                margin-bottom: 10px;  /* 添加下边距 */

            }

            .clear {
                font-family: "微软雅黑";
                font-size: 30px;
                font-weight: bold;
                text-align: center;
                background: linear-gradient(135deg, #F902FF, #00DBDE); /*设置渐变的方向从左到右 颜色从ff0000到ffff00*/
                -webkit-background-clip: text;/*将设置的背景颜色限制在文字中*/
                -webkit-text-fill-color: transparent;/*给文字设置成透明*/

            }

            .blur {
                font-family: "微软雅黑";   /* 设置字体 */
                font-size: 30px;           /* 设置文字大小 */
                color:transparent;  /*设置文字颜色透明*/
                text-shadow: 0px 1px 4px #8554C8;  /* 设置文字阴影 */
                text-align: center; 
            }
            """


def Super_resolution_img(img):
    # 处理图像并返回超分辨率图像
    img = torch.from_numpy(img.transpose(
        (2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img).round()
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return output


def Super_resolution_large(img, block_size=1024, extend=24):
    block_img_size = block_size - extend
    H, W = img.shape[:2]
    n_h = (H + block_img_size - 1) // block_img_size
    n_w = (W + block_img_size - 1) // block_img_size

    output = np.zeros((H * 4, W * 4, 3))

    for h in range(n_h):
        for w in range(n_w):
            if h != n_h - 1:
                h_start = h * block_img_size
                h_end = h_start + block_size

                h_start_block = 0
                h_end_block = block_img_size * 4

                h_start_sr = h_start * 4
                h_end_sr = h_start_sr + block_img_size * 4

            else:
                h_start = H - 1 - block_size
                h_end = H - 1

                h_start_block = -1 - block_img_size * 4
                h_end_block = -1

                h_start_sr = 4 * H - 1 - block_img_size * 4
                h_end_sr = 4 * H - 1

            if w != n_w - 1:
                w_start = w * block_img_size
                w_end = w_start + block_size

                w_start_block = 0
                w_end_block = block_img_size * 4

                w_start_sr = w_start * 4
                w_end_sr = w_start_sr + block_img_size * 4

            else:
                w_start = W - 1 - block_size
                w_end = W - 1

                w_start_block = -1 - block_img_size * 4
                w_end_block = -1

                w_start_sr = 4 * W - 1 - block_img_size * 4
                w_end_sr = 4 * W - 1

            img_block = img[h_start:h_end, w_start:w_end, :]
            time0 = time.time()

            out_block = Super_resolution_img(img_block)[
                h_start_block:h_end_block, w_start_block:w_end_block, :]
            output[h_start_sr:h_end_sr, w_start_sr:w_end_sr, :] = out_block

    return np.uint8(output[:H * 4 - 1, :W * 4 - 1, :])


def process_image(img):
    img_lr = img
    h, w = img.shape[:2]

    if h < 1024 or w < 1024:
        img_sr = Super_resolution_img(img_lr)

    else:
        img_sr = Super_resolution_large(img_lr)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    img_sr = cv2.filter2D(img_sr, -1, kernel)

    cv2.imwrite(f"{gradio_path}/img_sr.png",
                cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))

    window_lr = cv2.resize(img_lr, (w * 4, h * 4),
                           interpolation=cv2.INTER_LINEAR)

    window_lr = window_lr[2 * h - 200:2 * h + 200, 2 * w - 300:2 * w + 300]
    window_sr = img_sr[2 * h - 200:2 * h + 200, 2 * w - 300:2 * w + 300]
    # window_lr = window_lr[int(1.5 * h):int(2.5 * h), int(1.5 * w):int(2.5 * w)]
    # window_sr = img_sr[int(1.5 * h):int(2.5 * h), int(1.5 * w):int(2.5 * w)]

    # window1 = np.hstack(
    #     (window_lr[:int(h / 2), :int(w / 2)], window_sr[:int(h / 2),
    #                                                     int(w / 2):]))
    # window2 = np.hstack(
    #     (window_sr[int(h / 2):, :int(w / 2)], window_lr[int(h / 2):,
    #                                                     int(w / 2):]))
    # window = np.vstack((window1, window2))
    # window[h // 2 - h // 200:h // 2 + h // 200, :] = (0, 0, 0)
    # window[:, w // 2 - h // 200:w // 2 + h // 200] = (0, 0, 0)

    window1 = np.hstack((window_lr[:200, :300], window_sr[:200, 300:]))
    window2 = np.hstack((window_sr[200:, :300], window_lr[200:, 300:]))
    window = np.vstack((window1, window2))
    window[200 - 1:200 + 1, :] = (0, 0, 0)
    window[:, 300 - 1:300 + 1] = (0, 0, 0)

    cv2.imwrite(f"{gradio_path}/img_window.png",
                cv2.cvtColor(window, cv2.COLOR_RGB2BGR))

    return f"{gradio_path}/img_sr.png", f"{gradio_path}/img_window.png"


if __name__ == "__main__":
    # 加载模型
    args = param.args
    RCAN = model.RCAN(args)
    pre_path = args.pre_train if args.pre_train != '.' else args.dir_data + "/experiment/RCAN/model/model_best.pth"
    RCAN.load_state_dict(torch.load(pre_path, weights_only=True))

    model = RCAN.to(device)
    model.eval()

    with gr.Blocks(css=css_style) as iface:
        gr.HTML(value="<div class='title'>图片超分辨率</div>")
        gr.HTML(
            value=
            "<div style='text-align: center'><span class='clear'>画质糊成&nbsp;</span> <span class='blur'>马赛克</span><span class='clear'>&nbsp;? 超分辨率拯救你！为屏幕戴上老花镜！</span></div>"
        )
        gr.HTML(
            value=
            "<div style='text-align: center'><span class='blur'>1080P (X)</span> <span class='clear'>&nbsp;&nbsp;->&nbsp;&nbsp;4K (√)</span></div>"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="输入图片")
            with gr.Column():
                with gr.Row():
                    gr.Examples(examples=example_images,
                                inputs=input_image,
                                label="示例图片")
                with gr.Row():
                    submit_btn = gr.Button("Submit", elem_id="button-1")

                    clear_btn = gr.Button("Clear")

        with gr.Row():
            with gr.Column():
                output_image1 = gr.Image(type="numpy", label="SR-x4")
            with gr.Column():
                output_image2 = gr.Image(type="numpy", label="Contrast")

        submit_btn.click(fn=process_image,
                         inputs=input_image,
                         outputs=[output_image1, output_image2])

        clear_btn.click(fn=lambda: (None, None, None),
                        inputs=None,
                        outputs=[input_image, output_image1, output_image2])

    iface.launch(share=False)
