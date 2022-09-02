import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
from utils.datasets import LoadStreams
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, check_img_size
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def main():

    source = '2'
    imgsz  = 640

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    for path, image, im0s, vid_cap in dataset:
        image = torch.from_numpy(image).to(device)
        image = image.half()
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        if torch.cuda.is_available():
            image = image.half().to(device)
        t1 = time_synchronized()
        output, _ = model(image)
        t2 = time_synchronized()


        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        t3 = time_synchronized()
        with torch.no_grad():
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        cv2.imshow('test', nimg)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()