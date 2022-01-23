from torchvision.models.resnet import resnet18
from cifar10_models import resnet
import torchvision.transforms as T
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
from time import time
from math import ceil
import os
import argparse
from common_keys import *
from deployment_utils import save_onnx_model, onnx_to_tensorrt, \
    initialize_tensorrt_model, tensorrt_inference, to_numpy,\
    prepare_quantization_model, implement_calibration, print_size_of_model

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
import torch.nn.utils.prune as prune


def load_dataloader(batch_size, is_train=False):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10("/home/ek21/remote-pycharm/PyTorch_CIFAR10/data",
                      train=is_train, transform=transform, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader


def main_func(opset_version, precision, batch_size, inference_type, model,
              quantization_type=None,
              calibration_type=None,
              number_of_calibration_samples=None):
    resnet_models = resnet.__dict__
    MODEL = model
    weight_file = "weights/"
    device = "cuda"
    batch_size = batch_size
    number_of_calibration_batch = 0
    if precision == INT8 and quantization_type is not None and calibration_type is not None:
        number_of_calibration_batch = ceil(number_of_calibration_samples / batch_size)
        number_of_calibration_samples = batch_size * number_of_calibration_batch
        quantization_mode = True
        naming = f"{MODEL}_{quantization_type}_{calibration_type}_ncs_{number_of_calibration_samples}_op{opset_version}"
    else:
        quantization_mode = False
        naming = f"{MODEL}_{precision}_op{opset_version}"

    tensorrt_file = f"{weight_file}{naming}.{TENSORRT_EXTENSION}"
    onnx_file = f"{weight_file}{naming}.{ONNX_EXTENSION}"

    val_loader = load_dataloader(batch_size, is_train=False)
    train_loader = load_dataloader(batch_size, is_train=True)

    dummy_image = torch.zeros([batch_size, 3, 32, 32])
    dummy_image = dummy_image.to(device)
    prediction_prob = torch.zeros(batch_size, 10)
    dummy_outputs = {"out": prediction_prob}

    if inference_type == TENSORRT_INFERENCE:
        if not os.path.isfile(onnx_file):
            if quantization_mode:
                prepare_quantization_model(calibration_type)

            model = resnet_models[MODEL](pretrained=False, num_classes=10)
            model.load_state_dict(torch.load(f"./cifar10_models/state_dicts/{MODEL}.pt"))
            model.to(device)
            model.eval()

            if quantization_mode:
                implement_calibration(model=model, dataloader=train_loader, device=device,
                                      calibration_type=calibration_type,
                                      calibration_batch_count=number_of_calibration_batch)

            if quantization_mode:
                quant_nn.TensorQuantizer.use_fb_fake_quant = True
            save_onnx_model(inputs=dummy_image, outputs=dummy_outputs, model=model,
                            onnx_file=onnx_file, opset_version=opset_version)

        if not os.path.isfile(tensorrt_file):
            onnx_to_tensorrt(onnx_file=onnx_file,
                             tensorrt_file=tensorrt_file,
                             precision=precision)
            print_size_of_model(tensorrt_file)

        context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs = \
            initialize_tensorrt_model(tensorrt_file=tensorrt_file,
                                      image=to_numpy(dummy_image),
                                      output_names=["out"],
                                      outputs=dummy_outputs)

    elif inference_type == TORCH_INFERENCE:
        model = resnet_models[MODEL](pretrained=False, num_classes=10)
        model.load_state_dict(torch.load("./cifar10_models/state_dicts/resnet18.pt"))
        model.to(device)
        model.eval()

    accuracy_list = []
    time_list = []
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(val_loader):

            start_time = time()
            if inference_type == TENSORRT_INFERENCE:
                tensorrt_inference(device_input=device_input,
                                   context=context,
                                   bindings=bindings,
                                   device_tensorrt_outs=device_tensorrt_outs,
                                   stream=stream,
                                   image=image,
                                   host_tensorrt_outs=host_tensorrt_outs)

                prediction_prob = host_tensorrt_outs["out"]

            elif inference_type == TORCH_INFERENCE:
                image = image.to(device)
                label = label.to(device)
                prediction_prob = model(image)
                prediction_prob = to_numpy(prediction_prob)

            predictions = prediction_prob.argmax(1)
            label = to_numpy(label)

            end_time = time()
            inference_time = end_time - start_time
            if batch_index != 0:
                time_list.append(inference_time)
            number_of_corrects = np.sum(predictions == label)
            accuracy_list.append(number_of_corrects / batch_size)
        accuracy = np.mean(accuracy_list)
        print(f"average accuracy is {accuracy}")
        mean_inference_time = np.mean(time_list) * 1000
        print(f"mean inference time is {mean_inference_time: 0.4f} ms")
        if inference_type == TENSORRT_INFERENCE:
            print(tensorrt_file)
        else:
            print("torch inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch to TensorRT correction')
    parser.add_argument('--opset_version', type=int,
                        help='opset version of the onnx file', default=13, required=True)
    parser.add_argument('--precision', help='opset version of the onnx file', default=FP32,
                        choices=[FP32, FP16, INT8], required=True)
    parser.add_argument('--batch_size', type=int,
                        help='batch size of the overall scheme', default=16, required=False)
    parser.add_argument('--inference_type',
                        help='In order to decide whether tensorrt or torch inference', default=TENSORRT_INFERENCE,
                        required=False, choices=[TORCH_INFERENCE, TENSORRT_INFERENCE])
    parser.add_argument('--model',
                        help='architecture model', default=RESNET18, required=False,
                        choices=[RESNET18, RESNET34, RESNET50])
    parser.add_argument('--quantization_type', help='Quantization type, '
                                                    'which are post training quantization(ptq) or '
                                                    'quantization aware training (qat)',
                        required=False, choices=[POST_TRAINING, QUANTIZATION_AWARE])
    parser.add_argument('--calibration_type', help='Determines the calibration type of the int8 quantization',
                        required=False, choices=[CALIBRATION_MAX, CALIBRATION_MSE, CALIBRATION_PERCENTILE_9,
                                                 CALIBRATION_PERCENTILE_99, CALIBRATION_PERCENTILE_999,
                                                 CALIBRATION_PERCENTILE_9999, CALIBRATION_ENTROPY])
    parser.add_argument('--number_of_calibration_samples', help='determine the number_of_calibration_samples. '
                                                                'The given number is ceiled to make it '
                                                                'exactly divisible by the number of batch',
                        required=False, type=int)

    args = parser.parse_args()

    main_func(args.opset_version, args.precision, args.batch_size, inference_type=args.inference_type,
              model=args.model,
              quantization_type=args.quantization_type,
              calibration_type=args.calibration_type,
              number_of_calibration_samples=args.number_of_calibration_samples)
