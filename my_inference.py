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
    initialize_tensorrt_model, tensorrt_inference, to_numpy, compute_amax, collect_stats
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


def prepare_quantization_model(calibration_type):
    if calibration_type != CALIBRATION_MAX:
        quant_desc_input = QuantDescriptor(calib_method=CALIBRATION_HISTOGRAM)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    else:
        quant_desc_input = QuantDescriptor(calib_method=CALIBRATION_MAX, axis=None)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_modules.initialize()


def implement_calibration(model, dataloader, device, calibration_type, calibration_batch_count):
    quant_modules.deactivate()
    model.eval()
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model=model, data_loader=dataloader, device=device,
                      num_batches=calibration_batch_count)
        if PERCENTILE in calibration_type:
            percentile_ratio = float(f"99.{calibration_type.split('_')[-1]}")
            compute_amax(model, device,
                         method=PERCENTILE,
                         percentile=percentile_ratio)
        else:
            compute_amax(model, device,
                         method=calibration_type)


def main_func(opset_version, precision, batch_size,
              quantization_type=None,
              calibration_type=None,
              number_of_calibration_samples=None):
    resnet_models = resnet.__dict__
    MODEL = "resnet18"
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

    if not os.path.isfile(onnx_file):
        if quantization_mode:
            prepare_quantization_model(calibration_type)

        model = resnet_models[MODEL](pretrained=False, num_classes=10)
        model.load_state_dict(torch.load("./cifar10_models/state_dicts/resnet18.pt"))
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

    context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs = \
        initialize_tensorrt_model(tensorrt_file=tensorrt_file,
                                  image=to_numpy(dummy_image),
                                  output_names=["out"],
                                  outputs=dummy_outputs)

    accuracy_list = []
    time_list = []
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(val_loader):
            # Here Torch
            # image = image.to("cuda")
            # label = label.to("cuda")
            # prediction_prob = model(image)
            # prediction_prob = to_numpy(prediction_prob)

            start_time = time()

            tensorrt_inference(device_input=device_input,
                               context=context,
                               bindings=bindings,
                               device_tensorrt_outs=device_tensorrt_outs,
                               stream=stream,
                               image=image,
                               host_tensorrt_outs=host_tensorrt_outs)

            prediction_prob = host_tensorrt_outs["out"]
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
        mean_inference_time = np.mean(time_list)
        print(f"mean inference time is {mean_inference_time}")
        print(tensorrt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Torch to TensorRT correction')
    parser.add_argument('--opset_version', type=int,
                        help='opset version of the onnx file', default=13, required=True)
    parser.add_argument('--precision', help='opset version of the onnx file', default=FP32,
                        choices=[FP32, FP16, INT8], required=True)
    parser.add_argument('--batch_size', type=int,
                        help='batch size of the overall scheme', default=16, required=False)
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

    main_func(args.opset_version, args.precision, args.batch_size,
              quantization_type=args.quantization_type,
              calibration_type=args.calibration_type,
              number_of_calibration_samples=args.number_of_calibration_samples)
