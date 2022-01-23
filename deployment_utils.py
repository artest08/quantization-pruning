import subprocess
from typing import Tuple, List, Optional, Union, Dict
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
import os
import onnxruntime
from common_keys import *
from tqdm import tqdm
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
import tempfile


def to_numpy(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts torch tensor to numpy array by proper operations
    :param input_tensor:
    :return:
    """
    input_numpy = input_tensor.detach().cpu().numpy()
    return input_numpy


def my_flatten(inputs: Union[torch.Tensor,
                             Dict[str, torch.Tensor],
                             List[torch.Tensor],
                             List[List[torch.Tensor]],
                             Tuple[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Converts some form of torch Tensor to the list of torch tensor format
    :param inputs:
    :return:
    """
    return_list = []
    if isinstance(inputs, (list, tuple)):
        for inp in inputs:
            return_list += my_flatten(inp)
    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            return_list += my_flatten(value)
    else:
        return_list.append(inputs)
    return return_list


def snpe_net_run(input_list_file: str,
                 container_file: str,
                 output_dir: str):
    """
    SNPE net run utility. The functions initiates a bash command.


    The SNPE expect and raw image file list, the container(model) file in dlc format and
    the output directory in order to save the outputs in raw format.


    The outputs are written in sub-directories as Result_<output_idx>
    :param input_list_file: The txt file indicating the input raw files. The example raw txt file is given as below


        # Upsample_66 Upsample_71
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_0.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_1.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_2.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_3.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_4.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_5.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_6.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_7.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_8.raw
        weights/deeplabv3_resnet18_culane_400x232/image_files/demo_9.raw`

    In this example, Upsample_66 Upsample_71 denote the output nodes of the container or model

    :param container_file: The model file in dlc format
    :param output_dir: The output directory where the raw outputs will be saved
    :return: None
    """
    subprocess.run(f'snpe-net-run'
                   f' --input_list {input_list_file}'
                   f' --container {container_file}'
                   f' --output_dir {output_dir}',
                   check=True,
                   shell=True)


def snpe_onnx_to_dlc(onnx_file: str,
                     dlc_file: str):
    """
    This is a functions which initiates a bash command in order to convert onnx file to dlc format
    :param onnx_file:
    :param dlc_file:
    :return:
    """
    subprocess.run(f'snpe-onnx-to-dlc -i {onnx_file}'
                   f' -o {dlc_file}',
                   check=True,
                   shell=True)


def snpe_dlc_to_quantized_dlc(raw_list_python_file_location: str,
                              dlc_file: str,
                              dlc_quantized_file: str):
    """
    This function converts the dlc model to the quantized dlc model by using the sample images
    indicated or determined in the text file (raw list file)
    :param raw_list_python_file_location: The text file name in which the sample raw images are indicated
    :param dlc_file: The input file or model
    :param dlc_quantized_file: The output file where the quantized dlc file will be saved
    :return:
    """
    subprocess.run(f'snpe-dlc-quantize'
                   f' --input_list {raw_list_python_file_location}'
                   f' --input_dlc {dlc_file}'
                   f' --output_dlc {dlc_quantized_file}',
                   check=True,
                   shell=True)


def create_onnx_to_tensorrt_command_string(onnx_file: str, tensorrt_file: str):
    onnx_file = onnx_file.replace('(', '\(')
    onnx_file = onnx_file.replace(')', '\)')

    tensorrt_file = tensorrt_file.replace('(', '\(')
    tensorrt_file = tensorrt_file.replace(')', '\)')

    create_string = f'trtexec --onnx={onnx_file} --saveEngine={tensorrt_file} --explicitBatch'

    return create_string

def apply_conversion_command(command_string):
    subprocess.run(command_string,
                   check=True,
                   shell=True)

def onnx_to_tensorrt(onnx_file: str,
                     tensorrt_file: str,
                     precision: str):
    """
    A function that initiates a bash command which converts the onnx model to tensorrt model
    Args:
        :param onnx_file:
        :param tensorrt_file:
        :param precision: indicates the precision of tensorrt model
    :return:
    """

    command_string = create_onnx_to_tensorrt_command_string(onnx_file=onnx_file, tensorrt_file=tensorrt_file)
    if precision == FP32:
        pass
    elif precision == FP16:
        command_string += ' --fp16'
    elif precision == INT8:
        command_string += ' --int8'
    apply_conversion_command(command_string)


def tensorrt_inference(
        device_input: Union[pycuda._driver.DeviceAllocation, Dict[str, pycuda._driver.DeviceAllocation]],
        device_tensorrt_outs: Union[pycuda._driver.DeviceAllocation, Dict[str, pycuda._driver.DeviceAllocation]],
        image: Union[np.ndarray, Dict[str, np.ndarray]],
        stream: pycuda._driver.Stream,
        context: trt.tensorrt.IExecutionContext,
        bindings: List[int],
        host_tensorrt_outs: Union[np.ndarray, Dict[str, np.ndarray]]
):
    """
    This function implements the tensorRT inference for the given parameters.

    It should be utilized with initialize_tensorrt_model whose outputs are inputs to this function.
    An example utilization can be seen in check_tensorrt_model function.

    :param device_input:
    :param device_tensorrt_outs:
    :param image:
    :param stream:
    :param context:
    :param bindings:
    :param host_tensorrt_outs:
    :return:
    """
    if isinstance(image, torch.Tensor):
        image_numpy = to_numpy(image)
    elif isinstance(image, np.ndarray):
        image_numpy = image
    else:
        raise ValueError("Unsupported Image Format")

    # transfer input data to device
    cuda.memcpy_htod_async(device_input, image_numpy, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    for keys in host_tensorrt_outs.keys():
        cuda.memcpy_dtoh_async(host_tensorrt_outs[keys], device_tensorrt_outs[keys], stream)
        # syncronize threads
        stream.synchronize()


def initialize_tensorrt_model(tensorrt_file: str,
                              image: Union[torch.Tensor, np.ndarray],
                              output_names: List[str],
                              outputs: Optional[Dict[str, torch.Tensor]] = None,
                              number_of_classes: Optional[int] = None,
                              ) -> Tuple[trt.tensorrt.IExecutionContext, List[int],
                                         Union[pycuda._driver.DeviceAllocation, Dict[
                                             str, pycuda._driver.DeviceAllocation]],
                                         Union[pycuda._driver.DeviceAllocation, Dict[
                                             str, pycuda._driver.DeviceAllocation]],
                                         pycuda._driver.Stream,
                                         Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    A function that initializes or prepares a tensorRT model for inference.
    Either example outputs or number of classes should be given as input.

    :param tensorrt_file:
    :param image:
    :param output_names: The name of the outputs, which is are required input for the function
    :param outputs:
    :param number_of_classes:
    :return:
    """
    f = open(tensorrt_file, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    if isinstance(image, torch.Tensor):
        image_numpy = to_numpy(image)
    elif isinstance(image, np.ndarray):
        image_numpy = image
    else:
        raise ValueError("Unsupported Image Format")
    # allocate device memory
    device_input = cuda.mem_alloc(1 * image_numpy.nbytes)

    host_tensorrt_outs = {}
    device_tensorrt_outs = {}
    bindings = [int(device_input)]
    for output_name in output_names:
        if outputs:
            empty_out = np.empty_like(to_numpy(outputs[output_name]))
        else:
            if isinstance(number_of_classes, List):
                if len(number_of_classes) == 1:
                    dimension = [1, number_of_classes] + list(image.shape[2:])
                else:
                    dimension = None
                    for index, number_of_class in enumerate(number_of_classes):
                        if f'_{index+1}' in output_name:
                            dimension = [1, number_of_class] + list(image.shape[2:])
                            break
                    if dimension is None:
                        dimension = [1, number_of_classes[0]] + list(image.shape[2:])
            else:
                dimension = [1, number_of_classes] + list(image.shape[2:])
            empty_out = np.empty(dimension, dtype=np.float32)
        host_tensorrt_outs.update({output_name: empty_out})
        device_output_allocation = cuda.mem_alloc(1 * empty_out.nbytes)
        device_tensorrt_outs.update({output_name: device_output_allocation})
        bindings.append(int(device_output_allocation))

    stream = cuda.Stream()

    return context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs


def save_tensor_as_raw_image(image_tensor: torch.Tensor,
                             file_location: str):
    """
    The function converts torch tensor the raw file and save it.
    The output dimension is in the form of [width, height, channel]

    :param image_tensor:
    :param file_location:
    :return:
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError("torch Tensor format is expected")

    image_numpy = to_numpy(image_tensor)
    processed_image = image_numpy.squeeze().transpose(1, 2, 0).astype('float32')
    processed_image.tofile(file_location)


def read_raw_image(image_directory: str,
                   image_file_name: str,
                   image_size: List[int]) -> np.ndarray:
    """
    The functions read the raw image in float32 format.
    :param image_directory: The image directory in string format
    :param image_file_name: The image file in the given image directory in string format
    :param image_size: The image size in [width, height] format
    :return: return the read raw image in numpy format
    """

    image_file = os.path.join(image_directory, image_file_name)
    raw_image = np.fromfile(image_file, dtype=np.float32)
    image_dimension = image_size + [3]
    image_numpy = raw_image.reshape(image_dimension)
    return image_numpy


def check_tensorrt_model(image: torch.Tensor,
                         tensorrt_file: str,
                         output_names: List[str],
                         outputs: Optional[Dict[str, torch.Tensor]] = None,
                         number_of_classes: Optional[int] = None,
                         ) -> Dict[str, np.ndarray]:
    """
    The functions initialize the tensorrt model, and implements the inference in tensorrt for a given image.
    The comparison for the given output can be implemented with the given output if it is given.
    For the time being, rtol and atol values are constant which are 1e-02
    If optional outputs are not given, the number of classes should be given

    :param image: expected as torch tensor
    :param tensorrt_file: expected as string, indicates the tensorrt file with trt extensiom
    :param output_names: The output names of the output nodes
    :param outputs: if outputs are given, the error analysis are implemented with the tensorrt outs,
        optinal with the, number of classes input
    :param number_of_classes: if outputs are not given, number of classes should be given
    :return: output of tensorrt model in dictionary format
    """

    assert number_of_classes or outputs, "outputs or number of classes should be entered"

    context, bindings, device_input, device_tensorrt_outs, stream, host_tensorrt_outs = \
        initialize_tensorrt_model(tensorrt_file=tensorrt_file,
                                  image=image,
                                  output_names=output_names,
                                  outputs=outputs,
                                  number_of_classes=number_of_classes)

    tensorrt_inference(device_input=device_input,
                       device_tensorrt_outs=device_tensorrt_outs,
                       image=image,
                       stream=stream,
                       context=context,
                       bindings=bindings,
                       host_tensorrt_outs=host_tensorrt_outs)

    if outputs:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(outputs[output_key]),
                                    host_tensorrt_outs[output_key], rtol=1e-02, atol=1e-02)
         for output_key in host_tensorrt_outs.keys()]
        print("== Done ==")

    return host_tensorrt_outs


def read_raw_file(model_dir: str,
                  size: List[int],
                  number_of_classes: int,
                  output_names: List[str],
                  result_idx: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """
    The function reads the raw output format of SNPE
    in the form of model_dir/Result_<result_idx>/<output_name>.raw

    :param model_dir: the directory of the raw file in string format
    :param size: The size of the raw image in [width, height] format
    :param number_of_classes: the number of classes of the output raw file or the channel size
    :param output_names: The output names of the output nodes of the raw file
    :param result_idx: The result idx of the SNPE result
    :return: Return the output which is read from raw file in dictionary format
    """

    raw_result_dict = {}
    reshape_dimension = [1] + size + [number_of_classes]
    for name in output_names:
        result_file = os.path.join(model_dir, f'Result_{result_idx}', f'{name}.raw')
        raw_result = np.fromfile(result_file, dtype=np.float32)
        raw_result = raw_result.reshape(reshape_dimension)
        raw_result = raw_result.transpose(0, 3, 1, 2)
        raw_result_dict.update({name: raw_result})

    return raw_result_dict


def read_raw_file_object_detection(model_dir: str,
                                   number_of_boxes: int,
                                   number_of_classes: int,
                                   output_names: List[str],
                                   result_idx: Optional[int] = 0) -> Dict[str, np.ndarray]:
    """
    The function reads the raw output format of SNPE
    in the form of model_dir/Result_<result_idx>/<output_name>.raw

    :param model_dir: the directory of the raw file in string format
    :param size: The size of the raw image in [width, height] format
    :param number_of_classes: the number of classes of the output raw file or the channel size
    :param output_names: The output names of the output nodes of the raw file
    :param result_idx: The result idx of the SNPE result
    :return: Return the output which is read from raw file in dictionary format
    """

    raw_result_dict = {}
    reshape_dimension = [1] + [number_of_boxes] + [-1]
    for name in output_names:
        result_file = os.path.join(model_dir, f'Result_{result_idx}', f'{name}.raw')
        raw_result = np.fromfile(result_file, dtype=np.float32)
        raw_result = raw_result.reshape(reshape_dimension)
        # raw_result = raw_result.transpose(0, 3, 1, 2)
        raw_result_dict.update({name: raw_result})

    return raw_result_dict


def save_onnx_model(model: torch.nn.Module,
                    inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
                    outputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
                    onnx_file: str,
                    opset_version: int,
                    input_names: Optional[List[str]] = None,
                    output_names: Optional[List[str]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    The function converts the given torch model to the onnx format. An example input and output torch tensor are also required for the conversion
    :param model:
    :param inputs:
    :param outputs:
    :param onnx_file:
    :param opset_version: The latest opset version is 13, The SNPE supports up to opset version 9.
    :param input_names:
    :param output_names:
    :return: The flattened version of inputs and outputs are also returned. The outputs can optionally be utilized ort_inference function.
    """

    inputs_flatten = my_flatten(inputs)
    outputs_flatten = my_flatten(outputs)

    if input_names is None:
        input_names = []
        for i, _ in enumerate(inputs_flatten):
            input_names.append('input' + str(i + 1))
    else:
        np.testing.assert_equal(len(input_names), len(inputs_flatten),
                                "Number of input names provided is not equal to the number of inputs.")

    if output_names is None:
        output_names = []
        for i, _ in enumerate(outputs_flatten):
            output_names.append('output' + str(i + 1))
    else:
        np.testing.assert_equal(len(output_names), len(outputs_flatten),
                                "Number of output names provided is not equal to the number of output.")

    if not os.path.isfile(onnx_file):
        torch.onnx.export(model, inputs, onnx_file,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names,
                          example_outputs=outputs,
                          enable_onnx_checker=False,
                          do_constant_folding=True,
                          opset_version=opset_version)

    return inputs_flatten, outputs_flatten


# TODO, for the time being, it is commented out. Jetson does not have onnruntime package. Maybe we need to replace it with onnxruntime-gpu
def ort_inference(model_file: str,
                  inputs_flatten: List[torch.Tensor],
                  outputs_flatten: Optional[List[torch.Tensor]] = None) -> List[np.ndarray]:
    """
    The function controls the onnx outputs for the given inputs in the list format with the outputs given in the list format
    For the time being, the rtol and atol is fixed to 1e-02 and 1e-03, respectively
    :param model_file: The onnx model file in string format
    :param inputs_flatten: The input list
    :param outputs_flatten: The output list which will be used as the baseline
    :return:
    """

    print("====== ORT Inference ======")
    ort_sess = onnxruntime.InferenceSession(model_file)
    ort_inputs = dict(
        (ort_sess.get_inputs()[i].name,
         to_numpy(input_flatten)
         )
        for i, input_flatten in enumerate(inputs_flatten)
    )

    ort_outs = ort_sess.run(None, ort_inputs)
    if outputs_flatten is not None:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(output),
                                    ort_outs[i], rtol=1e-02, atol=1e-03)
         for i, output in enumerate(outputs_flatten)]
    print("== Done ==")
    return ort_outs


def print_size_of_model(trt_file):
    print('Size (MB):', os.path.getsize(trt_file) / 1e6)


def collect_stats(model, data_loader, device, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.to(device))
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, torch_device, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.to(torch_device)


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
