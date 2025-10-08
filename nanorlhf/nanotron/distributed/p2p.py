import torch
import torch.distributed as dist

from nanorlhf.nanotron.distributed.mpu import MPU, ParallelMode

NoneType = type(None)

TORCH_ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

TORCH_DTYPE_TO_ID = {
    dtype: idx for idx, dtype in enumerate(TORCH_ID_TO_DTYPE)
}

ID_TO_DTYPE = [
    bool,
    int,
    float,
    complex,
    str,
    type,
    list,
    tuple,
    set,
    dict,
    NoneType,
    torch.Size,
    torch.Tensor,
]

DTYPE_TO_ID = {
    dtype: idx for idx, dtype in enumerate(ID_TO_DTYPE)
}


def current_device():
    return torch.device(torch.cuda.current_device())


class P2P:
    def __init__(self):
        self.INSTRUCTIONS = {
            bool: {"send": self._send_bool, "recv": self._recv_bool},
            int: {"send": self._send_int, "recv": self._recv_int},
            float: {"send": self._send_float, "recv": self._recv_float},
            complex: {"send": self._send_complex, "recv": self._recv_complex},
            str: {"send": self._send_str, "recv": self._recv_str},
            type: {"send": self._send_type, "recv": self._recv_type},
            list: {"send": self._send_list, "recv": self._recv_list},
            tuple: {"send": self._send_tuple, "recv": self._recv_tuple},
            set: {"send": self._send_set, "recv": self._recv_set},
            dict: {"send": self._send_dict, "recv": self._recv_dict},
            NoneType: {"send": self._send_none, "recv": self._recv_none},
            torch.Size: {"send": self._send_size, "recv": self._recv_size},
            torch.Tensor: {"send": self._send_tensor, "recv": self._recv_tensor},
        }

    @staticmethod
    def _send_type(
        data: type,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send the type of the data to the destination rank.

        Args:
            data (type): The type of the data to send.
            dst_rank (int): The destination rank to send the data type to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, type), f"Wrong type: {data} must be {type} type."
        assert send_type is False, "to send `type`, you don't need to send type."

        group = mpu.get_group(parallel_mode)
        data = torch.tensor([DTYPE_TO_ID[data]], dtype=torch.long, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_type(
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> type:
        """
        Receive the type of the data from the source rank.

        Args:
            src_rank (int): The source rank to receive the data type from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            type: The received type of the data.
        """
        group = mpu.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(data, src=src_rank, group=group)
        return ID_TO_DTYPE[data.item()]

    def _send_none(
        self,
        data: NoneType,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send nothing, just assert data is None.

        Args:
            data (NoneType): The data to send, must be None.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, NoneType), f"Wrong type: {data} must be {NoneType}."
        if send_type:
            self._send_type(NoneType, dst_rank, mpu, parallel_mode, send_type)

    def _recv_none(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> NoneType:
        """
        Just return None.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            NoneType: None.
        """
        return None

    def _send_str(
        self,
        data: str,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a string to the destination rank.

        Args:
            data (str): The string to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, str), f"Wrong type: {data} must be {str}."

        if send_type is True:
            self._send_type(str, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)
        length = torch.tensor([len(data)], dtype=torch.long, device=current_device())
        dist.send(length, dst=dst_rank, group=group)

        data = torch.tensor([ord(s) for s in data], dtype=torch.long, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    def _recv_str(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> str:
        """
        Receive a string from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            str: The received string.
        """
        group = mpu.get_group(parallel_mode)
        length = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(length, src=src_rank, group=group)
        data = torch.tensor([0] * length.item(), dtype=torch.long, device=current_device())
        dist.recv(data, src=src_rank, group=group)
        return "".join([chr(i) for i in data.tolist()])

    def _send_bool(
        self,
        data: bool,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a boolean value to the destination rank.

        Args:
            data (bool): The boolean value to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, bool), f"Wrong type: {data} must be {bool}."

        if send_type is True:
            self._send_type(bool, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)
        data = torch.tensor([1 if data else 0], dtype=torch.long, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    def _recv_bool(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> bool:
        """
        Receive a boolean value from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            bool: The received boolean value.
        """
        group = mpu.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(data, src=src_rank, group=group)

        if data == 0:
            return False
        elif data == 1:
            return True
        else:
            raise ValueError(
                f"Wrong value for boolean. only 0 or 1 can be supported. "
                f"but your input is {data}."
            )

    def _send_int(
        self,
        data: int,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send an integer to the destination rank.

        Args:
            data (int): The integer to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, int), f"Wrong type: {data} must be {int}."

        if send_type is True:
            self._send_type(int, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)
        data = torch.tensor([data], dtype=torch.long, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    def _recv_int(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> int:
        """
        Receive an integer from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            int: The received integer.
        """
        group = mpu.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(data, src=src_rank, group=group)
        return data.item()

    def _send_float(
        self,
        data: float,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a float to the destination rank.

        Args:
            data (float): The float to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, float), f"Wrong type: {data} must be {float}."

        if send_type is True:
            self._send_type(float, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)
        data = torch.tensor([data], dtype=torch.float32, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    def _recv_float(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> float:
        """
        Receive a float from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            float: The received float.
        """
        group = mpu.get_group(parallel_mode)
        data = torch.tensor([0.0], dtype=torch.float32, device=current_device())
        dist.recv(data, src=src_rank, group=group)
        return data.item()

    def _send_complex(
        self,
        data: complex,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a complex number to the destination rank.

        Args:
            data (complex): The complex number to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, complex), f"Wrong type: {data} must be {complex}."

        if send_type is True:
            self._send_type(complex, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)
        data = torch.tensor([data.real, data.imag], dtype=torch.float32, device=current_device())
        dist.send(data, dst=dst_rank, group=group)

    def _recv_complex(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> complex:
        """
        Receive a complex number from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            complex: The received complex number.
        """
        group = mpu.get_group(parallel_mode)
        data = torch.tensor([0.0, 0.0], dtype=torch.float32, device=current_device())
        dist.recv(data, src=src_rank, group=group)
        return complex(data[0].item(), data[1].item())

    def _send_tensor(
        self,
        data: torch.Tensor,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a tensor to the destination rank.

        Args:
            data (torch.Tensor): The tensor to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, torch.Tensor), f"Wrong type: {data} must be {torch.Tensor}."

        if send_type is True:
            self._send_type(torch.Tensor, dst_rank=dst_rank, mpu=mpu, parallel_mode=parallel_mode)

        group = mpu.get_group(parallel_mode)

        dtype = torch.tensor(
            TORCH_DTYPE_TO_ID[data.dtype], dtype=torch.long, device=current_device()
        )
        dist.send(dtype, dst=dst_rank, group=group)

        requires_grad = torch.tensor(
            1 if data.requires_grad else 0, dtype=torch.long, device=current_device()
        )
        dist.send(requires_grad, dst=dst_rank, group=group)

        dims = torch.tensor(len(data.size()), dtype=torch.long, device=current_device())
        dist.send(dims, dst=dst_rank, group=group)

        shape = torch.tensor(list(data.size()), dtype=torch.long, device=current_device())
        dist.send(shape, dst=dst_rank, group=group)

        if not data.is_contiguous():
            data = data.contiguous()

        dist.send(data, dst=dst_rank, group=group)

    def _recv_tensor(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> torch.Tensor:
        """
        Receive a tensor from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            torch.Tensor: The received tensor.
        """
        group = mpu.get_group(parallel_mode)

        dtype = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(dtype, src=src_rank, group=group)
        dtype = TORCH_ID_TO_DTYPE[dtype]

        requires_grad = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(requires_grad, src=src_rank, group=group)
        requires_grad = True if requires_grad.item() == 1 else False

        dims = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(dims, src=src_rank, group=group)
        dims = dims.item()

        shape = torch.tensor([0] * dims, dtype=torch.long, device=current_device())
        dist.recv(shape, src=src_rank, group=group)
        shape = tuple(shape.tolist())

        data = torch.zeros(size=shape, dtype=dtype, device=current_device())
        data.requires_grad = requires_grad and data.is_floating_point()
        dist.recv(data, src=src_rank, group=group)
        return data

    def _send_list(
        self,
        data: list,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a list to the destination rank.

        Args:
            data (list): The list to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, list), f"wrong type: {data} must be {list} type."

        if send_type is True:
            self._send_type(list, mpu=mpu, dst_rank=dst_rank)

        len_list = len(data)

        self._send_int(
            len_list,
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )

        for item in data:
            _type = type(item)
            assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"
            self.INSTRUCTIONS[_type]["send"](
                item,
                dst_rank=dst_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
                send_type=True,
            )

    def _recv_list(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> list:
        """
        Receive a list from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            list: The received list.
        """
        output_list = []

        len_list = self._recv_int(
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )

        for _ in range(len_list):
            _type = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )

            assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"

            _item = self.INSTRUCTIONS[_type]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )
            output_list.append(_item)

        return output_list

    def _send_set(
        self,
        data: set,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a set to the destination rank.

        Args:
            data (set): The set to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, set), f"wrong type: {data} must be {set} type."

        if send_type is True:
            self._send_type(set, mpu=mpu, dst_rank=dst_rank)

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_set(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> set:
        """
        Receive a set from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            set: The received set.
        """
        output_list = self._recv_list(
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )
        return set(output_list)

    def _send_tuple(
        self,
        data: tuple,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a tuple to the destination rank.

        Args:
            data (tuple): The tuple to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, tuple), f"wrong type: {data} must be {tuple} type."

        if send_type is True:
            self._send_type(tuple, mpu=mpu, dst_rank=dst_rank)

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_tuple(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> tuple:
        """
        Receive a tuple from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            tuple: The received tuple.
        """
        output_list = self._recv_list(
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )
        return tuple(output_list)

    def _send_size(
        self,
        data: torch.Size,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a torch.Size to the destination rank.

        Args:
            data (torch.Size): The torch.Size to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, torch.Size), f"wrong type: {data} must be {torch.Size} type."

        if send_type is True:
            self._send_type(torch.Size, mpu=mpu, dst_rank=dst_rank)

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_size(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> torch.Size:
        """
        Receive a torch.Size from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            torch.Size: The received torch.Size.
        """
        output_list = self._recv_list(
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )
        return torch.Size(output_list)

    def _send_dict(
        self,
        data: dict,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        """
        Send a dictionary to the destination rank.

        Args:
            data (dict): The dictionary to send.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, dict), f"wrong type: {data} must be {dict} type."

        if send_type is True:
            self._send_type(dict, mpu=mpu, dst_rank=dst_rank)

        len_dict = len(data)

        self._send_int(
            len_dict,
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )

        for key, val in data.items():
            _type_key, _type_val = type(key), type(val)
            assert _type_key in ID_TO_DTYPE, f"unsupported type: {_type_key}"
            assert _type_val in ID_TO_DTYPE, f"unsupported type: {_type_val}"
            self.INSTRUCTIONS[_type_key]["send"](
                key,
                dst_rank=dst_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
                send_type=True,
            )
            self.INSTRUCTIONS[_type_val]["send"](
                val,
                dst_rank=dst_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
                send_type=True,
            )

    def _recv_dict(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ) -> dict:
        """
        Receive a dictionary from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            dict: The received dictionary.
        """
        output_dict = {}

        len_dict = self._recv_int(
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )

        for _ in range(len_dict):
            _type_key = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )
            assert _type_key in ID_TO_DTYPE, f"unsupported type: {_type_key}"
            _key = self.INSTRUCTIONS[_type_key]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )

            _type_val = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )
            assert _type_val in ID_TO_DTYPE, f"unsupported type: {_type_val}"
            _val = self.INSTRUCTIONS[_type_val]["recv"](
                src_rank=src_rank,
                mpu=mpu,
                parallel_mode=parallel_mode,
            )

            output_dict[_key] = _val

        return output_dict

    def send(
        self,
        data,
        dst_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        """
        Send data to the destination rank.

        Args:
            data: The data to send. Supported types are bool, int, float, complex, str, type,
                  list, tuple, set, dict, NoneType, torch.Size, torch.Tensor.
            dst_rank (int): The destination rank to send the data to.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.
        """
        _type = type(data)
        assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"
        self.INSTRUCTIONS[_type]["send"](
            data,
            dst_rank=dst_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
            send_type=True,
        )

    def recv(
        self,
        src_rank: int,
        mpu: MPU,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        """
        Receive data from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
            mpu (MPU): The model parallel unit handling the communication.
            parallel_mode (ParallelMode): The parallel mode to use for communication.

        Returns:
            The received data. Supported types are bool, int, float, complex, str, type,
            list, tuple, set, dict, NoneType, torch.Size, torch.Tensor.
        """
        _type = self.INSTRUCTIONS[type]["recv"](
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )
        assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"
        return self.INSTRUCTIONS[_type]["recv"](
            src_rank=src_rank,
            mpu=mpu,
            parallel_mode=parallel_mode,
        )
