import os
import json
import onnxruntime
from collections.abc import Mapping, Sequence

def load_meta_data(raw: Mapping) -> dict:
    """
    Recursively walk a Mapping, parsing any JSON‐encoded strings
    and turning nested mappings/lists into pure Python structures.
    """
    loaded = {}
    for k, v in raw.items():
        # 1) If it's a JSON string, try to parse it
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            loaded[k] = v

        # 2) If it's a mapping, recurse
        elif isinstance(v, Mapping):
            loaded[k] = load_meta_data(v)

        # 3) If it's a non‐string sequence, recurse into items
        elif isinstance(v, Sequence):
            new_list = []
            for item in v:
                if isinstance(item, Mapping):
                    new_list.append(load_meta_data(item))
                else:
                    new_list.append(item)
            loaded[k] = new_list

        # 4) Anything else, keep as is
        else:
            loaded[k] = v

    return loaded

def print_meta(d: Mapping, indent: int = 0, verbose: bool = False):
    all_keys = list(d.keys())
    # split into visible vs hidden
    visible_keys = [k for k in all_keys if not k.startswith('.')]
    hidden_keys  = [k for k in all_keys if k.startswith('.')] if verbose else []

    # decide print order: visible, then (if verbose) hidden
    key_groups = [visible_keys]
    if verbose:
        key_groups.append(hidden_keys)

    if not any(key_groups):
        return

    # compute padding across all keys we will print
    printed_keys = visible_keys + hidden_keys
    key_width = max(len(k) for k in printed_keys)

    def fmt(k): return f"{k}:".ljust(key_width + 1)

    # process each group in order
    for keys in key_groups:
        if not keys:
            continue

        # classify keys in this group
        simple_keys = []
        list_keys   = []
        dict_keys   = []
        for k in keys:
            v = d[k]
            if isinstance(v, Mapping):
                dict_keys.append(k)
            elif isinstance(v, Sequence) and not isinstance(v, str):
                list_keys.append(k)
            else:
                simple_keys.append(k)

        simple_keys.sort()
        list_keys.sort()
        dict_keys.sort()

        for k in simple_keys:
            v = d[k]
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if not isinstance(parsed, (Mapping, Sequence)):
                        v = parsed
                except json.JSONDecodeError:
                    pass
            print("  " * indent + fmt(k), v)

        for k in list_keys:
            v = d[k]
            print("  " * indent + fmt(k))
            for item in v:
                if isinstance(item, Mapping):
                    print_meta(item, indent + 1, verbose)
                else:
                    print("  " * (indent + 1) + f"- {item}")

        for k in dict_keys:
            print("  " * indent + fmt(k))
            print_meta(d[k], indent + 1, verbose)

class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False):
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )
        self.meta = self.ort_session.get_modelmeta()
        meta_data = self.meta.custom_metadata_map
        self.input_name = input_name
        self.awd = awd

        meta = {
            "Model":         os.path.basename(onnx_model_path),
            "Producer name": self.meta.producer_name,
            "Domain":        self.meta.domain,
            "Description":   self.meta.description,
            "Graph name":    self.meta.graph_name,
        }
        meta.update(meta_data)
        info = load_meta_data(meta)
        if not meta_data:
            return
        self.info = info

        self.action_offset = info['.ActionOffset']
        self.initial_action_pose = info['.InitialActionPose']
        self.joints = info['.Joints']
        self.contacts = info.get('.Contacts', False)
        self.quat = info.get('.Quat', True)

        config = info['.Config']
        self.sim_dt = config['sim_dt']
        self.ctrl_dt = config['ctrl_dt']
        self.action_scale = config['action_scale']
        self.dof_vel_scale = config['dof_vel_scale']
        self.COMMANDS_RANGE_X = config['lin_vel_x']
        self.COMMANDS_RANGE_Y = config['lin_vel_y']
        self.COMMANDS_RANGE_THETA = config['ang_vel_yaw']
        push_config = config['push_config']
        self.noise_config = config['noise_config']
        self.push_enable   = bool(push_config['enable'])
        self.push_mag_min, self.push_mag_max = map(float, push_config['magnitude_range'])
        self.push_int_min, self.push_int_max = map(float, push_config['interval_range'])
        raw = info['.nb_steps_in_period']
        if raw is None:
            raise ValueError("Expected 'nb_steps_in_period' in meta data")
        try:
            self.nb_steps_in_period = int(raw)
        except ValueError:
            raise ValueError(f"Expected integer metadata for 'nb_steps_in_period', got: {raw!r}")

        print_meta(info, 0)

    def infer(self, inputs):
        if self.awd:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]

