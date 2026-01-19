# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ultralytics import __version__
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    FLOAT_OR_INT,
    IS_VSCODE,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    STR_OR_PATH,
    TESTS_RUNNING,
    YAML,
    IterableSimpleNamespace,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
)

# Define valid tasks and modes
MODES = frozenset({"train", "val", "predict", "export", "track", "benchmark"})
TASKS = frozenset({"detect", "obb"})
TASK2DATA = {
    "detect": "coco8.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "obb": "metrics/mAP50-95(B)",
}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
CLI_HELP_MSG = f"""
    Arguments received: {["yolo", *ARGV[1:]]!s}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {list(TASKS)}
                MODE (required) is one of {list(MODES)}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. Predict an OBB model on a YouTube video at image size 320:
        yolo predict model=yolo11n-obb.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Validate a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLO11n detection model to ONNX format at image size 640:
        yolo export model=yolo11n.pt format=onnx imgsz=640

    5. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = frozenset(
    {  # integer or float arguments, i.e. x=2 and x=2.0
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "degrees",
        "shear",
        "time",
        "workspace",
        "batch",
    }
)
CFG_FRACTION_KEYS = frozenset(
    {  # fractional float arguments with 0.0<=values<=1.0
        "dropout",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "conf",
        "iou",
        "fraction",
    }
)
CFG_INT_KEYS = frozenset(
    {  # integer-only arguments
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
    }
)
CFG_BOOL_KEYS = frozenset(
    {  # boolean-only arguments
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "half",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "multi_scale",
    }
)


def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    """Convert a configuration object to a dictionary.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted. Can be a file path, a string, a
            dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace:
    """Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = str(cfg.get("model", "")).partition(".")[0]
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg: dict, hard: bool = True) -> None:
    """Check configuration argument types and values for the Ultralytics library.

    This function validates the types and values of configuration arguments, ensuring correctness and converting them if
    necessary. It checks for specific key types defined in global variables such as `CFG_FLOAT_KEYS`,
    `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`.

    Args:
        cfg (dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    """Return the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task', 'mode',
            and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name' or the
            'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        runs = (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        nested = args.project and len(Path(args.project).parts) > 1  # e.g. "user/project" or "org\repo"
        project = runs / args.project if nested else args.project or runs
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True, mkdir=True)

    return Path(save_dir).resolve()  # resolve to display full path in console


def _handle_deprecation(custom: dict) -> dict:
    """Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (dict): Configuration dictionary potentially containing deprecated keys.

    Returns:
        (dict): Updated configuration dictionary with deprecated keys replaced.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom


def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None:
    """Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (dict): The base configuration dictionary containing valid keys.
        custom (dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.
        allowed_custom_keys (set | None): Optional set of additional keys that are allowed in the custom dictionary.

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    # Allow 'augmentations' as a valid custom parameter for custom Albumentations transforms
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: list[str]) -> list[str]:
    """Merge arguments around isolated '=' in a list of strings and join fragments with brackets.

    This function handles the following cases:
        1. ['arg', '=', 'val'] becomes ['arg=val']
        2. ['arg=', 'val'] becomes ['arg=val']
        3. ['arg', '=val'] becomes ['arg=val']
        4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']

    Args:
        args (list[str]): A list of strings where each element represents an argument or fragment.

    Returns:
        (list[str]): A list of strings where the arguments around isolated '=' are merged and fragments with brackets
            are joined.

    Examples:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_equals_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle equals sign merging
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # Handle bracket joining
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # Append any remaining current string
    if current:
        new_args.append(current)

    return new_args


def handle_yolo_settings(args: list[str]) -> None:
    """Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset and updating individual settings. It should be
    called when executing a script with arguments related to YOLO settings management.

    Args:
        args (list[str]): A list of command line arguments for YOLO settings management.

    Examples:
        >>> handle_yolo_settings(["reset"])  # Reset YOLO settings
        >>> handle_yolo_settings(["default_cfg_path=yolo11n.yaml"])  # Update a specific setting

    Notes:
        - If no arguments are provided, the function will display the current settings.
        - The 'reset' command will delete the existing settings file and create new default settings.
        - Other arguments are treated as key-value pairs to update specific settings.
        - The function will check for alignment between the provided settings and the existing ones.
        - After processing, the updated settings will be displayed.
        - For more information on handling YOLO settings, visit:
          https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for k, v in new.items():
                    LOGGER.info(f"✅ Updated '{k}={v}'")

        LOGGER.info(SETTINGS)  # print the current settings
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"settings error: '{e}'. Please see {url} for help.")


def parse_key_value_pair(pair: str = "key=value") -> tuple:
    """Parse a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        key (str): The parsed key.
        value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo11n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo11n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v: str) -> Any:
    """Convert a string representation of a value to its appropriate Python type.

    This function attempts to convert a given string into a Python object of the most appropriate type. It handles
    conversions to None, bool, int, float, and other types that can be evaluated safely.

    Args:
        v (str): The string representation of the value to be converted.

    Returns:
        (Any): The converted value. The type can be None, bool, int, float, or the original string if no conversion is
            applicable.

    Examples:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'

    Notes:
        - The function uses a case-insensitive comparison for boolean and None values.
        - For other types, it attempts to use Python's ast.literal_eval() function for safe evaluation.
        - If no conversion is possible, the original string is returned.
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            return v


def entrypoint(debug: str = "") -> None:
    """Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and executing
    the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained OBB model at image size 320:
        >>> entrypoint("obb predict model=yolo11n-obb.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          Ultralytics documentation at https://docs.ultralytics.com.
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "copy-cfg": copy_default_cfg,
        "help": lambda: LOGGER.info(CLI_HELP_MSG),  # help below hub for -h flag precedence
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    from ultralytics import YOLO

    model = YOLO(model, task=task)
    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    # Recommend VS Code extension
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg() -> None:
    """Copy the default configuration file and create a new one with '_copy' appended to its name.

    This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it with '_copy'
    appended to its name in the current working directory. It provides a convenient way to create a custom configuration
    file based on the default settings.

    Examples:
        >>> copy_default_cfg()
        # Output: default.yaml copied to /path/to/current/directory/default_copy.yaml
        # Example YOLO command with this new custom cfg:
        #   yolo cfg='/path/to/current/directory/default_copy.yaml' imgsz=320 batch=8

    Notes:
        - The new configuration file is created in the current working directory.
        - After copying, the function prints a message with the new file's location and an example
          YOLO command demonstrating how to use the new configuration file.
        - This function is useful for users who want to modify the default configuration without
          altering the original file.
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolo11n.pt')
    entrypoint(debug="")
