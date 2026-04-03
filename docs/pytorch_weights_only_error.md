# PyTorch 2.6+ `UnpicklingError` when Loading Models

## The Problem
When running `test.py` (or any script that loads a saved `.pth` file), you might encounter the following error:

```text
Traceback (most recent call last):
...
  File "torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`...
        WeightsUnpickler error: Unsupported global: GLOBAL lib.models.hrnet.HighResolutionNet was not an allowed global by default...
```

## Root Cause
This is a security feature introduced in **PyTorch 2.6**. 

Historically, PyTorch model files (`.pth` or `.pt`) were saved using Python's `pickle` module. This means they can contain not just numbers (network weights), but also arbitrary Python code and class definitions (like `HighResolutionNet`). Since malicious actors could hide viruses in downloaded models, PyTorch changed their default behavior to `weights_only=True`. 

When `torch.load` runs with `weights_only=True`, it strictly expects numbers. If it detects custom class structures in your `.pth` file, it panics and blocks the load to protect your system.

## The Fix
Since you trained this model yourself and know the file is safe, you need to explicitly tell PyTorch to allow loading the full pickled object.

You bypass the security block by adding `weights_only=False` as a parameter to `torch.load()`.

**Before:**
```python
    # load model
    state_dict = torch.load(args.model_file)
```

**After:**
```python
    # load model
    state_dict = torch.load(args.model_file, weights_only=False)
```

### Note for the Future:
If you ever download a model checkpoint from an untrusted source on the internet, you should be very careful when using `weights_only=False`. For files you generated on your own machine, it is entirely safe.

---

## Secondary Issue: Full Model Instance vs. `state_dict`
After fixing the security issue, you might immediately hit another error:

```text
  File ".../tools/test.py", line 61, in main
    if 'state_dict' in state_dict.keys():
                       ^^^^^^^^^^^^^^^
AttributeError: 'HighResolutionNet' object has no attribute 'keys'
```

### Root Cause
This happens if the training script (`train.py`) saved the **entire PyTorch model object** (`HighResolutionNet` class) directly into the `.pth` file, instead of just saving the raw dictionary of numbers (`state_dict`). 

Because testing scripts generally expect a standard dictionary, the code attempts to call `.keys()`. Since the loaded object is actually a full neural network framework, not a dictionary, Python throws an `AttributeError`.

### The Fix
We update the test script to intelligently intercept the loaded data. If it detects a full `torch.nn.Module`, it extracts the dictionary automatically. If it detects a dictionary, it natively passes it through:

**In `tools/test.py`:**
```python
    # load model
    loaded_obj = torch.load(args.model_file, weights_only=False)
    
    # Check if the loaded object is a full model instance
    if isinstance(loaded_obj, torch.nn.Module):
        # Extract the dictionary format
        state_dict = loaded_obj.state_dict()
    else:
        # It's already a dictionary
        state_dict = loaded_obj

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)
```
