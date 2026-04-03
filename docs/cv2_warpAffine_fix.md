# Fix for OpenCV `warpAffine` Error in Cephalometric DataLoader

## The Problem
When running the training script (`train.py`) for the Cephalometric dataset, the DataLoader worker processes were crashing with the following error:

```text
cv2.error: Caught error in DataLoader worker process 0.
Original Traceback (most recent call last):
...
  File "/.../ceph/HRNet-Ceph-Landmark-Detection/lib/utils/transforms.py", line 212, in crop
    new_img = cv2.warpAffine(new_img, rot)
cv2.error: OpenCV(4.13.0) :-1: error: (-5:Bad argument) in function 'warpAffine'
> Overload resolution failed:
```

## Root Cause
The crash was occurring in `lib/utils/transforms.py` inside the `crop()` function. When data augmentation randomly selected a non-zero rotation angle (`rot`), the function attempted to rotate the cropped image by trying to execute:

```python
new_img = cv2.warpAffine(new_img, rot)
```

The bug here is that `cv2.warpAffine` mathematically expects a $2 \times 3$ affine transformation matrix as its second parameter, but the code was passing `rot`, which is just a single scalar number (the angle in degrees). 

## The Fix
To properly rotate the image, we first need to build a rotation matrix from the specific angle using OpenCV's `getRotationMatrix2D` method. Once we have the matrix, we can cleanly pass it to `warpAffine`.

Changed the logic in `lib/utils/transforms.py`:

**Before:**
```python
    if not rot == 0:
        # Remove padding
        new_img = cv2.warpAffine(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
```

**After:**
```python
    if not rot == 0:
        # Remove padding
        rot_mat = cv2.getRotationMatrix2D((new_img.shape[1] / 2.0, new_img.shape[0] / 2.0), rot, 1.0)
        new_img = cv2.warpAffine(new_img, rot_mat, (new_img.shape[1], new_img.shape[0]))
        new_img = new_img[pad:-pad, pad:-pad]
```

This properly generates the $2 \times 3$ rotation matrix centered precisely in the newly padded bounding box, applies the correct rotation, and then trims away the padding to produce the final valid crop for the HRNet model.
