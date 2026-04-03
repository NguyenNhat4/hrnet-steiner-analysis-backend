
## Plan: Update HRNet for Cephalometric Data

To adapt the internal HRNet implementation (HRNet-Ceph-Landmark-Detection) to read from the structured dataset at `ceph/data`, we can bypass intermediate CSV/JSON generation and write a new dataset loader directly integrated into the HRNet system.

**Steps**
1. Implement a custom dataset class `CephDataset` inside HRNet's dataset module that reads directly from the `ceph/data` structure (`Cephalograms/` images and `Annotations/` JSON files). 
2. Register the `CephDataset` in the HRNet dataset factory so it can be invoked via the configuration file.
3. Build a new `yaml` experiment configuration file pointing to the new `CephDataset`.
4. Update the number of keypoints (`num_joints`) within the configuration file to accurately reflect the cephalogram annotations.
5. Provide a quick dataloader test script or run a subset of `tools/train.py` to ensure it generates the correct `target` (heatmaps) and `target_weight`.

**Relevant files**
- ceph/HRNet-Ceph-Landmark-Detection/lib/datasets/ceph.py — New dataset class porting the direct JSON parsing from your ResNet dataset code into HRNet's augmentations format (generating center, scale, and calling `generate_target`).
- __init__.py — Register the dataset here.
- ceph/HRNet-Ceph-Landmark-Detection/experiments/ceph/hrnet_ceph.yaml — New config to orchestrate the dataloader inputs.

**Verification**
1. Write a standalone test (`test_dataloader.py`) inside `tools` to iterate through the first batch of the new `CephDataset` and verify the output heatmaps and image shapes match HRNet's expected structure.
2. Initialize `python tools/train.py --cfg experiments/ceph/hrnet_ceph.yaml` and verify that the forward pass to HRNet executes and loss is correctly calculated for the first few epochs.

**Decisions**
- Direct integration with the raw `ceph/data` structure is chosen over creating a preprocessing script to maintain a single source of truth for the data.
- Flip augmentation will be disabled by default since side profiles are asymmetrical.

**Further Considerations**
1. How many keypoints exactly are currently annotated in the `ceph/data` JSONs? The `num_joints` configuration variable in HRNet must match this number.
2. The current HRNet sideprofile.py relies on generating a pseudo-center and scale for cropping. Should I calculate the crop bounding box dynamically based on the Min/Max landmarks in each JSON image, or do you prefer to feed the entire `512x512` resized image directly to the network without cropping?