# Understanding Distributed Cellpose: A Deep Dive

**Audience:** Python users learning advanced topics  
**Goal:** Understand how distributed cellpose segments large images and how to fix bugs for 2D multi-channel data

---

## Table of Contents
1. [The Big Picture: Why Distributed Processing?](#the-big-picture)
2. [Core Concepts](#core-concepts)
3. [The Workflow: Step by Step](#the-workflow)
4. [Key Functions Explained](#key-functions-explained)
5. [The Bugs We Found](#the-bugs-we-found)
6. [The Patches: How We Fixed Them](#the-patches)
7. [Putting It All Together](#putting-it-all-together)

---

## The Big Picture: Why Distributed Processing?

### The Problem
Imagine you have a massive microscopy image: 50,000 × 60,000 pixels. That's **3 billion pixels**! 

If you tried to segment this entire image at once:
- **Memory**: Would need 12+ GB just to load the image
- **GPU memory**: Most GPUs can't handle images this large
- **Time**: Would take hours or fail completely

### The Solution: Divide and Conquer
Instead of processing the whole image, we:
1. **Divide** the image into smaller overlapping tiles (blocks)
2. **Distribute** those tiles across workers (CPU cores/processes)
3. **Segment** each tile independently with Cellpose
4. **Merge** the results back together seamlessly

Think of it like a puzzle:
```
Original Image (too big):
┌─────────────────────────┐
│                         │
│    50,000 x 60,000      │
│                         │
└─────────────────────────┘

Divided into blocks:
┌─────┬─────┬─────┬─────┐
│ 1   │ 2   │ 3   │ 4   │
├─────┼─────┼─────┼─────┤
│ 5   │ 6   │ 7   │ 8   │
├─────┼─────┼─────┼─────┤
│ 9   │ 10  │ 11  │ 12  │
└─────┴─────┴─────┴─────┘
Each block: 2048 x 2048 pixels
```

---

## Core Concepts

### 1. Zarr Arrays
**What is it?** A file format for storing large arrays on disk in chunks.

```python
# Traditional numpy array (all in memory)
image = np.array([...])  # Entire image loaded!

# Zarr array (chunks on disk)
image = zarr.open('image.zarr', mode='r')
# Only loads data when you access it:
block = image[0:2048, 0:2048]  # Loads just this chunk
```

**Why use it?** You can work with images larger than your RAM.

### 2. Dask for Parallel Processing
**What is it?** A library that schedules tasks across multiple workers.

```python
# Without Dask (sequential)
for block in blocks:
    result = process(block)  # One at a time

# With Dask (parallel)
futures = client.map(process, blocks)  # All at once!
results = client.gather(futures)  # Collect when done
```

**Workers**: Separate Python processes that run tasks in parallel.

### 3. Overlapping Blocks
**Why overlap?** Cells at block edges need context.

```
Without overlap (bad):           With overlap (good):
┌─────┬─────┐                   ┌─────┬─────┐
│  ○  │○    │  Cell cut!        │  ○  │ ○   │  Cell intact!
│     │     │                   │    ╱│╲    │
└─────┴─────┘                   └───╱─┴─╲───┘
                                   overlap
```

We add extra pixels around each block, then trim them after segmentation.

### 4. Block Faces
**What are they?** The edges where blocks touch.

```
Block faces for block [1,1]:
      Top face
        ↓
    ┌─────┐
Left→│     │←Right
face │     │ face
    └─────┘
        ↑
    Bottom face
```

**Why important?** Cells that cross block boundaries need to be merged together.

---

## The Workflow: Step by Step

### Phase 1: Setup and Planning

**Step 1: Calculate block grid**
```python
# Image: (48320, 63993)
# Blocksize: (2048, 2048)

n_blocks_y = ceil(48320 / 2048) = 24 blocks
n_blocks_x = ceil(63993 / 2048) = 32 blocks
Total blocks: 24 × 32 = 768 blocks
```

**Step 2: Add overlap**
```python
# If diameter=30, overlap = 2×30 = 60 pixels
# Each block becomes: 2048 + 2×60 = 2168 pixels
# (extra pixels on all sides)
```

**Step 3: Create block crops (slices)**
```python
# Block [0,0] (top-left):
crop = (slice(0, 2108), slice(0, 2108))

# Block [1,0] (second row, first column):
crop = (slice(1988, 4156), slice(0, 2108))
#           ↑ starts earlier due to overlap
```

### Phase 2: Distributed Processing

**Step 4: Send blocks to workers**
```python
# Dask distributes these tasks:
futures = client.map(
    process_block,  # The function to run
    block_indices,  # [(0,0), (0,1), (1,0), ...]
    block_crops,    # [crop_00, crop_01, crop_10, ...]
    ...
)
```

**What happens in each worker:**
1. Read block from zarr file
2. Run Cellpose segmentation
3. Remove overlaps
4. Assign unique IDs to segments
5. Save to temporary zarr
6. Return block faces for merging

### Phase 3: Merging

**Step 5: Find segments that cross boundaries**
```
Block 1:        Block 2:
┌─────┐         ┌─────┐
│  ○──│────────→│──●  │  Same cell!
│     │  Face   │     │
└─────┘         └─────┘
 ID: 42          ID: 137

These need to be merged!
```

**Step 6: Create a merge graph**
```python
# Graph shows which IDs should merge:
42 ← → 137  (same cell, different IDs)
58 ← → 91   (another cell crossing boundary)
...
```

**Step 7: Relabel everything**
```python
# Find connected components:
# 42 and 137 → become ID 1
# 58 and 91 → become ID 2
# Relabel entire image with new IDs
```

---

## Key Functions Explained

### Function 1: `distributed_eval()`
**Purpose:** The main entry point - orchestrates everything.

**Simplified version:**
```python
def distributed_eval(input_zarr, blocksize, write_path, ...):
    # 1. Calculate block grid
    block_indices, block_crops = get_block_crops(...)
    
    # 2. Create temporary storage
    temp_zarr = zarr.open(...)
    
    # 3. Distribute blocks to workers
    futures = client.map(process_block, block_indices, block_crops, ...)
    results = client.gather(futures)
    
    # 4. Merge segments across boundaries
    faces, boxes, box_ids = zip(*results)
    new_labeling = determine_merge_relabeling(...)
    
    # 5. Apply relabeling and save final result
    relabeled = apply_relabeling(temp_zarr, new_labeling)
    save_to_zarr(relabeled, write_path)
    
    return final_segmentation, bounding_boxes
```

**Key parameters:**
- `input_zarr`: Your huge image (zarr array)
- `blocksize`: Size of tiles (e.g., `(2048, 2048)`)
- `write_path`: Where to save final result
- `cluster_kwargs`: How many workers, memory limits, etc.

---

### Function 2: `process_block()`
**Purpose:** Segment one block (called by each worker).

**What it does:**
```python
def process_block(block_index, crop, input_zarr, ...):
    # 1. Read image data for this block
    image = input_zarr[crop]  # e.g., input_zarr[0:2108, 0:2108]
    
    # 2. Run Cellpose
    model = cellpose.models.CellposeModel(...)
    segmentation = model.eval(image)[0]
    # Returns: 2D array of segment IDs
    # Example: [[0, 0, 1, 1],
    #           [0, 1, 1, 2],
    #           [3, 3, 2, 2]]
    
    # 3. Remove overlaps (trim extra pixels)
    segmentation = remove_overlaps(segmentation, ...)
    
    # 4. Make IDs globally unique
    # Block [0,0] gets IDs: 00001, 00002, 00003, ...
    # Block [0,1] gets IDs: 00101, 00102, 00103, ...
    # Block [1,0] gets IDs: 01001, 01002, 01003, ...
    segmentation, id_map = global_segment_ids(segmentation, block_index)
    
    # 5. Save to temporary storage
    output_zarr[crop] = segmentation
    
    # 6. Extract faces for merging
    faces = block_faces(segmentation)
    
    return faces, boxes, id_map
```

**Why unique IDs?** So we can track which segments came from which block during merging.

---

### Function 3: `remove_overlaps()`
**Purpose:** Trim the extra overlap pixels we added.

**Visual example:**
```python
# Block with overlap:           After trimming:
#  ┌──────────┐                 ┌────────┐
#  │ overlap→ │                 │        │
#  │ ↓        │                 │        │
#  │ ┌────────┤                 │        │
#  │ │  DATA  │  ← Remove       │  DATA  │
#  │ │        │     outer       │        │
#  │ │        │     pixels      │        │
#  └─┴────────┘                 └────────┘

# Code:
def remove_overlaps(array, crop, overlap, blocksize):
    # If not at edge, remove 'overlap' pixels from start
    if crop[0].start != 0:
        array = array[overlap:, :]  # Trim top
    
    # If block too big, trim to exact blocksize
    if array.shape[0] > blocksize[0]:
        array = array[:blocksize[0], :]
    
    # Same for all dimensions...
    return array, adjusted_crop
```

---

### Function 4: `global_segment_ids()`
**Purpose:** Make each segment ID unique across all blocks.

**How it works:**
```python
def global_segment_ids(segmentation, block_index, nblocks):
    # Block index: (2, 3) = row 2, column 3
    # Convert to flat index: 2*n_cols + 3 = let's say 23
    
    # Original IDs in this block: [0, 1, 2, 3, 4]
    # New IDs: [0, 23_00001, 23_00002, 23_00003, 23_00004]
    #           ↑            ↑
    #           0 stays 0    block#_segment#
    
    block_id = ravel_multi_index(block_index, nblocks)
    
    unique_ids = np.unique(segmentation)  # [0, 1, 2, 3, 4]
    new_ids = []
    
    for old_id in unique_ids:
        if old_id == 0:
            new_ids.append(0)  # Background stays 0
        else:
            # Pack: block_id (5 digits) + old_id (5 digits)
            new_id = int(f"{block_id}{old_id:05d}")
            new_ids.append(new_id)
    
    # Remap the entire segmentation
    # Where segmentation==1, replace with 23_00001, etc.
    return remapped_segmentation, new_ids
```

**Result:** Every segment in every block has a unique ID!

---

### Function 5: `block_faces()`
**Purpose:** Extract the edges of each block for merging.

**Visual example:**
```python
# 3D block:
#     ┌─────┐
#    /     /│  ← Right face (X+)
#   /  ●  / │  ← Front face (Y+)
#  ┌─────┐  │
#  │     │  /
#  │     │ /   ← Bottom face (Z+)
#  └─────┘/
# ↑ Top, Back, Left faces on other sides

def block_faces(segmentation):
    faces = []
    for axis in range(segmentation.ndim):
        # Get first slice along this axis
        faces.append(segmentation[axis, 0:1, :, :])
        # Get last slice along this axis  
        faces.append(segmentation[axis, -1:, :, :])
    return faces
```

For 2D: Returns [top, bottom, left, right]  
For 3D: Returns [front, back, top, bottom, left, right]

---

### Function 6: `determine_merge_relabeling()`
**Purpose:** Figure out which segments should merge together.

**Step-by-step:**

**1. Pair adjacent faces**
```python
# Block [0,0]'s RIGHT face touches Block [0,1]'s LEFT face
face_pair = concatenate([block_00_right, block_01_left])
```

**2. Find which labels touch**
```python
# In the concatenated face:
# [[42, 42],     ← block [0,0]
#  [137, 137]]   ← block [0,1]
# 
# Label 42 touches label 137!
# Create graph edge: 42 ← → 137
```

**3. Find connected components**
```python
# Graph might look like:
# 42 ← → 137 ← → 215  (all same cell)
# 58 ← → 91          (different cell)
# 
# Connected components:
# Group 1: {42, 137, 215}
# Group 2: {58, 91}
```

**4. Create new labeling**
```python
# All IDs in Group 1 → become ID 1
# All IDs in Group 2 → become ID 2
# Unused IDs → become ID 0
# 
# new_labeling[42] = 1
# new_labeling[137] = 1
# new_labeling[215] = 1
# new_labeling[58] = 2
# new_labeling[91] = 2
```

---

### Function 7: `block_face_adjacency_graph()`
**Purpose:** Build the merge graph from all face pairs.

**This is where the 2D bug was!**

```python
def block_face_adjacency_graph(faces, nlabels):
    all_mappings = []
    
    # BUG: This was hardcoded to 3!
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    
    for face in faces:
        # Shrink labels to avoid merging near boundaries
        # Find which labels touch using the structure
        mapped = _across_block_label_grouping(face, structure)
        all_mappings.append(mapped)
    
    # Build sparse matrix graph
    i, j = np.concatenate(all_mappings, axis=1)
    graph = coo_matrix((ones, (i, j)))
    return graph
```

**What `generate_binary_structure(ndim, connectivity)` does:**

For 2D with connectivity=1:
```
[[0, 1, 0],
 [1, 1, 1],
 [0, 1, 0]]
```

For 3D with connectivity=1:
```
     [[[0, 0, 0],      [[0, 1, 0],      [[0, 0, 0],
       [0, 1, 0],       [1, 1, 1],       [0, 1, 0],
       [0, 0, 0]],      [0, 1, 0]],      [0, 0, 0]]]
```

This structure determines which neighboring pixels are considered "touching."

---

## The Bugs We Found

### Bug 1: Hardcoded 3D Structure

**Location:** `block_face_adjacency_graph()` line ~870

**The Problem:**
```python
structure = scipy.ndimage.generate_binary_structure(3, 1)
# Always creates 3D structure!
```

When processing 2D data:
- Face arrays are 2D: shape `(2048, 2)`
- Structure is 3D: shape `(3, 3, 3)`
- **Dimension mismatch!** → `RuntimeError: structure and input must have equal rank`

**Why it happened:** The code was only tested on 3D volumetric data.

**Example:**
```python
# 2D face (what we have):
face_2d = np.array([[42, 42],
                    [137, 137]])  # Shape: (2, 2)

# 3D structure (what code tries to use):
structure_3d = np.array([[[...], [...]], [[...], [...]], [[...], [...]]])
# Shape: (3, 3, 3)

# Trying to apply 3D structure to 2D data → ERROR!
```

---

### Bug 2: Multi-Channel Dimension Mismatch

**The Problem:**

**Input:**
- Shape: `(2, 48320, 63993)` - 2 channels, Y, X
- Dimensions: 3

**Cellpose Output:**
- Shape: `(48320, 63993)` - just segmentation labels
- Dimensions: 2

**What the code expected:**
- Input and output to have same dimensions!
- Tries to write 2D output to 3D zarr → **Shape mismatch error**

**Visual:**
```
Input block:                 Cellpose Output:
┌─────────────┐             ┌─────────┐
│ Channel 0   │             │  0 0 1  │  Only 2D!
│ ○ ○ ○       │  →  →  →   │  0 1 1  │
├─────────────┤             │  2 2 2  │
│ Channel 1   │             └─────────┘
│ ● ● ●       │
└─────────────┘
3D input                    2D output

Trying to write 2D output to 3D zarr:
output_zarr[0:2, 0:2048, 0:2048] = segmentation  # ← ERROR!
#            ↑    Expected 3D shape
#                 but segmentation is 2D!
```

**Additional issues:**
1. Block indices include channel dimension: `(0, 5, 10)`
2. Crops include channel dimension: `(slice(0,2), slice(...), slice(...))`
3. Output should be 2D, but code creates 3D zarr

---

### Bug 3: List vs Tuple for Indexing

**The Problem:**

```python
# remove_overlaps() returns crop as a list:
crop_trimmed = [slice(100, 2148), slice(50, 2098)]  # list

# But zarr indexing requires tuple:
output_zarr[[slice(100, 2148), slice(50, 2098)]] = data
#           ↑ list - ERROR!

output_zarr[(slice(100, 2148), slice(50, 2098))] = data
#           ↑ tuple - WORKS!
```

**Why lists don't work:** Zarr (and numpy) treat lists as fancy indexing:
```python
arr[[1, 2, 3]]  # Select elements at indices 1, 2, 3
arr[(1, 2, 3)]  # Select element at position (1, 2, 3)
```

---

## The Patches: How We Fixed Them

### Patch 1: Dynamic Structure Dimension

**Original code:**
```python
def block_face_adjacency_graph(faces, nlabels):
    structure = scipy.ndimage.generate_binary_structure(3, 1)  # HARDCODED!
    ...
```

**Fixed code:**
```python
def fixed_block_face_adjacency_graph(faces, nlabels):
    # Determine dimensionality from actual data
    ndim = faces[0].ndim if len(faces) > 0 else 3
    structure = scipy.ndimage.generate_binary_structure(ndim, 1)
    # Now structure matches face dimensions!
    ...
```

**How it works:**
```python
# For 2D faces:
faces[0].ndim = 2
structure = generate_binary_structure(2, 1)
# Creates 2D structure: [[0,1,0], [1,1,1], [0,1,0]]

# For 3D faces:
faces[0].ndim = 3
structure = generate_binary_structure(3, 1)
# Creates 3D structure (shown earlier)
```

**Result:** Code adapts to whatever dimension data you give it!

---

### Patch 2: Multi-Channel Handling

This is more complex - we need to:
1. Track which dimension is channels
2. Create output zarr with correct 2D shape
3. Strip channel dimension from coordinates
4. Adjust block indices for output

**Part 2a: Modified `distributed_eval()`**

```python
@cluster
def fixed_distributed_eval(
    input_zarr,
    blocksize,
    write_path,
    ...
    channel_axis=None,  # NEW parameter!
):
    # Calculate output shape (without channel dimension)
    if channel_axis is not None:
        output_shape = list(input_zarr.shape)
        del output_shape[channel_axis]  # Remove channel dim
        output_shape = tuple(output_shape)
        
        output_blocksize = list(blocksize)
        del output_blocksize[channel_axis]
        output_blocksize = tuple(output_blocksize)
    else:
        output_shape = input_zarr.shape
        output_blocksize = blocksize
    
    # Example:
    # input_zarr.shape = (2, 48320, 63993)
    # blocksize = (2, 2048, 2048)
    # channel_axis = 0
    # 
    # output_shape = (48320, 63993)  ← 2D!
    # output_blocksize = (2048, 2048)  ← 2D!
    
    # Create temp zarr with correct 2D shape
    temp_zarr = zarr.open(
        temp_zarr_path, 'w',
        shape=output_shape,  # 2D, not 3D!
        chunks=output_blocksize,
        dtype=np.uint32,
    )
    
    # Pass channel info to workers
    futures = client.map(
        fixed_process_block,
        ...
        channel_axis=channel_axis,
        output_shape=output_shape,
    )
    ...
```

**Part 2b: Modified `process_block()`**

```python
def fixed_process_block(
    block_index,    # (0, 5, 10) - includes channel dim
    crop,           # (slice(0,2), slice(...), slice(...))
    ...
    channel_axis=None,
    output_shape=None,
):
    # Read 3D block and segment (Cellpose returns 2D)
    segmentation = read_preprocess_and_segment(input_zarr, crop, ...)
    # segmentation shape: (2048, 2048) - 2D
    
    # Strip channel dimension from crop
    if channel_axis is not None:
        output_crop = list(crop)
        del output_crop[channel_axis]  # Remove channel dim
        output_crop = tuple(output_crop)
        
        output_blocksize = list(blocksize)
        del output_blocksize[channel_axis]
        output_blocksize = tuple(output_blocksize)
    else:
        output_crop = crop
        output_blocksize = blocksize
    
    # Example:
    # Original crop: (slice(0,2), slice(100,2148), slice(50,2098))
    # After removal: (slice(100,2148), slice(50,2098))  ← 2D!
    
    # Remove overlaps (works with 2D data)
    segmentation, output_crop_trimmed = remove_overlaps(
        segmentation, output_crop, overlap, output_blocksize
    )
    
    # FIX: Ensure crop is tuple, not list
    if isinstance(output_crop_trimmed, list):
        output_crop_trimmed = tuple(output_crop_trimmed)
    
    # Calculate bounding boxes (2D)
    boxes = bounding_boxes_in_global_coordinates(
        segmentation, output_crop_trimmed
    )
    
    # Adjust block index (remove channel dimension)
    if channel_axis is not None:
        block_index_output = list(block_index)
        del block_index_output[channel_axis]
        block_index_output = tuple(block_index_output)
    else:
        block_index_output = block_index
    
    # Example:
    # Original block_index: (0, 5, 10) - includes channel
    # After removal: (5, 10)  ← 2D grid position!
    
    # Use 2D shape for ID calculation
    shape_for_blocks = output_shape if output_shape else input_zarr.shape
    nblocks = get_nblocks(shape_for_blocks, output_blocksize)
    
    # Assign unique IDs using 2D block index
    segmentation, remap = global_segment_ids(
        segmentation, block_index_output, nblocks
    )
    
    # Write to 2D zarr (no dimension mismatch!)
    output_zarr[output_crop_trimmed] = segmentation
    
    return faces, boxes, remap
```

**What this accomplishes:**

```
Input Processing:
┌─────────────────────┐
│ Input: (2, Y, X)    │  3D with channels
│ Block: (2, 2048, 2048)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Cellpose segments   │
│ Returns: (Y, X)     │  2D segmentation
└──────────┬──────────┘
           │
           ↓
Output Writing:
┌─────────────────────┐
│ Strip channel dim   │
│ Crop: (Y, X)        │  2D coordinates
│ Block idx: (row,col)│  2D position
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Write to 2D zarr    │
│ Output: (Y, X)      │  No mismatch!
└─────────────────────┘
```

---

### Patch 3: List to Tuple Conversion

**Simple fix:**
```python
# After remove_overlaps:
segmentation, output_crop_trimmed = remove_overlaps(...)

# Ensure it's a tuple
if isinstance(output_crop_trimmed, list):
    output_crop_trimmed = tuple(output_crop_trimmed)

# Now safe to use with zarr
output_zarr[output_crop_trimmed] = segmentation
```

---

## Putting It All Together

### Complete Workflow with Patches

**1. User Code:**
```python
import zarr
from cellpose.contrib.distributed_segmentation import distributed_eval

# Load 2-channel image
array = zarr.open('image.zarr', mode='r')
# Shape: (2, 48320, 63993)

segments, boxes = distributed_eval(
    input_zarr=array,
    blocksize=(2, 2048, 2048),  # Include all channels
    write_path='output.zarr',
    model_kwargs={'gpu': True, 'pretrained_model': 'cpsam'},
    cluster_kwargs={'n_workers': 1, 'ncpus': 12, ...},
    channel_axis=0,  # Tell it which dim is channels
)
```

**2. Inside `distributed_eval()`:**
```python
# Calculates:
# output_shape = (48320, 63993)  ← 2D
# output_blocksize = (2048, 2048)

# Creates 2D temp zarr:
temp_zarr = zarr.open(..., shape=(48320, 63993))

# Sends blocks to workers
```

**3. Worker processes each block:**
```python
# Receives 3D crop: (slice(0,2), slice(0,2108), slice(0,2108))

# Reads 3D block: (2, 2108, 2108)
image_block = input_zarr[crop]

# Cellpose returns 2D: (2108, 2108)
segmentation = cellpose_model.eval(image_block)[0]

# Converts to 2D crop: (slice(0,2108), slice(0,2108))
# Trims overlaps: (slice(60,2048), slice(60,2048))
# Ensures tuple: (slice(60,2048), slice(60,2048))

# Writes to 2D zarr - SUCCESS!
output_zarr[(slice(60,2048), slice(60,2048))] = segmentation
```

**4. Merging:**
```python
# Extracts 2D faces (not 3D)
faces = [top_edge, bottom_edge, left_edge, right_edge]
# Each face is 2D

# Creates 2D structure (not 3D) - PATCH 1
ndim = faces[0].ndim  # = 2
structure = generate_binary_structure(2, 1)

# Builds merge graph and relabels
# Saves final 2D output: (48320, 63993)
```

---

## Summary of Key Lessons

### 1. Dimension Consistency
**Lesson:** Code must handle different input/output dimensions gracefully.

**Solution:** 
- Track channel dimension explicitly
- Strip it when needed
- Adapt structures to data dimensionality

### 2. Data Structure Requirements
**Lesson:** Different libraries have different requirements (tuple vs list).

**Solution:**
- Know your library's indexing rules
- Convert data structures as needed
- Use `isinstance()` checks for safety

### 3. Parallel Processing Complexity
**Lesson:** Distributed code is harder to debug than sequential code.

**Solution:**
- Test on single blocks first (`test_mode=True`)
- Use worker logs
- Understand the data flow through each stage

### 4. Hardcoded Assumptions Are Dangerous
**Lesson:** Code hardcoded for 3D breaks on 2D data.

**Solution:**
- Use data properties to determine behavior
- Make dimensions a parameter when possible
- Write flexible code that adapts to input

---

## Debugging Tips

### Test a Single Block
```python
from cellpose.contrib.distributed_segmentation import process_block

# Test without full distribution
seg, boxes, ids = process_block(
    block_index=(0, 0),
    crop=(slice(0, 2048), slice(0, 2048)),
    input_zarr=array,
    model_kwargs=model_kwargs,
    eval_kwargs=eval_kwargs,
    blocksize=(2048, 2048),
    overlap=60,
    output_zarr=None,  # Don't need for test
    test_mode=True,  # Returns data instead of writing
    channel_axis=0,
    output_shape=(48320, 63993),
)

print(f"Segmentation shape: {seg.shape}")
print(f"Found {len(boxes)} cells")
```

### Check Shapes at Each Stage
```python
print(f"Input shape: {input_zarr.shape}")
print(f"Blocksize: {blocksize}")
print(f"Output shape: {output_shape}")
print(f"Block crop: {crop}")
print(f"Segmentation shape: {segmentation.shape}")
```

### Use the Dask Dashboard
```
# When cluster starts, you'll see:
Cluster dashboard link: http://localhost:8787

# Open in browser to watch:
# - Which workers are busy
# - Memory usage
# - Task progress
```

---

## Advanced Topics

### Why Zarr Over HDF5?
- **Parallel writes:** Multiple workers can write simultaneously
- **Cloud-friendly:** Works with S3, Google Cloud Storage
- **Compression:** Built-in compression support
- **Chunking:** Natural fit for tiled processing

### Why Overlap Matters
Without overlap, cells at edges would be cut:
```
Block 1    Block 2
┌────┐    ┌────┐
│ ○──│    │  ? │  Cell cut in half - bad segmentation!
└────┘    └────┘
```

With overlap, both blocks see the whole cell:
```
Block 1       Block 2
┌──────┐    ┌──────┐
│ ○────│────│──○   │  Both see full cell - good!
└──────┘    └──────┘
  overlap
```

### Memory Management
Each worker needs enough memory for:
- One block of input data
- Cellpose model
- Temporary arrays during processing
- Output block

Formula:
```
memory_per_worker ≈ 
    block_size_MB × 4  (input + output + intermediates)
    + 2000 MB  (Cellpose model)
```

---

## Conclusion

The distributed cellpose module is powerful but was originally designed for 3D data. By understanding:
1. How blocks are processed independently
2. How merging works across boundaries  
3. Where dimension assumptions were hardcoded
4. How to adapt the code for different input formats

We can patch it to handle 2D multi-channel images correctly. The key insights are:
- **Separate input and output dimensions** (3D input → 2D output)
- **Adapt structures to data dimensionality** (2D vs 3D)
- **Track and strip channel dimensions** where needed
- **Ensure correct data types** (tuple vs list)

These patterns apply beyond just cellpose - they're common issues in distributed image processing!
