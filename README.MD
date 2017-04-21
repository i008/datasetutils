# 1) BSDS500 dataset 
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

#### Usage

```python
from bsds500 import BSDS500

bsds = BSDS500(
    path_to_bsds=None, # we will download the dataset
    images_to_gray=False, 
    target_size=(256,256), 
    masks_to_binary=True
)

fileids,contours,segments,images = bsds.get_train()

```
