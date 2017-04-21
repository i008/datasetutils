# 1) BSDS500 dataset 
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

#### Usage

```python
from bsds500 import BSDS500

bsds = BSDS500(target_size=(256, 256))
fileids,contours,segments,images = bsds.get_train()

```
