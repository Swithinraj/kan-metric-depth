"""Convert the NYU Depth V2 official validation tars (654 images) into
<root>/nyu_test/{image,depth}/ so ImageDepthClips can read them directly.
"""

import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

RAW_DIR = Path("datasets/nyu_v2_val/raw/data")
OUT_ROOT = Path("datasets/nyu_v2_val/nyu_test")


def main() -> None:
    image_dir, depth_dir = OUT_ROOT / "image", OUT_ROOT / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for shard in sorted(RAW_DIR.glob("val-*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".h5"):
                    continue
                stem = Path(member.name).stem
                data = tar.extractfile(member).read()
                with h5py.File(io.BytesIO(data), "r") as h5:
                    rgb = np.asarray(h5["rgb"]).transpose(1, 2, 0)
                    depth = np.asarray(h5["depth"], dtype=np.float32)
                Image.fromarray(rgb).save(image_dir / f"image_{stem}.png")
                np.save(depth_dir / f"depth_{stem}.npy", depth)
                count += 1
    print(f"wrote {count} image/depth pairs to {OUT_ROOT}")


if __name__ == "__main__":
    main()
