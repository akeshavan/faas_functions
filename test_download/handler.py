import simplejson as json
import urllib
from skimage.filters import threshold_otsu
import tempfile
import os
import nibabel as nib
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import base64
import numpy as np

# The output we'll print will go here
full_output = {"log": [],
               "data": []}


def download_image(inp, out):
    urllib.urlretrieve(inp, out)
    return out


def download_and_load(base_file, outname):
    # Create a temporary spot to download images
    outdir = tempfile.mkdtemp()
    basef = os.path.join(outdir, outname)
    download_image(base_file, basef)
    # base_file = basef
    img_data = nib.load(basef)
    affine = img_data.affine
    data = img_data.get_data()
    full_output["log"].append("Downloaded and loaded {} as {}".format(base_file, basef))
    return data, affine


def reorient_array(data, aff):
    # rearrange the matrix to RAS orientation
    orientation = nib.orientations.io_orientation(aff)
    data_RAS = nib.orientations.apply_orientation(data, orientation)
    # In RAS
    return nib.orientations.apply_orientation(data_RAS,
                                    nib.orientations.axcodes2ornt("IPL"))


def mplfig(data, outfile):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(float(data.shape[1])/data.shape[0], 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect=1, cmap=cm.Greys_r)  # used to be aspect="normal"
    fig.savefig(outfile, dpi=data.shape[0])
    plt.close()


def get_tile_image(base_tile):
    outdir = tempfile.mkdtemp()
    out_base_filename = os.path.join(outdir, "tile.jpg")
    mplfig(base_tile, out_base_filename)

    with open(str(out_base_filename), 'rb') as img:
        return base64.b64encode(img.read())


def make_mask_dict(tile_data):
    tile_dict = {}
    for i in range(tile_data.shape[0]):
        for j in range(tile_data.shape[1]):
            if (int(tile_data[i, j])):
                if not i in tile_dict.keys():
                    tile_dict[i] = {}
                tile_dict[i][j] = int(tile_data[i, j])
    return tile_dict


def create_tiles(base_file, mask_file=None, slice_direction="ax",
                 mask_threshold=8000):

    # Make sure our slice direction is valid
    slicer = {"ax": 0, "cor": 1, "sag": 2}
    assert slice_direction in slicer.keys(), "slice direction must be one of {}".format(slicer.keys())

    # download and load the base image
    data, base_aff = download_and_load(base_file, "base.nii.gz")
    base = reorient_array(data, base_aff)

    # If a mask file exists, download and load it.
    if mask_file:
        mask_data, mask_aff = download_and_load(mask_file, "mask.nii.gz")
        mask = reorient_array(mask_data, mask_aff)
        use_mask = True
        # Make sure images are in the same space
        assert np.isclose(base_aff, mask_aff, rtol=1e-3, atol=1e-3).all(), "affines are not close!! {} {}".format(img_data.affine, img_mask.affine)
    else:
        # since there is no mask, we create one from the base image
        t = threshold_otsu(base)
        full_output["log"].append("No mask provided, thresholding at {}".format(t))
        mask = base > t
        use_mask = False

    all_data_slicer = [slice(None), slice(None), slice(None)]

    num_slices = base.shape[slicer[slice_direction]]
    full_output["log"].append("Total number of slices in {} direction: {}".format(slice_direction, num_slices))

    for slice_num in range(num_slices):
        all_data_slicer[slicer[slice_direction]] = slice_num
        mask_tile = mask[all_data_slicer] > 0
        base_tile = base[all_data_slicer]
        if mask_tile.sum() >= mask_threshold:
            tile_base64 = get_tile_image(base_tile)
            output_tile = {"pic": tile_base64,
                           "slice_number": slice_num,
                           "slice_direction": slice_direction}
            if use_mask:
                # This means the mask file will be a ground truth image
                mask_dict = make_mask_dict(mask_tile.T)
                output_tile["mask"] = mask_dict

            full_output["data"].append(output_tile)

    return full_output


def handle(st):
    inp = json.loads(st)

    try:
        create_tiles(**inp)
    except Exception as e:
        full_output["log"].append(e)

    print(full_output)
