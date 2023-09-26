import os.path as osp
import mmcv
import scipy.io as sio
import glob
import numpy as np
import supervision as sv

from mmengine.fileio import dump

def extract_boxes(mask: np.ndarray) -> np.ndarray:
    """Compute bounding box from mask of one individual cell.

    Parameters
    ----------
    mask: np.array
        Mask of size [height, width] where pixels are either 1 or 0.

    Returns
    -------
    box: np.array
        [ymin, ymax, xmin, xmax]. Here, ymax and xmax are INCLUDED in the bounding
        box
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]

    xmin, xmax = horizontal_indicies[[0, -1]]
    ymin, ymax = vertical_indicies[[0, -1]]
    box = np.array([ymin, ymax, xmin, xmax])

    return box


def convert_consep_to_coco(out_file,
                           annot_prefix="labels",
                           image_prefix="tiles",
                           mode='train',
                           dir="/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112"):

    available_tiles = []
    for elt in glob.glob(dir + "/labels/*.mat"):
        filename = elt.split("/")[-1]
        filename = filename.split(".")[0]
        prefix_1 = filename.split("_")[0]
        prefix_2 = filename.split("_")[1]
        if mode == "test" and prefix_1 == "test":
            available_tiles.append(filename)
        if mode == "train" and prefix_1 == "train" and int(prefix_2) <= 18:
            available_tiles.append(filename)
        if mode == "val" and prefix_1 == "train" and int(prefix_2) > 18:
            available_tiles.append(filename)

    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(available_tiles):
        annotation = sio.loadmat(f"{dir}/{annot_prefix}/{filename}.mat")
        img_path = osp.join(dir, image_prefix, f"{filename}.png")
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=f"{filename}.png", height=height, width=width))

        for obj in range(int(annotation["n_cells_str"][0])):
            instance_mask: np.ndarray = np.equal(annotation["instance_map"], obj + 1)
            [y_min, y_max, x_min, x_max] = extract_boxes(instance_mask)
            try:
                coords = sv.mask_to_polygons(instance_mask)[0]
            except:
                # print(idx, filename, obj)
                continue
            poly = [[l[0], l[1]] for l in coords]
            # coords = np.argwhere(instance_mask)
            # poly = [[l[1], l[0]] for l in coords]
            poly = [p for x in poly for p in x]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = (x_max - x_min) * (y_max - y_min)

            if area == 0 or len(poly) <= 12:
                continue
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=annotation["class_labels"][0, obj].astype(int),
                bbox=bbox,
                area=area,
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[
            {
            'id': 1,
            'name': 'other'
        },
            {
                'id': 2,
                'name': 'inflammatory'
            },
            {
                'id': 3,
                'name': 'healthy epithelial'
            },
            {
                'id': 4,
                'name': 'dysplastic/malignant epithelial'
            },
            {
                'id': 5,
                'name': 'fibroblast'
            },
            {
                'id': 6,
                'name': 'muscle'
            },
            {
                'id': 7,
                'name': 'endothelial'
            },
        ])
    dump(coco_format_json, out_file)


if __name__ == '__main__':
    convert_consep_to_coco(out_file='/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/new_train/annotation_coco.json',
                           mode='train',
                           )
    convert_consep_to_coco(out_file='/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/new_val/annotation_coco.json',
                           mode='val',
                           )
    convert_consep_to_coco(out_file='/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/new_test/annotation_coco.json',
                        mode='test',
                        )