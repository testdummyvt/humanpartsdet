from torchvision import datasets
import torch

def test(image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    anno = target["annotations"]

    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    print(image_id)

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in anno])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return image, target

if __name__ == "__main__":


    IMAGES_PATH = "E:/datasets/testdata/valid"
    ANNOTATIONS_PATH = "E:/datasets/coco_dataset/humanparts_coco_format/person_humanparts_val2017_coco_format.json"
    dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

    image, target = dataset[0]
    image_id = dataset.ids[0]
    target = {'image_id': image_id, 'annotations': target}
    print(target)
    _, x = test(image, target)
    print(x)