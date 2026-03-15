import onnxruntime as ort
import numpy as np

root = __file__.replace('\\', '/').rsplit('/', 1)[0] + '/model/'

with open(root + 'yolov11-dic80.txt', encoding='utf-8') as f:
    dict80 = f.read().split('\n')

yolo_net = ort.InferenceSession(root + 'yolo11n_one.onnx')

def detect(img, dial=640*1.414, conf=0.25, iou=0.25, topk=1024):
    inputs = {
        'res_image': img,
        'res_dial': np.array(dial, dtype=np.int32),
        'nms_conf_thr': np.array(conf, dtype=np.float32),
        'nms_iou_thr': np.array(iou, dtype=np.float32),
        'nms_top_k': np.array(topk, dtype=np.int32)
    }
    boxes, scores, clss, batch = yolo_net.run(None, inputs)
    result = []
    for box, score, cls in zip(boxes, scores, clss):
        result.append((box, dict80[int(cls)], score))
    return result

def test():
    from imageio.v2 import imread
    import matplotlib.pyplot as plt

    img = imread(root + 'testimg.png')
    result = detect(img)
    
    plt.imshow(img)
    for box, label, score in result:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', linewidth=1, alpha=0.5)
        plt.text(x1 + 2, y1 + 2, f"{label} {score:.2f}", 
                 fontsize=8, color='white', weight='bold', va='top', ha='left',
                 bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1.5))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    test()
