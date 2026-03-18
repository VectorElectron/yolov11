import onnxruntime as ort
import numpy as np

root = __file__.replace('\\', '/').rsplit('/', 1)[0] + '/model/'

with open(root + 'yolov11-dic80.txt', encoding='utf-8') as f:
    dict80 = f.read().split('\n')

resize_net = ort.InferenceSession(root + 'yolo_resize.onnx', providers=['CPUExecutionProvider'])
yolov11_net = ort.InferenceSession(root + 'yolo11n.onnx', providers=['CPUExecutionProvider'])
nms_net = ort.InferenceSession(root + 'yolo_nms.onnx', providers=['CPUExecutionProvider'])

def detect_multi_stage(img, dial_val=640*1.414, conf=0.25, iou=0.25, topk=1024):
    dial = np.array(dial_val, dtype=np.int32)
    imgf, scale = resize_net.run(None, {'image': img, 'dial': dial})

    grid = yolov11_net.run(None, {'images': imgf})[0]

    conf_thr = np.array(conf, dtype=np.float32)
    iou_thr = np.array(iou, dtype=np.float32)
    top_k = np.array(topk, dtype=np.int32)
    
    boxes, scores, clss, batch = nms_net.run(None, {
        'grid': grid, 
        'conf_thr': conf_thr, 
        'iou_thr': iou_thr, 
        'top_k': top_k, 
        'scale': scale
    })

    result = []
    for box, score, cls in zip(boxes, scores, clss):
        result.append((box, dict80[int(cls)], score))
    return result

def test():
    from imageio.v2 import imread
    import matplotlib.pyplot as plt
    
    img = imread(root + 'testimg.png')
    result = detect_multi_stage(img)
    
    plt.imshow(img)
    for (x1, y1, x2, y2), label, score in result:
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', linewidth=1, alpha=0.5)
        plt.text(x1 + 2, y1 + 2, f"{label} {score:.2f}", 
                 fontsize=8, color='white', weight='bold', va='top', ha='left',
                 bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1.5))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    test()
